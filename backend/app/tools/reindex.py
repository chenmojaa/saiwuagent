# -*- coding: utf-8 -*-
"""笔记索引重建 —— **唯一**的「用新内容替换旧索引」实现。

## 为什么必须只有一份

这段逻辑原来被复制了两份，然后两份写出了不同的行为：

* ``feishu_sync._drop_and_reingest``：**先删旧再写新** —— 正确
* ``api.notes.api_reembed_note``：**先 add 再 delete** —— 错的

后者的问题：``add_chunks`` 用的向量 id 是 ``{note_id}_c{i}``，新旧**完全同 id**；
而 ``delete_note_chunks`` 是按 ``note_id`` 删的，于是**刚写入的新向量被自己删掉**。
接口返回 200，笔记却变成永久检索不到的孤儿，DB 还标记着 embedded=True ——
静默数据丢失，比报错难发现得多。

所以这里把它收敛成一份，谁要重建索引都走这里。

## 顺序（三步都不能调换）

1. **先算 embedding** —— 失败时原索引完全不动，笔记仍然可检索。
2. **再删旧** —— 必须在新向量写入**之前**。反过来会把自己写的删掉（见上）。
3. **最后写新**。

并且 ``embedded`` 必须反映**真实结果**，不能无条件写 True。否则写入失败时
用户看到「已索引」，实际什么都搜不到。
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlmodel import Session as SqlSession

from app.config import settings
from app.embeddings.factory import embed_texts
from app.storage.db import Note, get_engine
from app.storage.vector import add_chunks, delete_note_chunks
from app.tools.chunk import chunk_text

_log = logging.getLogger(__name__)

# 按 note_id 串行化重建。
#
# 为什么必须加：这个函数会被两个来源并发调用 —— 后台扫描线程（notes_watch）
# 和用户请求（PATCH /content、reembed）。两边的流程都是
# 「删旧 -> 写新 -> 更新行」，交错时会坏成这样：
#
#   A 删、B 删、B 写（5 段）、A 写（2 段）
#   -> Chroma 里留下 5 个向量（c0-c2 来自 A，c3-c4 来自 B），
#      而 chunk_count 被 A 写成 2，content_hash 是 A 的
#
# 结果是**索引与正文/指纹不一致**，而且不会报错：扫描器看到指纹匹配就不再
# 重试，用户搜到的是混在一起的旧内容。和 MCPSession 那个串线问题同一类。
#
# 按 note 加锁而不是全局一把锁：后台一轮要扫很多篇，全局锁会让用户的
# PATCH 排在一整轮扫描后面。
#
# 用 RLock：将来若有人在持锁路径里再调本函数（比如批量重建）不会自锁。
_note_locks: dict[str, threading.RLock] = {}
_note_locks_guard = threading.Lock()


def _lock_for(note_id: str) -> threading.RLock:
  """取（或建）某篇笔记的锁。

  锁对象按 note_id 常驻，不主动回收 —— 每把锁几十字节，几千篇笔记也才
  几百 KB，不值得为它引入引用计数或弱引用的复杂度。
  """
  with _note_locks_guard:
    lk = _note_locks.get(note_id)
    if lk is None:
      lk = threading.RLock()
      _note_locks[note_id] = lk
    return lk


@dataclass
class ReindexResult:
  """重建结果。调用方据此决定返回什么状态码 / 要不要推进 revision。"""

  note: Note | None = None
  ok: bool = False                 # 索引是否真的重建成功（向量已落库）
  embedded: bool = False
  chunk_count: int = 0
  embedding_failed: bool = False   # 是「算 embedding 失败」还是「写索引失败」
  reason: str = ""


def content_hash(text: str) -> str:
  """正文的内容指纹（sha1）。

  用来判断「磁盘上的 .md 被改过没有」—— 本地文件来源没有远端版本号可比，
  只能靠内容本身。飞书那条走 source_revision，不依赖这个。

  归一化：统一换行 + 去掉首尾空白。否则编辑器把 CRLF 换成 LF、或者末尾多一个
  换行，都会被判成「内容变了」而触发一次无谓的重新 embedding（要花钱的）。
  """
  norm = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
  return hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()


def _write_content_file(note: Note, content: str) -> None:
  """把正文写到磁盘。文件丢了就重建（content_path 可能是旧的临时路径）。"""
  try:
    path = note.content_path
    if not path or not os.path.exists(path):
      os.makedirs(settings.notes_dir, exist_ok=True)
      path = os.path.join(settings.notes_dir, note.id + ".md")
    with open(path, "w", encoding="utf-8") as f:
      f.write(content)
    note.content_path = path
  except Exception as e:
    # 正文写盘失败不该阻断索引重建 —— 索引才是检索用的，磁盘文件只是留档。
    _log.warning("reindex: 写正文失败 note=%s: %s", note.id, e)


def replace_note_index(
  note_id: str,
  content: str,
  *,
  title: str | None = None,
  source_type: str | None = None,
  source_revision: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  write_content: bool = True,
  keep_old_on_embed_failure: bool = False,
) -> ReindexResult:
  """用 ``content`` 替换 ``note_id`` 的正文与索引，**保持 note id 不变**。

  keep_old_on_embed_failure：
    * False（默认，用于手动重建 / 编辑内容）—— 算 embedding 失败就整体放弃，
      原索引原封不动，调用方报错让用户重试。
    * True（用于后台同步，如飞书）—— 正文先写盘、标题照更新，但**保留旧向量**
      且**不推进 revision**，这样下一轮同步会重试，而不是把笔记永久留在
      未索引状态。

  ``write_content=False`` 用于「正文没变、只想重建索引」的场景（reembed），
  避免无意义地重写文件。

  **按 note_id 串行执行**（见 _lock_for 的说明）—— 后台扫描和用户请求会
  并发调到这里，不串行会写出「索引与指纹不一致」且不报错的坏状态。
  """
  with _lock_for(note_id):
    return _replace_note_index_locked(
      note_id, content,
      title=title, source_type=source_type, source_revision=source_revision,
      api_key=api_key, base_url=base_url, embedding_model=embedding_model,
      write_content=write_content,
      keep_old_on_embed_failure=keep_old_on_embed_failure,
    )


def _replace_note_index_locked(
  note_id: str,
  content: str,
  *,
  title: str | None = None,
  source_type: str | None = None,
  source_revision: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  write_content: bool = True,
  keep_old_on_embed_failure: bool = False,
) -> ReindexResult:
  """真正的实现。调用方必须已持有 _lock_for(note_id)。"""
  with SqlSession(get_engine()) as s:
    note = s.get(Note, note_id)
    if note is None:
      return ReindexResult(reason="note 不存在")

  chunks = chunk_text(content) if content else []

  # ---- 1. 先算 embedding（不碰索引）----
  embeddings: list = []
  if chunks:
    try:
      embeddings = embed_texts(chunks, api_key=api_key, base_url=base_url,
                               model=embedding_model)
    except Exception as e:
      _log.warning("reindex: embedding 失败 note=%s: %s", note_id, e)
      if not keep_old_on_embed_failure:
        # 索引一个字节都没动，笔记保持原样可用。
        return ReindexResult(note=note, embedding_failed=True,
                             reason="%s: %s" % (type(e).__name__, e))
      # 后台同步路径：正文先落地，旧向量留着，revision 不推进 -> 下轮重试。
      with SqlSession(get_engine()) as s:
        n = s.get(Note, note_id)
        if n is not None:
          if write_content:
            _write_content_file(n, content)
          if title:
            n.title = title[:500]
          n.word_count = len(content)
          if source_type:
            n.source_type = source_type
          n.embedded = False
          s.add(n)
          s.commit()
          s.refresh(n)
          note = n
      return ReindexResult(note=note, embedding_failed=True, embedded=False,
                           reason="%s: %s" % (type(e).__name__, e))

  # ---- 2. 先删旧（必须在新向量写入之前）----
  try:
    delete_note_chunks(note_id)
  except Exception as e:
    # 删不掉旧的至少还能写新的：新旧同 id，写入时会覆盖旧的。
    _log.warning("reindex: delete_note_chunks(%s) 失败: %s", note_id, e)

  # ---- 3. 最后写新 ----
  n_chunks = 0
  embedded = False
  if chunks and embeddings:
    try:
      n_chunks = add_chunks(note_id, chunks, embeddings)
      embedded = True
    except Exception as e:
      _log.warning("reindex: add_chunks(%s) 失败: %s", note_id, e)

  with SqlSession(get_engine()) as s:
    n = s.get(Note, note_id)
    if n is None:
      return ReindexResult(note=note, ok=False, reason="note 在重建过程中消失")
    if write_content:
      _write_content_file(n, content)
    if title:
      n.title = title[:500]
    n.word_count = len(content)
    n.chunk_count = n_chunks
    n.embedded = embedded
    if source_type:
      n.source_type = source_type
    if embedded and chunks:
      n.summary = chunks[0][:200]
    # 只有索引真的建好才更新指纹：否则「内容变了但没索引成功」这个状态会被
    # 指纹抹掉，本地文件扫描再也不会重试它。
    if embedded:
      n.content_hash = content_hash(content)
    # revision 只在索引真的建好之后才推进 —— 否则后台同步会以为这条已经处理过，
    # 而它其实是未索引状态，永远不会被重试。
    #
    # 这里直接写字段而不是调 update_note_revision()：那个函数会另开一个 session
    # 再提交一次，导致本函数返回的 note 对象停留在推进 revision 之前的快照
    # （调用方拿到的 source_revision 是旧的）。同一个事务里写完最省事也最一致。
    if source_revision and embedded:
      n.source_revision = source_revision
      n.source_updated_at = datetime.now(timezone.utc)
    s.add(n)
    s.commit()
    s.refresh(n)
    note = n

  return ReindexResult(note=note, ok=embedded, embedded=embedded,
                       chunk_count=n_chunks,
                       reason="" if embedded else "向量库写入未成功")
