# -*- coding: utf-8 -*-
"""正文变更扫描 —— 直接编辑 data/notes/<id>.md 后自动重建索引。

## 为什么需要

入库时正文会落一份副本到 ``data/notes/<id>.md``（Note.content_path）。
用户/脚本直接改这份文件是最自然的更新方式，但系统原来**完全不知道它变了** ——
本地文件来源既不记 mtime 也不记 hash（只有飞书记 source_revision），
所以没有任何依据判断「内容是否被改过」。

这里给 Note 加 ``content_hash`` 指纹，周期性比对磁盘文件：
不一致就走 ``tools/reindex.py`` 重建（切分 + 向量 + FTS），note_id 保持不变。

## 覆盖范围

只扫**本地来源**（跳过 ``feishu_*``）。飞书有自己的增量同步
（``feishu_sync.py`` 的 source_revision），两边都管会重复重建。

## 升级安全（重要）

``content_hash`` 是新增列，老数据是 NULL。首轮扫描遇到 NULL 只**回填指纹**、
**不重建索引** —— 否则一次升级会把整个知识库重新 embedding 一遍，
既慢又白花 API 钱。回填之后，后续真实的改动才触发重建。

## 失败语义

embedding 失败 -> 不更新指纹（下次扫描会重试），旧索引原封不动，笔记仍可检索。
本模块**绝不抛异常**：它是后台旁路，坏掉不该影响服务。
"""
from __future__ import annotations

import logging
import os

from sqlmodel import Session as SqlSession, select

from app.config import settings
from app.storage.db import Note, get_engine
from app.tools.reindex import content_hash, replace_note_index

_log = logging.getLogger(__name__)

# 飞书来源有自己的增量同步（source_revision），这里不要重复管。
_SKIP_SOURCE_PREFIX = "feishu_"


def _read_text(path: str) -> str | None:
  try:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
      return f.read()
  except OSError as e:
    _log.info("notes_watch: 读不到 %s: %s", path, e)
    return None


def scan_once(api_key: str | None = None, base_url: str | None = None,
              embedding_model: str | None = None) -> dict:
  """扫描一轮。返回 {checked, backfilled, reindexed, failed, skipped}。"""
  stats = {"checked": 0, "backfilled": 0, "reindexed": 0,
           "failed": 0, "skipped": 0}
  try:
    with SqlSession(get_engine()) as s:
      notes = list(s.exec(select(Note)).all())
  except Exception as e:
    _log.warning("notes_watch: 读取笔记列表失败: %s", e)
    return stats

  for note in notes:
    try:
      src = (note.source_type or "")
      if src.startswith(_SKIP_SOURCE_PREFIX):
        stats["skipped"] += 1
        continue
      path = note.content_path
      if not path or not os.path.isfile(path):
        stats["skipped"] += 1
        continue

      text = _read_text(path)
      if text is None:
        stats["failed"] += 1
        continue

      stats["checked"] += 1
      disk_hash = content_hash(text)

      # ---- 首次（升级后回填）：只记指纹，不重建 ----
      if not note.content_hash:
        _set_hash(note.id, disk_hash)
        stats["backfilled"] += 1
        _log.info("notes_watch: 回填指纹 %s（不重建）", note.id)
        continue

      if disk_hash == note.content_hash:
        continue

      _log.info("notes_watch: 检测到正文变更 %s，重建索引", note.id)
      # write_content=False：磁盘上已经是新内容了，不必回写。
      res = replace_note_index(
        note.id, text,
        api_key=api_key, base_url=base_url, embedding_model=embedding_model,
        write_content=False,
      )
      if res.embedded:
        stats["reindexed"] += 1
        _log.info("notes_watch: %s 重建完成（%d 段）", note.id, res.chunk_count)
      else:
        # 指纹不更新 -> 下轮继续重试。旧索引还在，笔记仍可检索。
        stats["failed"] += 1
        _log.warning("notes_watch: %s 重建失败，下轮重试: %s", note.id, res.reason)
    except Exception as e:
      # 单条出错不能中断整轮扫描。
      stats["failed"] += 1
      _log.warning("notes_watch: 处理 %s 异常: %s", getattr(note, "id", "?"), e)

  return stats


def _set_hash(note_id: str, value: str) -> None:
  try:
    with SqlSession(get_engine()) as s:
      n = s.get(Note, note_id)
      if n is not None:
        n.content_hash = value
        s.add(n)
        s.commit()
  except Exception as e:
    _log.warning("notes_watch: 写指纹失败 %s: %s", note_id, e)


def enabled() -> bool:
  """扫描是否启用（间隔 > 0）。"""
  try:
    return int(settings.notes_autoreindex_interval_min) > 0
  except (TypeError, ValueError):
    return False
