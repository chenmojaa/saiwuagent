"""正文变更自动重建（notes_watch）的测试。

背景：入库时正文会落一份副本到 data/notes/<id>.md。用户/脚本直接改这份文件
是最自然的更新方式，但系统原来完全不知道它变了 —— 本地文件来源既不记 mtime
也不记 hash，所以没有任何依据判断内容是否被改过。这里靠 content_hash 指纹补齐。

重点锁三件事：
  1. **升级安全**：老数据 content_hash 为 NULL，首轮扫描只回填指纹、**不重建**。
     否则一次升级会把整个知识库重新 embedding 一遍，既慢又白花 API 钱。
  2. **只对真实变更重建**：内容没变不调 embedding（不花钱）；
     CRLF/末尾换行这类无意义差异也不该触发。
  3. **失败可重试**：embedding 失败时不更新指纹，下轮会再来一次；
     同时旧索引原封不动，笔记仍然可检索。

全部离线：embedding 用假向量，数据落临时目录。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp(prefix="hd_watch_test_")
os.environ["DATA_DIR"] = _TMP
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "notes.db")
os.environ["NOTES_DIR"] = os.path.join(_TMP, "notes")
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")


def _patch_embed(fail=False):
  """三处都要打：入库、重建共用实现、以及 endpoints。"""
  import app.tools.ingest as I
  import app.tools.reindex as R
  import app.api.notes as N

  def fake(chunks, **kw):
    if fail:
      raise RuntimeError("模拟 embedding 服务不可用")
    return [[0.1, 0.2, 0.3] for _ in chunks]

  I.embed_texts = fake
  R.embed_texts = fake
  N.embed_texts = fake


def _make_note(content: str, title: str = "测试笔记", source_type: str = "text"):
  from app.tools.ingest import _ingest
  return _ingest(title=title, content=content, source_type=source_type)


def _vectors(note_id: str) -> list:
  from app.storage.vector import get_collection
  return get_collection().get(where={"note_id": note_id})["ids"]


def _db_note(note_id: str):
  from sqlmodel import Session
  from app.storage.db import Note, get_engine
  with Session(get_engine()) as s:
    return s.get(Note, note_id)


def _edit_file(note, text: str) -> None:
  with open(note.content_path, "w", encoding="utf-8") as f:
    f.write(text)


def _clear_hash(note_id: str) -> None:
  """模拟「升级前入库的老数据」：列是新的，值为 NULL。"""
  from sqlmodel import Session
  from app.storage.db import Note, get_engine
  with Session(get_engine()) as s:
    n = s.get(Note, note_id)
    n.content_hash = None
    s.add(n)
    s.commit()


# ============ 1. 升级安全：首轮只回填，不重建 ============

def test_first_scan_backfills_without_reindexing():
  """content_hash 为 NULL 时只回填指纹 —— 不能把整个知识库重 embedding 一遍。

  这是最贵的一个坑：升级后如果直接判「hash 不一致」就重建，
  1000 条笔记就是 1000 次 embedding 调用。
  """
  _patch_embed()
  note = _make_note("安徽文旅：黄山门票 190 元。")
  _clear_hash(note.id)
  assert _db_note(note.id).content_hash is None

  import app.notes_watch as W
  stats = W.scan_once()

  assert stats["backfilled"] == 1, stats
  assert stats["reindexed"] == 0, "首轮不该重建任何笔记: %s" % stats
  assert _db_note(note.id).content_hash is not None, "指纹应被回填"


def test_second_scan_after_backfill_is_noop():
  """回填之后再扫一遍：没改内容就不该有任何动作（也不调 embedding）。"""
  _patch_embed()
  note = _make_note("内容没变。")
  _clear_hash(note.id)

  import app.notes_watch as W
  W.scan_once()                    # 回填
  before = _db_note(note.id).content_hash
  stats = W.scan_once()            # 再扫

  assert stats["reindexed"] == 0, stats
  assert stats["backfilled"] == 0, stats
  assert _db_note(note.id).content_hash == before


# ============ 2. 真实变更才重建 ============

def test_edited_file_triggers_reindex():
  """直接改 .md -> 下一轮扫描自动重建，新内容进索引。"""
  _patch_embed()
  note = _make_note("黄山门票 190 元。")

  _edit_file(note, "黄山门票 190 元。\n\n新增：索道票价 80 元，6 岁以下免票。")

  import app.notes_watch as W
  stats = W.scan_once()

  assert stats["reindexed"] == 1, stats
  from app.storage.vector import get_collection
  docs = " ".join(get_collection().get(where={"note_id": note.id})["documents"])
  assert "索道" in docs, "新内容没进索引: %s" % docs[:200]
  assert "免票" in docs


def test_reindex_keeps_note_id_and_updates_hash():
  _patch_embed()
  note = _make_note("原始内容。")
  nid = note.id
  old_hash = _db_note(nid).content_hash

  _edit_file(note, "改过的内容，完全不同。")
  import app.notes_watch as W
  W.scan_once()

  assert _db_note(nid) is not None, "note_id 必须保持不变"
  assert _db_note(nid).content_hash != old_hash, "指纹应更新，否则会反复重建"
  assert _db_note(nid).embedded is True


def test_unchanged_content_does_not_reindex():
  """没改内容 -> 不调 embedding（不花钱）。"""
  _patch_embed()
  note = _make_note("稳定内容。")
  import app.notes_watch as W
  for _ in range(3):
    stats = W.scan_once()
    assert stats["reindexed"] == 0, stats


def test_cosmetic_differences_do_not_trigger_reindex():
  """CRLF / 末尾换行 / 首尾空白 不算内容变更 —— 否则编辑器一保存就白花钱。"""
  _patch_embed()
  note = _make_note("黄山门票 190 元。")

  # 编辑器常见的无意义改动
  _edit_file(note, "黄山门票 190 元。\r\n")
  import app.notes_watch as W
  stats = W.scan_once()
  assert stats["reindexed"] == 0, "CRLF/末尾换行不该触发重建: %s" % stats

  _edit_file(note, "  黄山门票 190 元。  ")
  stats = W.scan_once()
  assert stats["reindexed"] == 0, "首尾空白不该触发重建: %s" % stats


# ============ 3. 失败可重试、不毁数据 ============

def test_embedding_failure_keeps_hash_so_next_scan_retries():
  """失败时不更新指纹 -> 下轮重试；旧索引不动，笔记仍可检索。"""
  _patch_embed()
  note = _make_note("重要的原始内容。")
  before_vectors = len(_vectors(note.id))
  before_hash = _db_note(note.id).content_hash

  _edit_file(note, "改过的内容，但这次 embedding 会失败。")
  _patch_embed(fail=True)

  import app.notes_watch as W
  stats = W.scan_once()

  assert stats["failed"] == 1, stats
  assert stats["reindexed"] == 0
  assert _db_note(note.id).content_hash == before_hash, \
    "失败时不能更新指纹，否则这轮改动永远不会被重试"
  assert len(_vectors(note.id)) == before_vectors, "失败时不该动旧索引"

  # 恢复之后下一轮应该能补上
  _patch_embed()
  stats2 = W.scan_once()
  assert stats2["reindexed"] == 1, "恢复后应能重试成功: %s" % stats2


def test_missing_file_is_skipped_not_fatal():
  """文件不在了 -> 跳过，不能让整轮扫描崩掉。"""
  _patch_embed()
  note = _make_note("内容。")
  os.remove(note.content_path)

  import app.notes_watch as W
  stats = W.scan_once()
  assert stats["skipped"] >= 1, stats
  assert stats["failed"] == 0, stats


def test_scan_never_raises():
  """整体不抛异常 —— 后台旁路坏掉不该影响服务。"""
  import app.notes_watch as W
  stats = W.scan_once()
  assert isinstance(stats, dict)
  for k in ("checked", "backfilled", "reindexed", "failed", "skipped"):
    assert k in stats, "缺少统计字段 %s" % k


# ============ 4. 覆盖范围 ============

def test_feishu_notes_are_skipped():
  """飞书来源有自己的增量同步，这里不要重复管（否则会双重重建）。"""
  _patch_embed()
  note = _make_note("飞书文档内容。", source_type="feishu_docx")
  _edit_file(note, "改过的飞书内容。")

  import app.notes_watch as W
  stats = W.scan_once()
  assert stats["reindexed"] == 0, "飞书来源不该被本地扫描重建: %s" % stats
  assert stats["skipped"] >= 1


def test_enabled_reflects_config():
  import app.notes_watch as W
  from app.config import settings
  orig = settings.notes_autoreindex_interval_min
  try:
    object.__setattr__(settings, "notes_autoreindex_interval_min", 0)
    assert W.enabled() is False
    object.__setattr__(settings, "notes_autoreindex_interval_min", 5)
    assert W.enabled() is True
  finally:
    object.__setattr__(settings, "notes_autoreindex_interval_min", orig)


def main() -> int:
  names = sorted(k for k in list(globals()) if k.startswith("test_"))
  passed = failed = 0
  for name in names:
    try:
      globals()[name]()
      passed += 1
      print("PASS %s" % name)
    except AssertionError as e:
      print("FAIL %s: %s" % (name, e))
      failed += 1
    except Exception as e:
      print("ERROR %s: %s: %s" % (name, type(e).__name__, e))
      failed += 1
  print("=" * 40)
  print("Passed: %d/%d  Failed: %d" % (passed, len(names), failed))
  return 0 if failed == 0 else 1


if __name__ == "__main__":
  raise SystemExit(main())
