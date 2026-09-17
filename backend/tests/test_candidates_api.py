"""候选库 API 的集成测试（策略 A 防污染边界的端到端验证）。

跑法：python tests/test_candidates_api.py

用独立的临时 data_dir，不碰真实知识库。embedding 被替换成确定性假向量，
所以离线可跑，也不会因为没配 API key 而失败。

覆盖的核心断言：
  1. 联网结果落库后是 pending，**主知识库不变**（这是防污染的关键）。
  2. approve 之后主知识库才有 Note，且候选状态变 approved 并回指 note_id。
  3. reject 只改状态，主知识库永远不受影响。
  4. 同一网页重复入候选被去重。
  5. 入库失败时候选**保持 pending**（不能出现「状态已批准但知识没进去」）。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _setup_env():
  """必须在 import app.* 之前设好路径，settings 是 lru_cache 的。

  **踩过的坑**：这几个路径字段是**四个独立字段**（data_dir / sqlite_path /
  notes_dir / chroma_dir），都默认 ./data，没有共同前缀，config.py 里也没用
  _hd() 包一层。所以只设 HD_DATA_DIR 完全无效（那个变量名根本不存在），
  测试会直接往真实的 backend/data/notes.db 写候选 —— 实测污染了 22 条。
  必须逐个设成同一临时目录。
  """
  tmp = tempfile.mkdtemp(prefix="hd_cand_test_")
  os.environ["DATA_DIR"] = tmp
  os.environ["SQLITE_PATH"] = os.path.join(tmp, "notes.db")
  os.environ["NOTES_DIR"] = os.path.join(tmp, "notes")
  os.environ["CHROMA_DIR"] = os.path.join(tmp, "chroma")
  return tmp


_DATA_DIR = _setup_env()


def _assert_isolated():
  """防回归：确认真的在临时目录里跑，别又写进开发库。"""
  from app.config import settings
  real = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
  got = os.path.abspath(settings.sqlite_path)
  if got.startswith(real):
    raise AssertionError(
      "测试未隔离！sqlite_path=%s 落在开发库 %s 内，会污染真实数据" % (got, real))
  return got

from fastapi.testclient import TestClient  # noqa: E402


def _fake_embed_patches():
  """把 embedding 换成确定性假实现，避免联网 + 保证可复现。"""
  import app.tools.ingest as I
  orig_embed, orig_add = I.embed_texts, I.add_chunks
  I.embed_texts = lambda chunks, **kw: [[0.1, 0.2, 0.3] for _ in chunks]
  I.add_chunks = lambda note_id, chunks, embs: len(chunks)
  return I, orig_embed, orig_add


def _raw_client():
  from app.main import app
  return TestClient(app)


def _client():
  """带鉴权头的客户端。TestClient 支持在构造时注入默认 headers。"""
  return TestClient(_raw_client().app, headers=_auth())


_AUTH_HEADERS: dict | None = None


def _auth() -> dict:
  """注册（或登录）一个测试账号，返回鉴权头。

  所有 /api 路由都在 _require_auth 中间件后面（只有 /api/auth/* 与
  /api/health 放行），所以不带 token 一律 401。
  注意登录用的是 `account` 字段（不是 phone），注册成功则直接返回 token。
  """
  global _AUTH_HEADERS
  if _AUTH_HEADERS:
    return _AUTH_HEADERS
  c = _raw_client()
  creds = {"phone": "13800138000", "password": "Test1234!"}
  r = c.post("/api/auth/register", json=creds)
  token = None
  if r.status_code in (200, 201):
    token = (r.json() or {}).get("token")
  if not token:
    # 已注册过：走登录。account 可以是手机号。
    r = c.post("/api/auth/login", json={"account": creds["phone"],
                                        "password": creds["password"]})
    if r.status_code != 200:
      raise AssertionError("登录失败: %s %s" % (r.status_code, r.text))
    token = (r.json() or {}).get("token") or (r.json() or {}).get("access_token")
  if not token:
    raise AssertionError("未能拿到 token: %s" % r.text)
  _AUTH_HEADERS = {"Authorization": "Bearer " + token}
  return _AUTH_HEADERS


def _kb_note_count() -> int:
  from app.storage.db import Note, get_session
  from sqlmodel import select
  with get_session() as s:
    return len(s.exec(select(Note)).all())


def _seed_raw(url="https://example.com/a", text="腾讯 2025 年营收 6600 亿元", title="财报页"):
  """写一条候选，返回新建 id 列表（可能是空的，用于去重测试）。"""
  from app.storage import candidates as store
  return store.add_candidates(
    [{"source_url": url, "title": title, "text": text}], query="腾讯营收")


def _seed(url="https://example.com/a", text="腾讯 2025 年营收 6600 亿元", title="财报页"):
  """写一条候选并断言成功。

  断言很重要：如果 add_candidates 因为去重/异常返回 []，测试会静默地拿空
  列表往下跑，最后变成"空对空"的假通过（本文件早期版本就出现过 ——
  batch_approve 用 ids=[] 断言通过，实际什么都没验证）。
  """
  ids = _seed_raw(url, text, title)
  assert ids, "候选应写入成功（url=%s）" % url
  return ids


def test_candidate_lands_pending_without_touching_kb():
  """核心边界：候选入库后是 pending，主知识库**一条都不能多**。"""
  before = _kb_note_count()
  ids = _seed()
  assert ids, "候选应写入成功"
  from app.storage import candidates as store
  row = store.get_candidate(ids[0])
  assert row["status"] == "pending", row
  assert _kb_note_count() == before, "候选落库阶段绝不能写主知识库"


def test_list_and_stats_endpoints():
  c = _client()
  r = c.get("/api/candidates")
  assert r.status_code == 200, r.text
  body = r.json()
  assert "items" in body and "counts" in body
  assert body["counts"]["pending"] >= 1

  r2 = c.get("/api/candidates/stats")
  assert r2.status_code == 200
  assert r2.json()["pending"] >= 1


def test_approve_writes_to_kb_and_marks_approved():
  """approve 是联网内容进主库的唯一通道。"""
  I, orig_embed, orig_add = _fake_embed_patches()
  try:
    ids = _seed("https://example.com/approve", "待批准的内容 12345")
    cid = ids[0]
    before = _kb_note_count()

    r = _client().post("/api/candidates/%d/approve" % cid, json={})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["ok"] is True
    assert payload["note_id"], "approve 应返回生成的 note_id"
    assert _kb_note_count() == before + 1, "approve 后主知识库应新增一条"

    from app.storage import candidates as store
    row = store.get_candidate(cid)
    assert row["status"] == "approved", row
    assert row["note_id"] == payload["note_id"], "候选应回指生成的 Note"
    assert row["reviewed_at"], "应记录审核时间"

    from app.storage.db import Note, get_session
    with get_session() as s:
      note = s.get(Note, payload["note_id"])
    assert note is not None
    assert note.source_url == "https://example.com/approve", "来源 URL 必须保留，便于溯源"
    assert note.source_type == "url"
  finally:
    I.embed_texts, I.add_chunks = orig_embed, orig_add


def test_approve_twice_is_rejected():
  """重复批准要挡住，否则会重复入库。"""
  I, orig_embed, orig_add = _fake_embed_patches()
  try:
    ids = _seed("https://example.com/dup-approve", "重复批准测试")
    cid = ids[0]
    assert _client().post("/api/candidates/%d/approve" % cid, json={}).status_code == 200
    r2 = _client().post("/api/candidates/%d/approve" % cid, json={})
    assert r2.status_code == 409, r2.text
  finally:
    I.embed_texts, I.add_chunks = orig_embed, orig_add


def test_reject_never_touches_kb():
  """驳回只改状态，主知识库不受影响。"""
  ids = _seed("https://example.com/reject", "应当被驳回的内容")
  cid = ids[0]
  before = _kb_note_count()
  r = _client().post("/api/candidates/%d/reject" % cid, json={"note": "来源不可靠"})
  assert r.status_code == 200, r.text
  assert _kb_note_count() == before, "驳回绝不能写主知识库"
  from app.storage import candidates as store
  row = store.get_candidate(cid)
  assert row["status"] == "rejected"
  assert row["review_note"] == "来源不可靠"


def test_duplicate_candidate_is_deduped():
  """同一 url + 同一内容重复入候选 -> 不新增。"""
  first = _seed("https://example.com/same", "一模一样的内容")
  second = _seed_raw("https://example.com/same", "一模一样的内容")
  assert first, "首次应写入"
  assert second == [], "重复内容应被去重"


def test_same_url_different_content_is_new():
  """同一 url 内容变了 -> 重新提审（页面改版是真实场景）。"""
  _seed("https://example.com/rev", "第一版内容")
  again = _seed_raw("https://example.com/rev", "第二版内容完全不同")
  assert again, "同 url 不同内容应重新提审"


def test_failed_ingest_keeps_candidate_pending():
  """入库失败时状态必须留在 pending，不能出现「批准了但知识没进去」。"""
  # 注意打桩位置：candidates.py 用的是 `from app.tools.ingest import
  # ingest_web_candidate`，模块里持有的是自己的引用。patch
  # app.tools.ingest.ingest_web_candidate 不会生效（本文件踩过这个坑，
  # 结果是 patch 无效 + 断言失败）。
  import app.api.candidates as C
  orig = C.ingest_web_candidate

  def _fail(**kw):
    raise RuntimeError("模拟入库失败")

  C.ingest_web_candidate = _fail
  try:
    ids = _seed("https://example.com/fail", "会失败的内容")
    cid = ids[0]
    before = _kb_note_count()
    r = _client().post("/api/candidates/%d/approve" % cid, json={})
    assert r.status_code == 400, r.text
    from app.storage import candidates as store
    row = store.get_candidate(cid)
    assert row["status"] == "pending", "入库失败必须保持 pending，否则会静默丢知识"
    assert _kb_note_count() == before
  finally:
    C.ingest_web_candidate = orig


def test_batch_reject():
  ids = _seed("https://example.com/b1", "批量驳回 A") + _seed("https://example.com/b2", "批量驳回 B")
  assert len(ids) == 2
  r = _client().post("/api/candidates/batch-reject", json={"ids": ids, "note": "批量"})
  assert r.status_code == 200, r.text
  assert sorted(r.json()["ok"]) == sorted(ids)
  from app.storage import candidates as store
  for cid in ids:
    assert store.get_candidate(cid)["status"] == "rejected"


def test_batch_approve_reports_per_item_failures():
  """批量批准要逐条隔离：单条失败不能拖垮其它条目。"""
  I, orig_embed, orig_add = _fake_embed_patches()
  try:
    ids = _seed("https://example.com/ba1", "批量批准 A") + _seed("https://example.com/ba2", "批量批准 B")
    payload = {"ids": ids + [999999], "note": "批量"}   # 999999 不存在
    r = _client().post("/api/candidates/batch-approve", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert sorted(body["ok"]) == sorted(ids), body
    assert len(body["failed"]) == 1 and body["failed"][0]["id"] == 999999
  finally:
    I.embed_texts, I.add_chunks = orig_embed, orig_add


def test_unknown_candidate_returns_404():
  assert _client().post("/api/candidates/888888/approve", json={}).status_code == 404
  assert _client().post("/api/candidates/888888/reject", json={}).status_code == 404


def main() -> int:
  # 先确认隔离生效，否则后面的用例会污染开发库
  try:
    path = _assert_isolated()
    print("隔离检查通过：sqlite_path=%s" % path)
  except AssertionError as e:
    print("FAIL 隔离检查: %s" % e)
    return 1
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
