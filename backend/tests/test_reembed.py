"""重建索引（reembed）的测试 —— 知识库内容改动后如何生效。

这是「改了知识库内容，怎么让它生效」的唯一现存路径，之前是坏的：

  api_reembed_note 的顺序是「先 add 再 delete」，而
  delete_note_chunks() 按 note_id 删、add_chunks() 用的 id 就是 {note_id}_c{i}
  —— 新旧完全同 id，于是**刚写入的新向量被自己删掉**。
  实测：接口返回 200，Chroma 0 条、FTS 0 行，DB 却仍标记 embedded=True，
  笔记变成永久检索不到的孤儿。静默数据丢失，比报错难发现得多。

正确顺序：先算 embedding -> 再删旧 -> 最后写新。
  1) 先算 embedding：失败时原索引完全不动，笔记仍可检索。
  2) 先删后写：反过来会把自己写的删掉（上面的坑）。
  3) embedded 必须反映真实结果，不能无条件 True。

全部离线：embedding 用假向量，Chroma 落临时目录。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = tempfile.mkdtemp(prefix="hd_reembed_test_")
os.environ["DATA_DIR"] = _TMP
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "notes.db")
os.environ["NOTES_DIR"] = os.path.join(_TMP, "notes")
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")

from fastapi.testclient import TestClient  # noqa: E402


def _patch_embed(fail=False):
  """把 embedding 换成确定性假实现（或在重建路径上让它失败）。

  三个都要打，因为三处各自持有引用：
    * app.tools.ingest.embed_texts     —— 首次入库用
    * app.api.notes.embed_texts        —— 历史遗留引用
    * app.tools.reindex.embed_texts    —— 重建索引的共用实现（现在的主路径）
  """
  import app.tools.ingest as I
  import app.api.notes as N
  import app.tools.reindex as R

  def fake(chunks, **kw):
    if fail:
      raise RuntimeError("模拟 embedding 服务不可用")
    return [[0.1, 0.2, 0.3] for _ in chunks]

  I.embed_texts = fake
  N.embed_texts = fake
  R.embed_texts = fake


def _client() -> TestClient:
  from app.main import app
  return TestClient(app)


_AUTH: dict | None = None


def _auth() -> dict:
  global _AUTH
  if _AUTH:
    return _AUTH
  c = _client()
  creds = {"phone": "13800138000", "password": "Test1234!"}
  r = c.post("/api/auth/register", json=creds)
  token = (r.json() or {}).get("token") if r.status_code in (200, 201) else None
  if not token:
    r = c.post("/api/auth/login", json={"account": creds["phone"],
                                        "password": creds["password"]})
    token = (r.json() or {}).get("token")
  _AUTH = {"Authorization": "Bearer " + token}
  return _AUTH


def _authed_client() -> TestClient:
  return TestClient(_client().app, headers=_auth())


def _make_note(content: str, title: str = "测试笔记"):
  from app.tools.ingest import ingest_text
  return ingest_text(content, title=title)


def _vectors(note_id: str) -> list:
  from app.storage.vector import get_collection
  return get_collection().get(where={"note_id": note_id})["ids"]


def _fts_count(note_id: str) -> int:
  from sqlalchemy import text
  from app.storage.db import get_engine
  with get_engine().connect() as conn:
    return conn.execute(
      text("SELECT COUNT(*) FROM chunk_fts WHERE note_id=:n"), {"n": note_id}
    ).scalar()


def _db_note(note_id: str):
  from sqlmodel import Session
  from app.storage.db import Note, get_engine
  with Session(get_engine()) as s:
    return s.get(Note, note_id)


# ============ 1. 回归守卫：reembed 不能把笔记清空 ============

def test_reembed_preserves_vectors():
  """核心回归：reembed 之后笔记必须仍然可检索。

  旧代码在这里失败：返回 200，但 Chroma/FTS 全空，DB 还写着 embedded=True。
  """
  _patch_embed()
  note = _make_note("黄山风景区 2025 年门票价格为 190 元。")
  assert len(_vectors(note.id)) > 0, "入库应产生向量"

  r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
  assert r.status_code == 200, r.text

  assert len(_vectors(note.id)) > 0, "reembed 后向量被清空了（笔记变成检索不到的孤儿）"
  assert _fts_count(note.id) > 0, "reembed 后 FTS 索引被清空了"

  n = _db_note(note.id)
  assert n.embedded is True
  assert n.chunk_count == len(_vectors(note.id)), \
    "chunk_count 应与实际向量数一致（%s vs %s）" % (n.chunk_count, len(_vectors(note.id)))


def test_reembed_is_idempotent():
  """连续重建多次结果稳定，不能每次少一点。"""
  _patch_embed()
  note = _make_note("皖南古村落西递宏村是世界文化遗产。")
  before = len(_vectors(note.id))
  for i in range(3):
    r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
    assert r.status_code == 200, "第 %d 次重建失败: %s" % (i + 1, r.text)
  assert len(_vectors(note.id)) == before, "多次重建后向量数应保持稳定"


# ============ 2. 内容改动要真的生效 ============

def test_reembed_picks_up_edited_content():
  """改几个字 / 加几段之后重建，新内容必须进索引。

  这就是「知识库更新了怎么生效」的实际场景：改磁盘上的 md -> 调 reembed。
  """
  _patch_embed()
  note = _make_note("黄山门票 190 元。")

  n = _db_note(note.id)
  new_content = "黄山门票 190 元。\n\n新增：2026 年起 6 岁以下儿童免票，索道票价 80 元。"
  with open(n.content_path, "w", encoding="utf-8") as f:
    f.write(new_content)

  r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
  assert r.status_code == 200, r.text

  from app.storage.vector import get_collection
  docs = get_collection().get(where={"note_id": note.id})["documents"]
  joined = " ".join(docs)
  assert "免票" in joined or "索道" in joined, "新增内容没有进索引: %s" % joined[:200]


def test_edited_content_replaces_old_in_fts():
  """FTS 也要跟着换，否则旧内容还能被关键词搜到。"""
  _patch_embed()
  note = _make_note("旧版本关键词：猕猴桃种植技术。")

  n = _db_note(note.id)
  with open(n.content_path, "w", encoding="utf-8") as f:
    f.write("新版本关键词：蓝莓种植技术。")

  r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
  assert r.status_code == 200, r.text

  from sqlalchemy import text
  from app.storage.db import get_engine
  with get_engine().connect() as conn:
    old = conn.execute(text(
      "SELECT COUNT(*) FROM chunk_fts WHERE note_id=:n AND content LIKE '%猕猴桃%'"),
      {"n": note.id}).scalar()
    new = conn.execute(text(
      "SELECT COUNT(*) FROM chunk_fts WHERE note_id=:n AND content LIKE '%蓝莓%'"),
      {"n": note.id}).scalar()
  assert new > 0, "新内容没进 FTS"
  assert old == 0, "旧内容仍在 FTS 里（会搜到已删除的内容）"


# ============ 3. 失败不能毁数据 ============

def test_embedding_failure_keeps_note_intact():
  """embedding 失败时原索引必须原封不动 —— 不能先删后算。

  这是「先算 embedding」这一步存在的意义。
  """
  _patch_embed()
  note = _make_note("重要的原始内容，不能被失败的重建毁掉。")
  before = len(_vectors(note.id))
  before_fts = _fts_count(note.id)
  assert before > 0

  _patch_embed(fail=True)          # 之后所有 embedding 都失败
  r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
  assert r.status_code == 400, "embedding 失败应报错，实际 %s: %s" % (r.status_code, r.text)

  assert len(_vectors(note.id)) == before, "embedding 失败时不该动原索引"
  assert _fts_count(note.id) == before_fts, "embedding 失败时不该动 FTS"
  assert _db_note(note.id).embedded is True, "笔记应保持可用状态"


def test_missing_note_returns_404():
  _patch_embed()
  r = _authed_client().post("/api/notes/n_does_not_exist/reembed", json={})
  assert r.status_code == 404, r.text


def test_empty_content_does_not_crash():
  """内容为空（用户把文件清空了）不能 500，也不能留下假 embedded=True。"""
  _patch_embed()
  note = _make_note("先有内容。")
  n = _db_note(note.id)
  with open(n.content_path, "w", encoding="utf-8") as f:
    f.write("")

  r = _authed_client().post("/api/notes/%s/reembed" % note.id, json={})
  assert r.status_code in (200, 400), r.text
  assert len(_vectors(note.id)) == 0, "清空内容后不该还有向量"
  assert _db_note(note.id).embedded is False, \
    "内容为空时不能标记为已索引（否则用户以为索引正常）"


# ============ 4. PATCH /notes/{id}/content —— 改内容的正式入口 ============

def test_patch_content_updates_index():
  """改几个字 / 加几段 -> 调 PATCH -> 新内容立即可检索。"""
  _patch_embed()
  note = _make_note("黄山门票 190 元。")

  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": "黄山门票 190 元。\n\n新增：索道票价 80 元。"})
  assert r.status_code == 200, r.text

  from app.storage.vector import get_collection
  docs = " ".join(get_collection().get(where={"note_id": note.id})["documents"])
  assert "索道" in docs, "新内容没有进索引: %s" % docs[:200]


def test_patch_content_keeps_note_id():
  """note_id 必须保持不变。

  删掉重建也能「更新内容」，但 id 变了 —— 历史回答里 [n] 指向的 note_id
  会全部失效，引用链接直接烂掉。这是不能用「删+建」实现的原因。
  """
  _patch_embed()
  note = _make_note("原始内容。")
  before_id = note.id

  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": "改过的内容。"})
  assert r.status_code == 200, r.text
  assert r.json()["id"] == before_id, "note_id 被改掉了，历史引用会失效"
  assert _db_note(before_id) is not None


def test_patch_content_can_update_title():
  _patch_embed()
  note = _make_note("内容。", title="旧标题")
  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": "新内容。", "title": "新标题"})
  assert r.status_code == 200, r.text
  assert r.json()["title"] == "新标题"
  assert _db_note(note.id).title == "新标题"


def test_patch_content_old_text_gone_from_fts():
  """旧正文必须从 FTS 消失，否则会搜到已经改掉的内容。"""
  _patch_embed()
  note = _make_note("旧版：猕猴桃种植技术。")
  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": "新版：蓝莓种植技术。"})
  assert r.status_code == 200, r.text

  from sqlalchemy import text
  from app.storage.db import get_engine
  with get_engine().connect() as conn:
    old = conn.execute(text(
      "SELECT COUNT(*) FROM chunk_fts WHERE note_id=:n AND content LIKE '%猕猴桃%'"),
      {"n": note.id}).scalar()
    new = conn.execute(text(
      "SELECT COUNT(*) FROM chunk_fts WHERE note_id=:n AND content LIKE '%蓝莓%'"),
      {"n": note.id}).scalar()
  assert new > 0, "新内容没进 FTS"
  assert old == 0, "旧内容仍在 FTS 里"


def test_patch_content_embedding_failure_leaves_note_usable():
  """embedding 失败 -> 400，且旧正文与旧索引都不能被破坏。"""
  _patch_embed()
  note = _make_note("重要的原始内容。")
  before_vectors = len(_vectors(note.id))
  assert before_vectors > 0

  _patch_embed(fail=True)
  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": "新内容（这次会失败）。"})
  assert r.status_code == 400, r.text
  assert len(_vectors(note.id)) == before_vectors, "失败时不该动原索引"
  assert _db_note(note.id).embedded is True, "笔记应保持可用"

  from app.storage.vector import get_collection
  docs = " ".join(get_collection().get(where={"note_id": note.id})["documents"])
  assert "重要的原始内容" in docs, "失败时正文也被改掉了，用户数据丢失"


def test_patch_content_missing_note_returns_404():
  _patch_embed()
  r = _authed_client().patch("/api/notes/n_nope/content", json={"content": "x"})
  assert r.status_code == 404, r.text


def test_patch_content_empty_string_is_allowed_but_not_marked_indexed():
  """允许清空（用户可能先清再写），但不能假装索引还在。"""
  _patch_embed()
  note = _make_note("先有内容。")
  r = _authed_client().patch("/api/notes/%s/content" % note.id,
                             json={"content": ""})
  assert r.status_code == 200, r.text
  assert len(_vectors(note.id)) == 0
  assert _db_note(note.id).embedded is False


# ============ 5. 共用重建实现（feishu 后台同步路径） ============

def test_shared_helper_keep_old_on_embed_failure():
  """keep_old_on_embed_failure=True：正文落地但保留旧向量、不推进 revision。

  后台同步没有交互对象，不能像手动重建那样直接报错放弃 —— 那样这条笔记会
  永远停在未索引状态，而且下一轮同步因为 revision 已推进还会跳过它。
  """
  import app.tools.reindex as R
  from app.tools.reindex import replace_note_index

  _patch_embed()
  note = _make_note("第一版内容。")
  before_vectors = len(_vectors(note.id))

  _patch_embed(fail=True)
  res = replace_note_index(note.id, "第二版内容。", source_revision="rev-2",
                           write_content=True, keep_old_on_embed_failure=True)

  assert res.embedding_failed is True, res
  assert res.embedded is False
  assert len(_vectors(note.id)) == before_vectors, "应保留旧向量以便继续可检索"
  n = _db_note(note.id)
  assert n.embedded is False, "应如实标记为未索引，下一轮同步才会重试"
  assert n.source_revision != "rev-2", \
    "embedding 失败时不能推进 revision，否则下轮同步会跳过这条未索引的笔记"


def test_shared_helper_advances_revision_only_on_success():
  """成功时才推进 revision。"""
  from app.tools.reindex import replace_note_index

  _patch_embed()
  note = _make_note("内容。")
  res = replace_note_index(note.id, "新内容。", source_revision="rev-9",
                           write_content=True, keep_old_on_embed_failure=True)
  assert res.embedded is True, res
  assert _db_note(note.id).source_revision == "rev-9"


def test_shared_helper_missing_note():
  from app.tools.reindex import replace_note_index
  _patch_embed()
  res = replace_note_index("n_missing", "x")
  assert res.note is None and res.ok is False


def test_feishu_drop_and_reingest_wrapper():
  """飞书同步的包装函数（重构后只剩参数传递，但要确认没传错）。

  这条路径原先是对的，我把它改成调用共用实现 —— 改动「本来正常」的代码
  必须直接测一次，不能只靠共用实现的单测推断。
  """
  from app.feishu_sync import _drop_and_reingest
  from app.storage.db import get_engine, Note
  from sqlmodel import Session

  _patch_embed()
  note = _make_note("飞书第一版。", title="飞书文档")

  with Session(get_engine()) as s:
    existing = s.get(Note, note.id)

  out = _drop_and_reingest(existing, "飞书第二版标题", "飞书第二版内容。",
                           "feishu_docx", "rev-100",
                           None, None, None)

  assert out is not None and out.id == note.id, "note id 必须保持不变"
  assert out.title == "飞书第二版标题"
  assert out.embedded is True, "正常路径应重建成功"
  assert out.source_revision == "rev-100", "成功时应推进 revision"

  from app.storage.vector import get_collection
  docs = " ".join(get_collection().get(where={"note_id": note.id})["documents"])
  assert "飞书第二版内容" in docs, "新内容没进索引: %s" % docs[:200]
  assert "飞书第一版" not in docs, "旧内容仍在索引里"


# ============ 6. 并发重建（后台扫描 vs 用户请求） ============

def test_concurrent_reindex_keeps_index_consistent():
  """同一篇笔记被并发重建时，索引不能与正文/指纹不一致。

  这个函数会被两个来源并发调用：后台扫描线程（notes_watch）和用户请求
  （PATCH /content、reembed）。两边的流程都是「删旧 -> 写新 -> 更新行」，
  交错时会出现：
      A 删、B 删、B 写（5 段）、A 写（2 段）
      -> Chroma 留下 5 个向量，而 chunk_count 被 A 写成 2
  结果是**索引与正文不一致且不报错** —— 扫描器看到指纹匹配就不再重试，
  用户搜到的是混在一起的旧内容。和 MCPSession 串线是同一类问题。

  这里用多轮并发去撞这个窗口，断言核心不变量：
  chunk_count 必须等于 Chroma 里实际的向量数。
  """
  import threading
  import time
  import app.tools.reindex as R

  _patch_embed()
  note = _make_note("初始内容。")
  nid = note.id

  real_add = R.add_chunks
  real_del = R.delete_note_chunks

  def slow_add(note_id, chunks, embeddings):
    time.sleep(0.002)                  # 放大窗口，让交错真的发生
    return real_add(note_id, chunks, embeddings)

  def slow_del(note_id):
    time.sleep(0.002)
    return real_del(note_id)

  R.add_chunks = slow_add
  R.delete_note_chunks = slow_del
  try:
    # 关键：文本长度必须产生**不同的 chunk 数**。
    # chunk_size=500，所以 600/1200/1800… 字符分别切出 2/3/4… 段。
    # 第一版这里写成 "短。"*(3+i*7)（最多 104 字符），每次都是 1 段 ——
    # 所有线程用同一个 id {note_id}_c0 互相覆盖，永远撞不出不一致，
    # 测试看着通过其实什么都没验证。
    contents = [("第%d篇。" % i) + ("内容填充。" * (60 + i * 55)) for i in range(6)]

    def worker(text):
      R.replace_note_index(nid, text, write_content=False)

    for _round in range(4):
      threads = [threading.Thread(target=worker, args=(c,)) for c in contents]
      for t in threads:
        t.start()
      for t in threads:
        t.join()
  finally:
    R.add_chunks = real_add
    R.delete_note_chunks = real_del

  from app.storage.vector import get_collection
  vectors = len(get_collection().get(where={"note_id": nid})["ids"])
  n = _db_note(nid)
  assert n.chunk_count == vectors, \
    "索引与记录不一致：chunk_count=%d 但 Chroma 有 %d 个向量" % (n.chunk_count, vectors)
  assert n.embedded is True


def test_reindex_lock_is_per_note():
  """锁按 note 粒度 —— 不同笔记之间不该互相阻塞。

  后台一轮要扫很多篇，如果是一把全局锁，用户的 PATCH 会排在一整轮扫描后面。
  """
  import threading
  import time
  import app.tools.reindex as R

  _patch_embed()
  a = _make_note("笔记 A 的内容。")
  b = _make_note("笔记 B 的内容。")

  real_embed = R.embed_texts
  entered = threading.Semaphore(0)
  release = threading.Event()

  def blocking_embed(chunks, **kw):
    entered.release()              # 通知：我进来了
    release.wait(timeout=5)        # 卡住，模拟慢 embedding
    return real_embed(chunks, **kw)

  R.embed_texts = blocking_embed
  try:
    t1 = threading.Thread(target=lambda: R.replace_note_index(a.id, "A 新内容。"))
    t1.start()
    entered.acquire(timeout=5)     # A 已进入 embedding

    # B 不该被 A 阻塞：它应该也能进到 embedding
    t2 = threading.Thread(target=lambda: R.replace_note_index(b.id, "B 新内容。"))
    t2.start()
    got = entered.acquire(timeout=3)
    assert got, "不同笔记之间被互相阻塞了 —— 锁的粒度不对"
  finally:
    release.set()
    R.embed_texts = real_embed
    t1.join(timeout=10)
    t2.join(timeout=10)


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
