# -*- coding: utf-8 -*-
"""歧义澄清链路测试（模型驱动，始终开启）：

- router_node 歧义检测：ambiguous JSON -> clarify_request；skip_clarify 二次防重
- route_by_intent：clarify_request -> "clarify"，skip_clarify 后恢复正常路由
- _build_initial_state 泄露防护：clarify_request / skip_clarify 每轮显式重置
- clarify broker：create/resolve/wait_answer 闭环、未知 id、超时跳过
- /chat/clarify 端点 + 图结构 smoke compile

不依赖真实 LLM：router 的 LLM 调用用 fake model 替换。
"""
import sys, os, tempfile, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# 用临时 DB，避免污染 data/notes.db。
# 注意：真正生效的是下面那行直接改 settings.sqlite_path。
# data_dir/sqlite_path/notes_dir/chroma_dir 是四个独立字段，都默认 ./data，
# 设 HD_DATA_DIR **无效**（config.py 里没有这个名字）—— 别照抄那种写法，
# 实测会把测试数据写进开发库（候选库那次污染了 22 条）。
tmp = tempfile.mkdtemp()

from app.storage import db as dbm
dbm._engine = None  # reset engine if imported elsewhere
from app.config import settings
settings.sqlite_path = os.path.join(tmp, "test.db")


class _FakeChat:
  """invoke 返回预置 JSON 文本，模拟 router LLM。"""
  def __init__(self, payload: dict):
    self._payload = payload
    self.called = {"n": 0}

  def invoke(self, prompt):
    self.called["n"] += 1
    class _R:  # mimic LangChain message
      content = json.dumps(self._payload, ensure_ascii=False)
    return _R()


def _patch_router_model(payload: dict):
  """把 router 模块里的 _build_model 换成 fake；返回 (fake, restore)。"""
  from app.agent.nodes import router as router_mod
  fake = _FakeChat(payload)
  orig = router_mod._build_model
  router_mod._build_model = lambda **kw: fake
  orig_enabled = settings.router_enabled
  settings.router_enabled = True

  def _restore():
    router_mod._build_model = orig
    settings.router_enabled = orig_enabled
  return fake, _restore


def test_router_ambiguous_sets_clarify_request():
  """router 判定歧义 -> 输出 clarify_request {question, options}。"""
  from app.agent.nodes.router import router_node
  fake, restore = _patch_router_model({
    "intent": "research", "rewritten_query": "帮我调研哪吒",
    "ambiguous": True,
    "clarify": {"question": "你想了解的是哪个「哪吒」?",
                "options": ["哪吒汽车(新能源汽车品牌)", "动漫/神话中的哪吒角色", ""]},
  })
  try:
    out = router_node({"query": "帮我调研哪吒", "messages": [], "step_count": 0})
  finally:
    restore()
  assert out["intent"] == "research"
  assert out["clarify_request"]["question"] == "你想了解的是哪个「哪吒」?"
  # 空选项被过滤，最多 4 个
  assert out["clarify_request"]["options"] == ["哪吒汽车(新能源汽车品牌)", "动漫/神话中的哪吒角色"]
  print("PASS router ambiguous -> clarify_request")


def test_router_unambiguous_no_clarify():
  """无歧义 -> clarify_request 必须是 None（覆盖 checkpointer 残留）。"""
  from app.agent.nodes.router import router_node
  fake, restore = _patch_router_model({
    "intent": "chat", "rewritten_query": "牛魔王的来历", "ambiguous": False,
  })
  try:
    out = router_node({"query": "牛魔王的来历", "messages": [], "step_count": 0})
  finally:
    restore()
  assert out["clarify_request"] is None, out
  print("PASS router unambiguous -> None")


def test_router_skip_clarify_never_reasks():
  """skip_clarify=True（用户已回答后的二次执行）-> 即使再判歧义也不追问。"""
  from app.agent.nodes.router import router_node
  fake, restore = _patch_router_model({
    "intent": "research", "rewritten_query": "帮我调研哪吒（用户澄清：哪吒汽车）",
    "ambiguous": True,
    "clarify": {"question": "哪个哪吒?", "options": ["a", "b"]},
  })
  try:
    out = router_node({"query": "帮我调研哪吒（用户澄清：哪吒汽车）",
                       "messages": [], "step_count": 0, "skip_clarify": True})
  finally:
    restore()
  assert out["clarify_request"] is None, "skip_clarify 时不得再次挂起澄清"
  print("PASS router skip_clarify guard")


def test_route_by_intent_clarify_branch():
  """route_by_intent：clarify_request 优先于一切意图；skip_clarify 恢复正常。"""
  from app.agent.nodes.router import route_by_intent
  spec = {"question": "q?", "options": ["a", "b"]}
  assert route_by_intent({"intent": "research", "clarify_request": spec}) == "clarify"
  assert route_by_intent({"intent": "chat", "clarify_request": spec}) == "clarify"
  # skip_clarify 后按原意图走
  assert route_by_intent({"intent": "research", "clarify_request": spec,
                          "skip_clarify": True}) == "research"
  assert route_by_intent({"intent": "chat", "clarify_request": None}) == "chat"
  print("PASS route_by_intent clarify branch")


def test_build_initial_state_resets_clarify():
  """checkpointer 泄露防护：新一轮必须 clarify_request=None / skip_clarify=False。"""
  from app.api.chat import ChatRequest, Message, _build_initial_state
  body = ChatRequest(messages=[Message(role="user", content="帮我调研哪吒")])
  st = _build_initial_state(body, "帮我调研哪吒", "sess-1", None, None, None, None, None)
  assert st["clarify_request"] is None, st.get("clarify_request")
  assert st["skip_clarify"] is False
  # 不再存在 plan_override 字段
  assert "plan_override" not in st, "plan_override 已随审批功能删除"
  print("PASS _build_initial_state clarify reset")


def test_clarify_broker_roundtrip():
  """broker 闭环：create -> resolve -> wait_answer 拿到答案。"""
  from app.agent import clarify as c

  async def _run():
    rid, fut = c.create_request()
    assert c.resolve(rid, "哪吒汽车") is True
    got = await c.wait_answer(rid, timeout=1)
    # 已消费的 id 再 resolve -> False
    assert c.resolve(rid, "again") is False
    return got
  got = asyncio.run(_run())
  assert got == "哪吒汽车", got
  print("PASS broker roundtrip")


def test_clarify_broker_timeout_and_unknown():
  """未知 id -> False；超时 -> 空串（视为跳过，按原问题继续）。"""
  from app.agent import clarify as c
  assert c.resolve("no-such-id", "x") is False

  async def _run():
    rid, fut = c.create_request()
    got = await c.wait_answer(rid, timeout=0.05)
    return rid, got
  rid, got = asyncio.run(_run())
  assert got == "", got
  # 超时后 pending 已清理
  assert rid not in c._pending
  print("PASS broker timeout/unknown")


def test_clarify_endpoint():
  """/chat/clarify 端点：resolve 成功 ok=True，未知 id ok=False。"""
  from app.api.chat import resolve_clarify, ClarifyDecision
  from app.agent import clarify as c

  async def _run():
    rid, fut = c.create_request()
    ok = await resolve_clarify(ClarifyDecision(request_id=rid, answer="动漫角色"))
    got = await c.wait_answer(rid, timeout=1)
    ok2 = await resolve_clarify(ClarifyDecision(request_id="ghost", answer="x"))
    return ok, got, ok2
  ok, got, ok2 = asyncio.run(_run())
  assert ok == {"ok": True}
  assert got == "动漫角色", got
  assert ok2 == {"ok": False}
  print("PASS /chat/clarify endpoint")


def test_graph_compiles_with_clarify_edge():
  """图结构 smoke：含 clarify 边的 workflow 能正常 compile。"""
  from app.agent.graph import _build_workflow
  g = _build_workflow().compile()
  assert g is not None
  print("PASS graph compile (clarify edge)")


def _reset_graph_singleton():
  """aiosqlite/AsyncSqliteSaver 绑定创建时的 event loop；每个 e2e 测试各自
  asyncio.run 新循环，必须重置模块级单例（生产环境 uvicorn 单循环无此问题）。"""
  import app.agent.graph as graph_mod
  graph_mod._graph = None
  graph_mod._conn = None


async def _teardown_graph():
  """在当前事件循环内关闭 aiosqlite 连接。

  aiosqlite 的 worker 线程是非 daemon 线程：不 close 的话线程永远等在队列上，
  测试进程打印完结果也无法退出（首次运行时挂起的根因）。
  """
  import app.agent.graph as graph_mod
  if graph_mod._conn is not None:
    try:
      await graph_mod._conn.close()
    except Exception:
      pass
  graph_mod._graph = None
  graph_mod._conn = None


def test_chat_two_pass_clarify_e2e():
  """端到端两段式：/chat SSE 流中 router 判歧义 -> clarify 事件 ->
  应答 -> 带精炼问题二次执行图 -> 正常输出答案。LLM 全部 mock。"""
  import asyncio
  from app.api import chat as chat_mod
  from app.api.chat import ChatRequest, Message, resolve_clarify, ClarifyDecision
  from app.agent.nodes import router as router_mod

  _reset_graph_singleton()
  calls = {"router": 0}

  class FakeChat:
    """第一轮返回歧义 research；第二轮（拿到澄清后）返回 chat。"""
    def invoke(self, prompt):
      calls["router"] += 1
      if calls["router"] == 1:
        payload = {"intent": "research", "rewritten_query": "帮我调研哪吒",
                   "ambiguous": True,
                   "clarify": {"question": "你想了解的是哪个「哪吒」?",
                               "options": ["哪吒汽车", "动漫角色"]}}
      else:
        payload = {"intent": "chat",
                   "rewritten_query": "帮我调研哪吒（用户澄清：哪吒汽车）",
                   "ambiguous": False}
      class _R:
        content = json.dumps(payload, ensure_ascii=False)
      return _R()

  async def fake_answer_stream(state):
    # 断言二次执行的 query 已并入澄清答案
    assert "用户澄清：哪吒汽车" in (state.get("query") or ""), state.get("query")
    yield ("text_delta", "这是澄清后的回答。")

  orig_model = router_mod._build_model
  orig_enabled = settings.router_enabled
  orig_ans = chat_mod.answer_node_stream
  router_mod._build_model = lambda **kw: FakeChat()
  settings.router_enabled = True
  chat_mod.answer_node_stream = fake_answer_stream

  async def _run():
    try:
      body = ChatRequest(messages=[Message(role="user", content="帮我调研哪吒")])
      resp = await chat_mod.chat(body, None)
      out = []
      async for chunk in resp.body_iterator:
        out.append(chunk)
        if "event: clarify" in chunk:
          data_line = [l for l in chunk.splitlines() if l.startswith("data: ")][0][6:]
          rid = json.loads(data_line)["request_id"]
          ok = await resolve_clarify(ClarifyDecision(request_id=rid, answer="哪吒汽车"))
          assert ok == {"ok": True}
      return "".join(out)
    finally:
      await _teardown_graph()

  try:
    sse = asyncio.run(_run())
  finally:
    router_mod._build_model = orig_model
    settings.router_enabled = orig_enabled
    chat_mod.answer_node_stream = orig_ans

  assert "event: clarify" in sse, "必须先下发澄清事件"
  assert calls["router"] == 2, f"router 应执行两轮，实际 {calls['router']}"
  assert "你想了解的是哪个「哪吒」?" in sse
  assert "这是澄清后的回答。" in sse, "二次执行后应正常输出答案"
  print("PASS chat two-pass clarify e2e")


def test_chat_clarify_timeout_falls_back():
  """澄清超时/跳过：空答案 -> 按原问题二次执行（不注入澄清）。"""
  import asyncio
  from app.api import chat as chat_mod
  from app.api.chat import ChatRequest, Message, resolve_clarify, ClarifyDecision
  from app.agent.nodes import router as router_mod
  from app.agent import clarify as clarify_mod

  _reset_graph_singleton()
  calls = {"router": 0}

  class FakeChat:
    def invoke(self, prompt):
      calls["router"] += 1
      if calls["router"] == 1:
        payload = {"intent": "research", "rewritten_query": "帮我调研哪吒",
                   "ambiguous": True,
                   "clarify": {"question": "哪个哪吒?", "options": ["a", "b"]}}
      else:
        payload = {"intent": "chat", "rewritten_query": "帮我调研哪吒",
                   "ambiguous": False}
      class _R:
        content = json.dumps(payload, ensure_ascii=False)
      return _R()

  async def fake_answer_stream(state):
    # 跳过场景：query 保持原样，不带「用户澄清」
    assert "用户澄清" not in (state.get("query") or ""), state.get("query")
    yield ("text_delta", "按原问题回答。")

  orig_model = router_mod._build_model
  orig_enabled = settings.router_enabled
  orig_ans = chat_mod.answer_node_stream
  orig_wait = clarify_mod.wait_answer
  router_mod._build_model = lambda **kw: FakeChat()
  settings.router_enabled = True
  chat_mod.answer_node_stream = fake_answer_stream

  async def _fast_wait(request_id, timeout=300.0):
    return ""  # 模拟用户跳过/超时
  clarify_mod.wait_answer = _fast_wait

  async def _run():
    try:
      body = ChatRequest(messages=[Message(role="user", content="帮我调研哪吒")])
      resp = await chat_mod.chat(body, None)
      out = []
      async for chunk in resp.body_iterator:
        out.append(chunk)
      return "".join(out)
    finally:
      await _teardown_graph()

  try:
    sse = asyncio.run(_run())
  finally:
    router_mod._build_model = orig_model
    settings.router_enabled = orig_enabled
    chat_mod.answer_node_stream = orig_ans
    clarify_mod.wait_answer = orig_wait

  assert "event: clarify" in sse
  assert calls["router"] == 2
  assert "按原问题回答。" in sse
  print("PASS clarify timeout/skip fallback")


if __name__ == "__main__":
  test_router_ambiguous_sets_clarify_request()
  test_router_unambiguous_no_clarify()
  test_router_skip_clarify_never_reasks()
  test_route_by_intent_clarify_branch()
  test_build_initial_state_resets_clarify()
  test_clarify_broker_roundtrip()
  test_clarify_broker_timeout_and_unknown()
  test_clarify_endpoint()
  test_graph_compiles_with_clarify_edge()
  test_chat_two_pass_clarify_e2e()
  test_chat_clarify_timeout_falls_back()
  print("\nALL CLARIFY TESTS PASSED")
