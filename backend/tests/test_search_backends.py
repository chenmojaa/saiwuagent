"""联网搜索后端与自动升级的测试。

背景：httpx 抓 Bing 对无 cookie 客户端经常返回**完全无关**的结果。
2026-09-17 实测（同一批查询）：

  | 查询 | httpx 相关度 | 真实浏览器 |
  |---|---|---|
  | 贵州茅台 2026 年一季度财报 | 0.00（Wikipedia / Paramount+） | 1.00 |
  | 上海 2026 年落户政策 | 0.00（Microsoft / X-Ray 材质包） | 0.83 |
  | 比亚迪 2025 年销量 | 0.40（官网实体页，过闸门） | — |

所以 web_search 加了自动升级：httpx 结果过不了相关性闸门时才改用真实浏览器。
这个文件锁定三件事：
  1. **该升级时升级** —— 结果为空或不相关时切到浏览器
  2. **不该升级时不升级** —— 结果够相关时必须走快速路径，不能平白多花 5-12s
  3. **降级安全** —— 浏览器不可用/返回空/没更相关时，保留原结果，绝不更差

全部离线可跑：httpx 与浏览器两层都打桩，不真的联网、不开浏览器。
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 隔离数据目录（读 MCP 注册表要用）。注意必须逐个设 —— 只设 HD_DATA_DIR 无效。
_TMP = tempfile.mkdtemp(prefix="hd_search_test_")
os.environ["DATA_DIR"] = _TMP
os.environ["SQLITE_PATH"] = os.path.join(_TMP, "notes.db")
os.environ["NOTES_DIR"] = os.path.join(_TMP, "notes")
os.environ["CHROMA_DIR"] = os.path.join(_TMP, "chroma")


# ---------- 脚手架 ----------

@contextlib.contextmanager
def _override(**kw):
  from app.config import settings
  old = {k: getattr(settings, k) for k in kw}
  for k, v in kw.items():
    object.__setattr__(settings, k, v)
  try:
    yield
  finally:
    for k, v in old.items():
      object.__setattr__(settings, k, v)


def _row(title: str, url: str, snippet: str) -> dict:
  return {"title": title, "url": url, "snippet": snippet}


@contextlib.contextmanager
def _patch(html_rows=None, browser_rows=None, browser_available=True,
           browser_raises=False, calls=None):
  """打桩 httpx 层（_search_raw）与浏览器层（browser_search）。

  calls 会记录浏览器被调了几次，用来断言「该不该升级」。
  """
  import app.tools.web_search as W
  import app.tools.browser_search as B

  orig_raw = W._search_raw
  orig_avail = B.available
  orig_bs = B.browser_search

  def fake_raw(q, limit, timeout):
    return list(html_rows or [])

  def fake_avail():
    return browser_available

  def fake_bs(q, max_results=5, fetch_top=0):
    if calls is not None:
      calls.append(q)
    if browser_raises:
      raise RuntimeError("模拟浏览器失败")
    return list(browser_rows or [])

  W._search_raw = fake_raw
  B.available = fake_avail
  B.browser_search = fake_bs
  try:
    yield
  finally:
    W._search_raw = orig_raw
    B.available = orig_avail
    B.browser_search = orig_bs


def _mk(rows, prefix="web:b"):
  """把原始行转成 chunk 形态。

  默认用 `web:b` 前缀，与 browser_search 的真实输出一致（b 段 = browser 后端），
  这样断言 note_id 才是在测代码而不是在测桩。
  """
  out = []
  for i, r in enumerate(rows):
    out.append({
      "note_id": "%s%d:%s" % (prefix, i, r["url"][:120]),
      "title": r["title"], "chunk_index": 0, "text": r["snippet"],
      "source_type": "web", "source_url": r["url"],
      "final_score": 0.5, "matched_query": "",
    })
  return out


def _mk_html(rows):
  """httpx 后端的 chunk 形态（web:<n>: 前缀，无 b 段）。"""
  return _mk(rows, prefix="web:")


# ============ 1. 该升级时升级 ============

def test_escalates_when_html_results_irrelevant():
  """httpx 结果完全不相关 -> 切到浏览器。"""
  import app.tools.web_search as W
  calls: list = []
  with _patch(html_rows=[_row("Microsoft", "https://ms.com", "AI cloud productivity")],
              browser_rows=_mk([_row("上海落户新政策", "https://gov.cn/x",
                                     "上海 2026 年落户政策 最新条件")]),
              calls=calls):
    out = W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0)
  assert calls, "结果不相关时必须尝试浏览器"
  assert out and out[0]["title"] == "上海落户新政策", out
  assert out[0]["note_id"].startswith("web:b"), "应使用浏览器结果（note_id 带 b 段）"


def test_escalates_when_html_returns_nothing():
  """httpx 一条都没有 -> 也要试浏览器。"""
  import app.tools.web_search as W
  calls: list = []
  with _patch(html_rows=[],
              browser_rows=_mk([_row("财报", "https://a.com", "贵州茅台 2026 一季度财报")]),
              calls=calls):
    out = W.web_search("贵州茅台 2026 年一季度财报", max_results=5, fetch_top=0)
  assert calls, "无结果时必须尝试浏览器"
  assert out, out


# ============ 2. 不该升级时不升级（省时间） ============

def test_does_not_escalate_when_html_results_relevant():
  """httpx 结果够相关 -> 必须走快速路径，不启动浏览器。"""
  import app.tools.web_search as W
  calls: list = []
  good = [_row("比亚迪 2025 年销量突破 400 万辆",
               "https://byd.com/report", "比亚迪 2025 年销量 400 万辆")]
  with _patch(html_rows=good,
              browser_rows=_mk([_row("别的", "https://b.com", "不相关内容")]),
              calls=calls):
    out = W.web_search("比亚迪 2025 年销量", max_results=5, fetch_top=0)
  assert calls == [], "相关时不该启动浏览器（会白花 5-12s）"
  assert out[0]["title"].startswith("比亚迪"), out


def test_allow_browser_false_never_escalates():
  """allow_browser=False 强制只走 httpx（测试/排查用）。"""
  import app.tools.web_search as W
  calls: list = []
  with _patch(html_rows=[_row("Microsoft", "https://ms.com", "cloud")],
              browser_rows=_mk([_row("好结果", "https://a.com", "上海 2026 落户政策")]),
              calls=calls):
    W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0, allow_browser=False)
  assert calls == [], "allow_browser=False 时不得调浏览器"


def test_config_switch_disables_escalation():
  """HD_WEB_SEARCH_BROWSER_FALLBACK=false 时整体关闭升级。"""
  import app.tools.web_search as W
  calls: list = []
  with _override(web_search_browser_fallback=False):
    with _patch(html_rows=[_row("Microsoft", "https://ms.com", "cloud")],
                browser_rows=_mk([_row("好结果", "https://a.com", "上海 2026 落户政策")]),
                calls=calls):
      W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0)
  assert calls == [], "开关关闭时不得调浏览器"


# ============ 3. 降级安全：绝不比原结果更差 ============

def test_keeps_html_results_when_browser_unavailable():
  """没装 Playwright MCP -> 静默沿用 httpx 结果，功能退化为旧行为。"""
  import app.tools.web_search as W
  calls: list = []
  html = [_row("Microsoft", "https://ms.com", "cloud productivity")]
  with _patch(html_rows=html, browser_available=False, calls=calls):
    out = W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0)
  assert calls == [], "不可用时应直接跳过，不尝试调用"
  assert out and out[0]["title"] == "Microsoft", "应沿用 httpx 结果"


def test_keeps_html_results_when_browser_returns_nothing():
  """浏览器返回空 -> 保留 httpx 结果，不能变成「什么都搜不到」。"""
  import app.tools.web_search as W
  html = [_row("Microsoft", "https://ms.com", "cloud")]
  with _patch(html_rows=html, browser_rows=[]):
    out = W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0)
  assert out and out[0]["title"] == "Microsoft", out


def test_keeps_html_results_when_browser_not_better():
  """浏览器结果也没更相关 -> 保留原结果（不能因为换了后端就更差）。"""
  import app.tools.web_search as W
  # httpx 相关度 0.40（含「比亚迪」「销量」），浏览器结果完全无关
  html = [_row("比亚迪 2025 年销量突破 400 万辆", "https://byd.com", "比亚迪 销量 400 万")]
  with _patch(html_rows=html,
              browser_rows=_mk([_row("无关页", "https://x.com", "完全不相干的内容")])):
    out = W.web_search("比亚迪 2025 年销量", max_results=5, fetch_top=0)
  assert out[0]["title"].startswith("比亚迪"), "浏览器更差时应保留 httpx 结果"


def test_never_raises_when_browser_backend_explodes():
  """浏览器层抛异常 -> web_search 仍然返回 httpx 结果，不把上层带崩。"""
  import app.tools.web_search as W
  html = [_row("Microsoft", "https://ms.com", "cloud")]
  with _patch(html_rows=html, browser_raises=True):
    out = W.web_search("上海 2026 年落户政策", max_results=5, fetch_top=0)
  assert out and out[0]["title"] == "Microsoft", out


def test_empty_query_returns_empty():
  import app.tools.web_search as W
  with _patch(html_rows=[_row("x", "https://a.com", "y")]):
    assert W.web_search("") == []
    assert W.web_search("   ") == []


# ============ 4. 注册表读取 ============

def _write_registry(servers: list) -> Path:
  d = Path(_TMP) / "mcp"
  d.mkdir(parents=True, exist_ok=True)
  p = d / "servers.json"
  p.write_text(json.dumps({"version": 1, "servers": servers}), encoding="utf-8")
  return p


def test_find_playwright_server_requires_enabled():
  """只认已启用的 playwright；禁用/别的预设都不算。"""
  from app.tools import browser_search as B
  _write_registry([
    {"id": "p1", "preset_id": "playwright", "name": "Playwright", "enabled": False,
     "command": "npx", "args": [], "env": {}},
    {"id": "f1", "preset_id": "filesystem", "name": "Filesystem", "enabled": True,
     "command": "npx", "args": [], "env": {}},
  ])
  assert B.find_playwright_server() is None, "禁用的不算"
  assert B.available() is False

  _write_registry([
    {"id": "p1", "preset_id": "playwright", "name": "Playwright", "enabled": True,
     "command": "npx", "args": ["-y", "@playwright/mcp@latest"], "env": {}},
  ])
  found = B.find_playwright_server()
  assert found and found["id"] == "p1", found
  assert B.available() is True


def test_find_playwright_server_tolerates_missing_registry():
  """注册表不存在/损坏 -> 返回 None，不抛异常。"""
  from app.tools import browser_search as B
  p = Path(_TMP) / "mcp" / "servers.json"
  if p.exists():
    p.unlink()
  assert B.find_playwright_server() is None
  p.parent.mkdir(parents=True, exist_ok=True)
  p.write_text("{ 这不是合法 json", encoding="utf-8")
  assert B.find_playwright_server() is None


def test_browser_search_returns_empty_without_playwright():
  """没装 Playwright 时 browser_search 返回 []，不抛异常。"""
  from app.tools import browser_search as B
  p = Path(_TMP) / "mcp" / "servers.json"
  if p.exists():
    p.unlink()
  B.close()
  assert B.browser_search("任意查询") == []


# ============ 5. MCP 返回解析 ============

def test_parse_evaluate_result_handles_mcp_wrapper():
  """MCP 返回带 markdown 包装 + 转义引号，必须能正确抠出数组。"""
  from app.tools.browser_search import _parse_evaluate_result
  payload = [{"title": "标题 A", "url": "https://a.com", "snippet": "摘要 A"},
             {"title": "标题 B", "url": "https://b.com", "snippet": "摘要 B"}]
  inner = json.dumps(payload, ensure_ascii=False)
  raw = '### Result\n%s\n### Ran Playwright code\n```js\nawait page.evaluate(...)\n```' % json.dumps(inner, ensure_ascii=False)
  got = _parse_evaluate_result(raw, 5)
  assert len(got) == 2, got
  assert got[0]["title"] == "标题 A"
  assert got[1]["url"] == "https://b.com"


def test_parse_evaluate_result_tolerates_plain_json():
  """没有包装、直接是数组时也能解析。"""
  from app.tools.browser_search import _parse_evaluate_result
  raw = json.dumps([{"title": "T", "url": "https://a.com", "snippet": "S"}])
  got = _parse_evaluate_result(raw, 5)
  assert len(got) == 1 and got[0]["title"] == "T"


def test_parse_evaluate_result_returns_empty_on_garbage():
  """解析不出来返回 []，不抛异常（上层会降级回 httpx 结果）。"""
  from app.tools.browser_search import _parse_evaluate_result
  assert _parse_evaluate_result("", 5) == []
  assert _parse_evaluate_result("完全不是 JSON", 5) == []
  assert _parse_evaluate_result("### Result\n\n### Ran Playwright", 5) == []


def test_parse_evaluate_result_respects_limit():
  from app.tools.browser_search import _parse_evaluate_result
  payload = [{"title": "T%d" % i, "url": "https://a.com", "snippet": "S"} for i in range(10)]
  got = _parse_evaluate_result(json.dumps(payload), 3)
  assert len(got) == 3, got


# ============ 6. 并发安全 ============

def test_browser_search_serializes_concurrent_calls():
  """并发调用必须串行 —— MCPSession 的 _request 是先写后读、无锁的。

  不串行的后果不是报错，而是**静默串线**：A 的 navigate 结果被 B 的 evaluate
  读走，拿到别的查询的结果。这比崩溃更难发现。

  verify 是同步节点，LangGraph 会放进线程池执行，所以并发请求真的会撞上。
  """
  import threading
  import time
  from app.tools import browser_search as B

  active = {"n": 0, "max": 0}
  guard = threading.Lock()

  class FakeSession:
    alive = True

    def call(self, name, args):
      with guard:
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
      time.sleep(0.05)                      # 模拟 RPC 往返
      with guard:
        active["n"] -= 1
      if name == "browser_evaluate":
        return ("### Result\n"
                + json.dumps([{"title": "T", "url": "https://a.com", "snippet": "S"}])
                + "\n### Ran Playwright code")
      return ""

    def close(self):
      pass

  orig_get = B._get_session
  B._get_session = lambda: FakeSession()
  try:
    out: list = []
    out_guard = threading.Lock()

    def worker(i):
      r = B.browser_search("查询 %d" % i)
      with out_guard:
        out.append(r)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
  finally:
    B._get_session = orig_get

  assert active["max"] == 1, \
    "并发调用发生了交错（同时活跃 %d 个），RPC 响应会串线" % active["max"]
  assert len(out) == 4 and all(out), out


def test_get_session_returns_none_when_handshake_fails():
  """握手失败时不得缓存坏会话（否则会一直失败下去）。"""
  from app.tools import browser_search as B

  class DeadSession:
    alive = True

    def list_tools(self):
      return []

    def close(self):
      self.closed = True

  closed = {"n": 0}

  class _Dead(DeadSession):
    def close(self):
      closed["n"] += 1

  orig_find = B.find_playwright_server
  orig_cls = B.MCPSession
  B.find_playwright_server = lambda: {
    "id": "p1", "preset_id": "playwright", "enabled": True,
    "command": "npx", "args": [], "env": {}}
  B.MCPSession = lambda *a, **k: _Dead()
  B._session = None
  try:
    assert B._get_session() is None, "握手失败应返回 None"
    assert B._session is None, "不得缓存坏会话"
    assert closed["n"] == 1, "应关掉那个坏会话"
  finally:
    B.find_playwright_server = orig_find
    B.MCPSession = orig_cls
    B._session = None


def test_mcp_session_serializes_concurrent_requests():
  """MCPSession 的「写请求 -> 读响应」必须原子。

  会话是**池化共享**的（mcp_tools._sessions、browser_search._session 都是
  模块级），两个并发请求会拿到同一个 session。不串行的后果是**静默串线**：
  A 的结果被 B 读走，不报错，只是返回了别人的数据。
  """
  import threading
  import time
  from app.agent.tools import mcp_client as M

  active = {"n": 0, "max": 0}
  guard = threading.Lock()
  written: list = []

  orig_write, orig_read = M._write_message, M._read_message

  def fake_write(proc, msg):
    with guard:
      written.append(msg.get("id"))

  def fake_read(proc, timeout):
    with guard:
      active["n"] += 1
      active["max"] = max(active["max"], active["n"])
    time.sleep(0.03)                      # 模拟 RPC 往返
    with guard:
      active["n"] -= 1
    return {"jsonrpc": "2.0", "id": 0,
            "result": {"content": [{"type": "text", "text": "ok"}]}}

  M._write_message, M._read_message = fake_write, fake_read
  try:
    spec = M.MCPServerSpec(server_id="t", command="x", args=[], env={})
    sess = M.MCPSession(spec)
    sess._ensure_started = lambda: True   # 跳过真实 spawn
    sess._proc = None

    out: list = []
    out_guard = threading.Lock()

    def worker():
      r = sess.call("some_tool", {})
      with out_guard:
        out.append(r)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
  finally:
    M._write_message, M._read_message = orig_write, orig_read

  assert active["max"] == 1, \
    "并发请求交错（读峰值 %d），响应会串线" % active["max"]
  assert len(out) == 5 and all(r == "ok" for r in out), out
  assert len(set(written)) == 5, "请求 id 必须唯一，否则无法配对响应: %s" % written


# ============ 7. 判据模块 ============

def test_search_quality_module_is_shared():
  """判据抽到 search_quality 后，verify 侧仍能用同名入口（兼容既有测试）。"""
  from app.tools.search_quality import is_relevant, query_terms, relevance_ratio
  from app.agent.nodes.verify import _query_terms, _web_material_is_relevant
  assert _query_terms is query_terms
  assert _web_material_is_relevant is is_relevant
  assert relevance_ratio("上海 2026 年落户政策",
                         _mk([_row("上海落户新政策", "https://a.com",
                                   "上海 2026 年落户政策 最新")])) > 0.5


def test_relevance_ratio_none_for_symbol_only_query():
  """纯符号查询抽不出词 -> 比例 None（调用方据此不拦）。"""
  from app.tools.search_quality import relevance_ratio
  assert relevance_ratio("？？？", _mk([_row("x", "https://a.com", "y")])) is None


# ============ runner ============

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
