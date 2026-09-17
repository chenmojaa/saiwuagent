# -*- coding: utf-8 -*-
"""浏览器搜索后端 —— 通过 Playwright MCP 驱动**真实浏览器**联网搜索。

## 为什么需要它

`web_search.py` 用 httpx 直接抓 Bing 的 HTML。它对**无 cookie 客户端**会返回
泛化结果。2026-09-17 实测同一个查询「腾讯控股 2025 年营收」：

| 方式 | 结果 |
|---|---|
| httpx 抓 HTML | 腾讯视频频道 / 腾讯网 / Tencent 官网（全是实体导航页） |
| 真实浏览器 | 腾讯控股2025财报解析：营收7518亿净利2248亿<br>腾讯2025年营收7518亿元，利润2596.26亿元<br>腾讯公布二零二五年度及第四季业绩 |

同一个 Bing，差别在于浏览器带完整 cookie、JS 环境和真实 UA —— Bing 才肯给
真正的搜索结果。排查过并已排除：query 编码（URL 里是正确的 UTF-8）、解析
（`li.b_algo` 稳定 10 条）、`mkt`/`ensearch` 参数、代理（`trust_env=False` 无效）、
英文查询同样泛化。**是 Bing 侧行为，不是我们这边能修的。**

## 定位

* **兜底，不是主路径**。正常情况仍走 httpx（0.7s vs 浏览器 5-10s）。
  只有 httpx 的结果过不了相关性闸门时才升级到这里。
* **永不抛异常**。任何失败返回 []，由调用方决定降级。
* 没装/没启用 Playwright MCP 时静默返回 []，功能自动退化为旧行为。

## 依赖

需要注册表里有一个**已启用**的 Playwright MCP（预设 id = `playwright`）。
没有就返回 []，不会尝试自己装浏览器。
"""
from __future__ import annotations

import json
import logging
import re
import threading
import urllib.parse
from pathlib import Path

from app.config import settings
from app.agent.tools.mcp_client import MCPServerSpec, MCPSession

_log = logging.getLogger(__name__)

# 预设 id。api/mcp.py 的 MCP_PRESETS 里 playwright 用的就是这个 id。
_PLAYWRIGHT_PRESET = "playwright"

_MAX_SNIPPET = 800
_NAV_TIMEOUT = 45.0
_CALL_TIMEOUT = 60.0

# 长连接复用：npx 冷启动 ~4s + 浏览器启动 ~2s，每次搜索都重来太浪费。
# 一个进程可能服务很多轮对话，所以按进程缓存。
_session_lock = threading.Lock()
_session: MCPSession | None = None

# 调用锁：MCPSession **不是线程安全的** —— 它的 _request() 是「写请求 -> 读响应」，
# 没有把这两步做成原子操作（见 agent/tools/mcp_client.py）。两个线程同时用它，
# 响应就会串线：A 的 navigate 结果被 B 的 evaluate 读走。
#
# 这个风险在 browser_search 里比在 mcp_tools 里更严重，原因有二：
#   1. mcp_tools 的会话只在**单轮**内使用，调用是串行的；这里跨请求共享。
#   2. 一次 browser_search 要发**两个** RPC（navigate + evaluate），
#      交错的机会翻倍，而且结果会静默错乱（拿到别的查询的结果）。
#
# verify 是同步节点，LangGraph 会把它放进线程池跑，所以并发请求真的会撞上。
# 因此这里把「一次搜索」整体串行化 —— 浏览器搜索本来就慢（5-12s）且稀少，
# 串行的代价可以忽略，正确性优先。
_call_lock = threading.Lock()

# 提取搜索结果的 JS。用 JSON.stringify 而不是自己拼分隔符 —— MCP 返回的是
# JSON 字符串，换行/引号会变成字面量转义，手工解析很容易出错（第一版就踩了）。
_EXTRACT_JS = """() => JSON.stringify(
  Array.from(document.querySelectorAll('li.b_algo')).slice(0, %d).map(li => {
    const a = li.querySelector('h2 a');
    const p = li.querySelector('p');
    return {
      title: a ? a.innerText.trim() : '',
      url: a ? (a.href || '') : '',
      snippet: p ? p.innerText.trim() : ''
    };
  })
)"""


def _registry_path() -> Path:
  return Path(settings.data_dir) / "mcp" / "servers.json"


def find_playwright_server() -> dict | None:
  """注册表里第一个已启用的 Playwright 服务。没有返回 None。

  直接读 JSON 而不复用 api/mcp.py 的 _read_servers()：tool 层不该依赖 api 层
  （会形成反向依赖，也让工具模块没法在没有 web 框架时单独使用）。
  """
  path = _registry_path()
  if not path.is_file():
    return None
  try:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
  except (OSError, json.JSONDecodeError) as e:
    _log.warning("browser_search: 读 MCP 注册表失败: %s", e)
    return None
  items = payload.get("servers", []) if isinstance(payload, dict) else []
  for item in items:
    if not isinstance(item, dict) or not item.get("enabled"):
      continue
    if str(item.get("preset_id") or "") == _PLAYWRIGHT_PRESET:
      return item
  return None


def available() -> bool:
  """浏览器搜索是否可用（注册表里有已启用的 Playwright）。"""
  return find_playwright_server() is not None


def _get_session() -> MCPSession | None:
  """取（或建）长连接会话。失败返回 None。"""
  global _session
  with _session_lock:
    if _session is not None and _session.alive:
      return _session
    entry = find_playwright_server()
    if entry is None:
      return None
    try:
      spec = MCPServerSpec.from_registry_entry(entry)
    except Exception as e:
      _log.warning("browser_search: 服务配置无效: %s", e)
      return None
    session = MCPSession(spec, cwd=None, init_timeout=_NAV_TIMEOUT,
                         call_timeout=_CALL_TIMEOUT)
    # 先握一次手，确认真的能用；否则缓存一个坏会话会一直失败。
    if not session.list_tools():
      _log.warning("browser_search: Playwright MCP 握手失败，浏览器搜索不可用")
      session.close()
      return None
    _session = session
    return _session


def close() -> None:
  """释放浏览器会话（进程退出或测试清理时用）。"""
  global _session
  with _session_lock:
    if _session is not None:
      try:
        _session.close()
      except Exception:
        pass
      _session = None


def _unwrap_bing_url(href: str) -> str:
  """Bing 的结果链接是 /ck/a?...&u=a1<base64url> 跳转包装。

  复用 web_search 里的实现，避免两处逻辑漂移。
  """
  from app.tools.web_search import _unwrap_bing_url as unwrap
  return unwrap(href)


def _parse_evaluate_result(raw: str, limit: int) -> list[dict]:
  """从 browser_evaluate 的返回文本里抠出结果数组。

  MCP 的返回长这样（content 是 JSON 编码过的字符串）：
      ### Result
      "[{\\"title\\":...}]"
      ### Ran Playwright code
  所以取 ### Result 之后、### Ran Playwright 之前那段，剥引号后 json.loads。
  解析不出来就退回「找第一个 [ 到最后一个 ]」—— 不同版本的包装格式略有差异。
  """
  if not raw:
    return []
  body = raw.split("### Result", 1)[-1]
  body = body.split("### Ran Playwright", 1)[0].strip()
  if not body:
    return []

  candidates = [body]
  if body.startswith('"') and body.endswith('"'):
    try:
      candidates.append(json.loads(body))      # 外层 JSON 字符串 -> 内层文本
    except Exception:
      pass
  start, end = body.find("["), body.rfind("]")
  if start >= 0 and end > start:
    candidates.append(body[start:end + 1])

  for text in candidates:
    if not isinstance(text, str):
      continue
    try:
      rows = json.loads(text)
    except Exception:
      continue
    if isinstance(rows, list):
      return [r for r in rows if isinstance(r, dict)][:limit]
  _log.info("browser_search: 无法解析 evaluate 结果，前 120 字: %r", body[:120])
  return []


def browser_search(query: str, max_results: int = 5,
                   fetch_top: int = 0) -> list[dict]:
  """用真实浏览器搜一次，返回与 ``web_search`` 对齐的 chunk 列表。

  fetch_top 保留是为了接口一致，但浏览器路径**不额外抓正文** ——
  Bing 结果页的摘要已经够用，再逐条开页面会让耗时翻几倍。
  调用方真要正文可以拿到 source_url 后自己走 fetch_url。

  永不抛异常；不可用或失败一律返回 []。
  """
  q = (query or "").strip()
  if not q:
    return []

  session = _get_session()
  if session is None:
    return []

  url = "https://www.bing.com/search?q=" + urllib.parse.quote(q)
  # navigate + evaluate 必须在同一把锁里完成，否则并发时两个请求的
  # RPC 会交错，拿到彼此的结果（见 _call_lock 的说明）。
  # _get_session() 放在锁外：它内部的握手很慢，没必要把并发调用都堵在那上面。
  try:
    with _call_lock:
      session.call("browser_navigate", {"url": url})
      raw = session.call("browser_evaluate",
                         {"function": _EXTRACT_JS % max(1, int(max_results))})
  except Exception as e:
    _log.warning("browser_search: 搜索失败 q=%r: %s", q[:60], e)
    return []

  rows = _parse_evaluate_result(raw, max_results)
  if not rows:
    _log.info("browser_search: 无结果 q=%r", q[:60])
    return []

  chunks: list[dict] = []
  for i, r in enumerate(rows):
    title = (r.get("title") or "").strip()
    if not title:
      continue
    page_url = _unwrap_bing_url((r.get("url") or "").strip())
    text = (r.get("snippet") or "").strip()
    if not text:
      continue
    chunks.append({
      # note_id 加 "web:" 前缀，与 web_search 保持一致：一眼能看出不是知识库内容，
      # 也保证不会和真实 note_id 撞车。browser 段用来区分来源后端。
      "note_id": "web:b%d:%s" % (i, page_url[:120]),
      "title": title,
      "chunk_index": 0,
      "text": text[:_MAX_SNIPPET],
      "source_type": "web",
      "source_url": page_url,
      "final_score": 0.5,
      "matched_query": q,
    })
  _log.info("browser_search: q=%r -> %d 条", q[:60], len(chunks))
  return chunks
