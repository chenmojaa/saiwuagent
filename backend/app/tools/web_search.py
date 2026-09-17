"""Web search fallback — 知识库无结果时的联网兜底。

设计约束（重要）：
  * **结果是一次性的**。只存在于本轮 graph state 与当轮对话记录里，
    **绝不写回知识库**（不碰 notes / chunk_fts / 向量库）。
    这是产品的明确边界：联网内容不能被"回填"成用户资料。
  * **永不抛异常**。任何失败都返回 []，由调用方决定降级。
  * 免 API key：默认走 Bing 的搜索结果页。

关于搜索引擎的选择（实测记录，2026-09）：
  * DuckDuckGo HTML / Lite —— **已被人机验证拦截**（返回 202 + "Select all
    squares containing a duck"），不可用。曾偶发成功，属运气，不要依赖。
  * searx.be —— 同样被拦。
  * **Bing —— 可用**，返回 10 条可解析结果。但它是 HTML 抓取，
    结构或反爬策略变化都可能失效，属于已知脆弱点。

**关于 Bing 的泛化问题（2026-09-17 补充）**：
  httpx 抓 Bing 对**无 cookie 客户端**会返回泛化结果。同一个查询
  「腾讯控股 2025 年营收」：
    - 本模块（httpx）：腾讯视频频道 / 腾讯网 / Tencent 官网
    - 真实浏览器：腾讯控股2025财报解析：营收7518亿净利2248亿 …（正确）
  排查过并已排除：query 编码正确、解析正确、mkt/ensearch 参数无效、
  trust_env=False 绕代理无效、英文查询同样泛化。**是 Bing 侧行为。**
  对策见下方 web_search() 的「自动升级」：跑偏时改用 tools/browser_search.py
  的 Playwright 真实浏览器重搜。根治则需换搜索 API（Brave / Serper / Tavily）。

返回的 dict 结构与 hybrid_search 的 chunk 对齐，便于直接塞进
retrieved_chunks 走同一条答案链路（source_type 固定为 "web"）。
"""
from __future__ import annotations

import base64
import logging
import re
import urllib.parse

import httpx

from app.config import settings

_log = logging.getLogger(__name__)

_BING = "https://www.bing.com/search"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)
_DEFAULT_TIMEOUT = 15.0
_MAX_SNIPPET = 800


def _unwrap_bing_url(href: str) -> str:
    """Bing 的结果链接是跳转包装：/ck/a?...&u=a1<base64url>。

    解出真实 URL；解不出就原样返回（调用方会做 http 前缀校验）。
    """
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    try:
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
        u = (qs.get("u") or [""])[0]
        if u.startswith("a1"):
            raw = u[2:]
            raw += "=" * (-len(raw) % 4)          # 补齐 base64 padding
            decoded = base64.urlsafe_b64decode(raw).decode("utf-8", "ignore")
            if decoded.startswith("http"):
                return decoded
    except Exception:
        pass
    return href


def _search_raw(query: str, limit: int, timeout: float) -> list[dict]:
    """抓 Bing 搜索页并解析。失败返回 []。"""
    try:
        from bs4 import BeautifulSoup
    except Exception as e:  # pragma: no cover - bs4 是既有依赖
        _log.warning("web_search: beautifulsoup4 不可用: %s", e)
        return []

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(
                _BING,
                params={"q": query, "setlang": "zh-CN", "count": str(max(10, limit))},
                headers={"User-Agent": _UA,
                         "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            )
            if resp.status_code >= 400:
                _log.warning("web_search: Bing 返回 %s", resp.status_code)
                return []
            page = resp.text
    except Exception as e:
        _log.warning("web_search: 搜索请求失败 q=%r: %s", query[:60], e)
        return []

    try:
        soup = BeautifulSoup(page, "html.parser")
    except Exception as e:
        _log.warning("web_search: HTML 解析失败: %s", e)
        return []

    out: list[dict] = []
    for li in soup.select("li.b_algo"):
        h2 = li.find("h2")
        a = h2.find("a") if h2 else None
        if a is None:
            continue
        url = _unwrap_bing_url(a.get("href") or "")
        if not url.startswith("http"):
            continue
        title = a.get_text(" ", strip=True)
        if not title:
            continue
        p = li.find("p")
        snippet = p.get_text(" ", strip=True) if p is not None else ""
        out.append({"title": title, "url": url, "snippet": snippet})
        if len(out) >= limit:
            break
    return out


def _fetch_page_text(url: str) -> str:
    """抓正文；失败返回空串（由调用方回退到搜索摘要）。"""
    try:
        from app.tools.fetch_url import fetch_url
        data = fetch_url(url)
        return (data or {}).get("content") or ""
    except Exception as e:
        _log.info("web_search: 正文抓取失败 %s: %s", url[:80], e)
        return ""


def web_search(query: str,
               max_results: int = 5,
               fetch_top: int = 2,
               timeout: float = _DEFAULT_TIMEOUT,
               allow_browser: bool = True) -> list[dict]:
    """联网搜索，返回可直接当 chunk 用的 dict 列表。

    fetch_top: 前 N 条额外抓正文（内容更完整，但更慢）。0 = 只用搜索摘要。

    **自动升级**：httpx 抓 Bing 对无 cookie 客户端会返回泛化结果（实测
    「腾讯控股 2025 年营收」只拿到腾讯视频/腾讯网这类实体导航页）。所以这里
    加了一道质量闸门 —— 结果为空、或与查询的相关性过低时，自动改用真实浏览器
    （Playwright MCP，见 tools/browser_search.py）重搜一次。

    为什么不直接只用浏览器：httpx 路径 0.7s，浏览器 5-12s。多数查询 httpx
    够用，只在它跑偏时才付这个代价。没装/没启用 Playwright MCP 时静默跳过升级。

    allow_browser=False 可强制只走 httpx（测试与排查用）。

    绝不抛异常；任何失败都返回 []。
    """
    q = (query or "").strip()
    if not q:
        return []

    chunks = _html_search(q, max_results, fetch_top, timeout)

    if not allow_browser or not settings.web_search_browser_fallback:
        return chunks

    # ---- 质量闸门：跑偏就升级到真实浏览器 ----
    from app.tools.search_quality import is_relevant, relevance_ratio
    if chunks and is_relevant(q, chunks):
        return chunks

    try:
        from app.tools.browser_search import available, browser_search
    except Exception as e:              # 缺依赖/导入失败都不该影响主路径
        _log.warning("web_search: 浏览器兜底不可用: %s", e)
        return chunks
    if not available():
        _log.info("web_search: HTML 结果不理想但没有可用的 Playwright MCP，沿用原结果")
        return chunks

    _log.info("web_search: HTML 结果不理想（%d 条，相关比例 %s），尝试浏览器重搜 q=%r",
              len(chunks), "n/a" if not chunks else "%.2f" % (relevance_ratio(q, chunks) or 0),
              q[:60])
    # 升级是「尽力而为」的增强，不能有能力把主路径带崩 —— browser_search 自身
    # 承诺不抛，但这里再包一层：万一它将来破了约定，最差也只是沿用 httpx 结果。
    try:
        alt = browser_search(q, max_results=max_results)
    except Exception as e:
        _log.warning("web_search: 浏览器重搜异常，沿用 HTML 结果: %s", e)
        return chunks
    if not alt:
        return chunks
    # 浏览器结果也没更相关时，保留原结果 —— 不能因为「换了后端」就更差
    if chunks and (relevance_ratio(q, alt) or 0) <= (relevance_ratio(q, chunks) or 0):
        _log.info("web_search: 浏览器结果未更相关，沿用 HTML 结果")
        return chunks
    _log.info("web_search: 已切换到浏览器结果（%d 条）", len(alt))
    return alt


def _html_search(query: str, max_results: int, fetch_top: int,
                 timeout: float) -> list[dict]:
    """httpx 抓 Bing 搜索页 + 抽取正文。这是原始实现，作为快速主路径。"""
    q = (query or "").strip()
    if not q:
        return []

    rows = _search_raw(q, max_results, timeout)
    if not rows:
        _log.info("web_search: 无结果 q=%r", q[:60])
        return []

    chunks: list[dict] = []
    for i, r in enumerate(rows):
        text = _fetch_page_text(r["url"]) if i < fetch_top else ""
        if not text:
            text = r["snippet"]
        if not text:
            continue
        chunks.append({
            # note_id 加 "web:" 前缀，一眼能看出不是知识库内容；
            # 也保证不会和真实 note_id 撞车。
            "note_id": "web:%d:%s" % (i, r["url"][:120]),
            "title": r["title"],
            "chunk_index": 0,
            "text": text[:_MAX_SNIPPET],
            "source_type": "web",
            "source_url": r["url"],
            # 联网结果没有知识库那种融合分；给一个稳定的中等值，
            # 只用于排序与展示，不参与阈值判断。
            "final_score": 0.5,
            "matched_query": q,
        })
    _log.info("web_search: q=%r -> %d 条", q[:60], len(chunks))
    return chunks
