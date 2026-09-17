# -*- coding: utf-8 -*-
"""联网校验节点 —— 策略 A：优先知识库 + 联网做校验。

与 web_search 兜底的区别（两者都存在，别混淆）：
  * **兜底**（research.py:replan_node）：知识库**完全搜不到**时才联网，
    目的是「没有材料可用时找点材料」。每轮最多一次。
  * **校验**（本模块）：知识库**有结果时也联网**，目的是「核对时效性与
    事实冲突」。这是策略 A 的核心 —— 旧实现里知识库一旦命中就再也不会
    联网，过期的知识会被原样当成事实输出。

裁决结果与去向：
  | status      | 含义                     | 去向                          |
  |-------------|--------------------------|-------------------------------|
  | consistent  | 知识库与联网一致         | 以知识库为主作答               |
  | conflict    | 存在事实性冲突           | 作答但**强制挂冲突警告**       |
  | kb_stale    | 知识库明显过期           | 剔除过期片段后作答             |
  | web_only    | 知识库无相关材料         | 仅凭联网结果作答               |
  | unverified  | 联网失败/裁决不可用      | 照常作答，标注「未能核对」     |
  | skipped     | 非时效敏感 / 模式跳过     | 照常作答                       |
  | disabled    | 总开关关闭               | 照常作答                       |

**防污染边界**：联网结果只进两处 —— 本轮 retrieved_chunks（供作答引用）
和待审核候选库（供人工审批）。**绝不写 notes / chunk_fts / 向量库**。

失败语义：本节点**绝不抛异常、绝不阻断作答**。校验是增强，不是门禁；
它坏掉时最差退化成「旧行为」，而不是「问不了问题」。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.context import strip_think
from app.agent.state import AgentState
from app.config import settings
from app.llm.factory import _build_model
from app.tools.search_quality import is_relevant, query_terms, relevance_ratio

_log = logging.getLogger(__name__)

# 裁决提示词。要求严格 JSON 输出，因为下游要按字段做分支 —— 自然语言
# 结论没法可靠地映射到 status。输出里带 refs，是为了让「剔除过期片段」
# 和「列出冲突双方原话」能精确定位到具体材料，而不是整篇丢掉。
ADJUDICATE_PROMPT = """你是事实核对员。针对同一个问题，下面有两组材料。

【知识库片段】本地资料，视为可信但**可能过期**：
<<KB>>

【联网结果】实时抓取，但可能不权威、可能含噪声或广告：
<<WEB>>

问题：<<QUESTION>>

请判断联网结果与知识库的关系，只输出一个 JSON 对象，不要任何解释或 markdown 代码块：

{
  "status": "consistent" | "conflict" | "kb_stale" | "insufficient",
  "conflicts": [
    {"claim": "争议点（一句话）", "kb_says": "知识库的说法", "web_says": "联网的说法",
     "kb_refs": [1, 2], "web_refs": ["A"]}
  ],
  "stale_kb_refs": [1],
  "note": "一句话说明判断依据"
}

判定规则：
- "consistent"：两边讲的是同一事实且不矛盾。**联网结果与问题无关、或信息量不足以核对时，也算 consistent**（不要硬找冲突）。
- "conflict"：同一事实给出了**不同数值、不同结论或不同政策口径**。仅「时间更新」不算冲突，那是 kb_stale。
- "kb_stale"：知识库内容被联网结果**明确取代**（年份、政策、价格、人事、版本等已变更）。stale_kb_refs 填被取代的片段编号。
- "insufficient"：联网结果完全无法用于核对（全是无关内容/广告/抓取失败）。
- kb_refs 填知识库片段编号（数字），web_refs 填联网结果字母。没有就给空数组。
- 只依据上面给出的材料判断，不要引入你自己的记忆。"""

_YEAR_RE = re.compile(r"(19|20)\d{2}\s*年?")
# 时效敏感信号。stale_only 模式靠它决定要不要联网 —— 命中任一即联网。
# 故意保守（宁可多联网也别漏），因为漏掉一条过期政策的代价远大于多花几秒。
_TIME_SENSITIVE_RE = re.compile(
    r"最新|目前|当前|现在|现任|近期|今年|去年|明年|截至|"
    r"政策|法规|税率|利率|汇率|股价|市值|营收|利润|排名|榜单|"
    r"价格|报价|费用|收费|标准|版本|v?\d+\.\d+|"
    r"(19|20)\d{2}\s*年?"
)


def _build_adjudicator(state: AgentState):
  """裁决用的模型：廉价档（router 档），失败返回 None。"""
  try:
    return _build_model(
      provider=None,
      model=(settings.web_verify_model or settings.router_model
             or state.get("model_override")),
      api_key=state.get("api_key_override"),
      base_url=settings.router_base_url or state.get("base_url_override") or None,
      reasoning_level=None,
    )
  except Exception as e:
    _log.warning("verify: 裁决模型初始化失败: %s", e)
    return None


def _extract_json(text: str) -> dict | None:
  """从模型回复里抠出 JSON 对象。

  必须剥思维链：推理模型的响应以 <think> 开头，里面经常带示例 JSON，
  直接取第一个 { 会拿到示例而不是结论（follow-up 生成踩过同样的坑）。
  """
  cleaned = strip_think(text or "")
  if not cleaned:
    return None
  start = cleaned.find("{")
  end = cleaned.rfind("}")
  if start < 0 or end <= start:
    return None
  try:
    obj = json.loads(cleaned[start:end + 1])
  except Exception:
    return None
  return obj if isinstance(obj, dict) else None


def _refs_to_indices(raw: Any, upper: int) -> list[int]:
  """把模型给的片段编号规整成 0-based 下标，越界/非法一律丢掉。

  模型偶尔会返回 "1,2" 或 "第1条" 这种脏数据，静默丢弃比抛异常好 ——
  这个节点的定位是「绝不阻断作答」。
  """
  out: list[int] = []
  if not isinstance(raw, list):
    return out
  for item in raw:
    n = None
    if isinstance(item, int):
      n = item
    elif isinstance(item, str):
      m = re.search(r"\d+", item)
      if m:
        n = int(m.group(0))
    if n is None:
      continue
    idx = n - 1          # 提示词里用的是 1-based 编号
    if 0 <= idx < upper and idx not in out:
      out.append(idx)
  return out


def _looks_time_sensitive(query: str, kb_chunks: list[dict]) -> bool:
  """知识库材料里是否含时效敏感信号（年份/价格/政策/人事…）。

  只看知识库片段正文，不看问题 —— 问题问「介绍一下 X」本身不含时效词，
  但如果知识库里写的是「2021 年 X 的政策是……」，那它就是过期的。
  """
  if _TIME_SENSITIVE_RE.search(query or ""):
    return True
  for c in kb_chunks[: int(settings.web_verify_max_kb_chunks)]:
    if _TIME_SENSITIVE_RE.search(str(c.get("text") or "")[:600]):
      return True
  return False


# ---- 联网结果相关性闸门 ----
#
# 判据本体已抽到 tools/search_quality.py（web_search 也要用它来决定是否升级到
# 浏览器搜索，放在这里会形成 web_search -> verify -> web_search 的循环导入）。
#
# 为什么必须有这道闸门：Bing 的 HTML 抓取对无 cookie 客户端会返回**泛化结果**。
# 实测（2026-09-17）：查「腾讯控股 2025 年营收」返回腾讯视频/腾讯网；
# 查「新能源汽车补贴政策 最新」返回汉字「新」的百科词条。
#
# 危害不在于「搜不到」，而在于**搜到的垃圾会让裁决模型误报冲突**：
# 它看到知识库说 A、一堆无关网页说 B，就会判 conflict，用户于是开始怀疑
# 本来正确的知识库内容。这比不校验更糟。
#
# 注意：现在 web_search 内部已经会自动升级到浏览器搜索（见 tools/browser_search.py），
# 所以这道闸门更多是兜底 —— 浏览器也拿不到相关结果时才会落到 unverified。
_query_terms = query_terms
_web_material_is_relevant = is_relevant


def _log_relevance_reject(query: str, web_chunks: list[dict]) -> None:
  """闸门拦下时打日志。用 WARNING 是因为这通常意味着搜索后端出问题了。"""
  ratio = relevance_ratio(query, web_chunks)
  _log.warning("verify: 联网结果与问题相关性过低（命中比例 %s），不用于裁决。"
               "q=%r 首条=%r", "n/a" if ratio is None else "%.2f" % ratio,
               (query or "")[:60], (web_chunks[0].get("title") or "")[:60])


def _format_kb(kb_chunks: list[dict], limit: int) -> str:
  lines: list[str] = []
  for i, c in enumerate(kb_chunks[:limit], start=1):
    title = c.get("title") or c.get("note_id") or "未命名"
    text = re.sub(r"\s+", " ", str(c.get("text") or "")).strip()[:600]
    lines.append("%d. [%s] %s" % (i, title, text))
  return "\n".join(lines) or "（无）"


def _format_web(web_chunks: list[dict], limit: int) -> str:
  lines: list[str] = []
  for i, c in enumerate(web_chunks[:limit]):
    letter = chr(ord("A") + i)
    title = c.get("title") or ""
    url = c.get("source_url") or ""
    text = re.sub(r"\s+", " ", str(c.get("text") or "")).strip()[:600]
    lines.append("%s. [%s] %s\n   %s" % (letter, title, url, text))
  return "\n".join(lines) or "（无）"


def _adjudicate(kb_chunks: list[dict], web_chunks: list[dict], query: str,
                state: AgentState) -> dict | None:
  """让廉价模型裁决。失败/解析不出返回 None。"""
  chat = _build_adjudicator(state)
  if chat is None:
    return None
  payload = (
    ADJUDICATE_PROMPT
    .replace("<<KB>>", _format_kb(kb_chunks, int(settings.web_verify_max_kb_chunks)))
    .replace("<<WEB>>", _format_web(web_chunks, int(settings.web_verify_max_web_results)))
    .replace("<<QUESTION>>", query[:500])
  )
  try:
    resp = chat.invoke(payload)
    text = getattr(resp, "content", None) or str(resp)
    if isinstance(text, list):     # LangChain 的 content 可能是分片列表
      text = "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in text
      )
    return _extract_json(str(text or ""))
  except Exception as e:
    _log.warning("verify: 裁决调用失败: %s", e)
    return None


def verify_node(state: AgentState) -> dict:
  """联网校验主节点。"""
  base = {"step_count": int(state.get("step_count") or 0) + 1}

  if not settings.web_verify_enabled:
    return {**base, "web_verify_status": "disabled"}

  query = (state.get("rewritten_query") or state.get("query") or "").strip()
  if not query:
    return {**base, "web_verify_status": "skipped"}

  chunks = list(state.get("retrieved_chunks") or [])
  kb_chunks = [c for c in chunks if c.get("source_type") != "web"]
  web_chunks = [c for c in chunks if c.get("source_type") == "web"]
  # 来自 state 的联网片段**已经在 chunks 里**（research 兜底路径抓的），
  # 新抓的还没有。这个标记决定后面要不要合并，避免重复插入同一批片段。
  web_from_state = bool(web_chunks)

  mode = (settings.web_verify_mode or "always").strip().lower()
  if mode == "stale_only" and kb_chunks and not _looks_time_sensitive(query, kb_chunks):
    _log.info("verify: 非时效敏感，跳过联网核对")
    return {**base, "web_verify_status": "skipped",
            "web_verify_note": "问题与材料均无时效敏感信号，未联网核对"}

  # 联网取材料。research 的兜底路径可能已经抓过了，别重复抓。
  if not web_chunks and settings.web_search_enabled:
    from app.tools.web_search import web_search
    try:
      web_chunks = web_search(
        query,
        max_results=int(settings.web_search_max_results),
        fetch_top=int(settings.web_search_fetch_top),
      )
    except Exception as e:        # web_search 自身不抛，这里只是双保险
      _log.warning("verify: web_search 异常: %s", e)
      web_chunks = []

  # 统一上下文列表：知识库片段 + 全部联网片段。
  # 这里必须合并，不能只返回 state 里原有的 chunks —— 否则「知识库为空、
  # 靠联网作答」这条路径会把刚抓到的网页全丢掉，带着空材料去作答。
  all_chunks = list(chunks) if web_from_state else list(chunks) + list(web_chunks)

  # 相关性闸门：联网结果跑偏时直接放弃核对，**不拿去裁决、也不进候选库**。
  # 详见 tools/search_quality.py 与 tools/browser_search.py 的注释 ——
  # 让裁决模型看无关材料会导致误报冲突，比不校验更糟。
  # 无关材料也不该占人工审核队列。
  if web_chunks and not _web_material_is_relevant(query, web_chunks):
    _log_relevance_reject(query, web_chunks)
    return {**base, "web_verify_status": "unverified",
            "web_verify_note": "联网结果与问题不相关，未用于核对（搜索引擎返回了无关内容）"}

  # 候选库：联网结果一律先落这里等人工审批，**绝不自动并入主知识库**。
  # 放在裁决之前 —— 就算裁决失败，候选也该被记录下来供人工判断。
  if web_chunks:
    try:
      from app.storage.candidates import add_candidates
      add_candidates(web_chunks, query=query, session_id=state.get("session_id"))
    except Exception as e:
      _log.warning("verify: 候选入库失败（不影响作答）: %s", e)

  if not web_chunks:
    _log.info("verify: 联网无结果，未能核对")
    return {**base, "web_verify_status": "unverified",
            "web_verify_note": "联网检索无结果，未能核对时效性"}

  if not kb_chunks:
    # 知识库没材料：直接用联网原文作答（这就是原来的兜底语义）。
    _log.info("verify: 知识库无材料，仅凭联网作答（%d 条）", len(web_chunks))
    return {**base, "web_verify_status": "web_only",
            "web_verify_note": "知识库无相关材料，以下内容仅来自联网检索",
            "retrieved_chunks": all_chunks}

  verdict = _adjudicate(kb_chunks, web_chunks, query, state)
  if verdict is None:
    _log.info("verify: 裁决不可用，标注未核对")
    return {**base, "web_verify_status": "unverified",
            "web_verify_note": "联网结果已获取，但自动核对未完成",
            "retrieved_chunks": all_chunks}

  status = str(verdict.get("status") or "").strip().lower()
  note = str(verdict.get("note") or "")[:300]

  if status == "insufficient":
    status = "unverified"

  if status == "conflict":
    conflicts = _normalize_conflicts(verdict.get("conflicts"),
                                     kb_chunks, web_chunks)
    _log.warning("verify: 检测到 %d 处冲突", len(conflicts))
    return {**base, "web_verify_status": "conflict",
            "web_verify_conflicts": conflicts,
            "web_verify_note": note,
            "retrieved_chunks": all_chunks}

  if status == "kb_stale":
    stale_idx = _refs_to_indices(verdict.get("stale_kb_refs"), len(kb_chunks))
    stale_notes = {str(kb_chunks[i].get("note_id") or "") for i in stale_idx}
    stale_notes.discard("")
    # 同一 note 的其它片段一并剔除：一份文档的时效性是一致的，
    # 只删被点名的那一段会留下同源的过期内容继续被引用。
    kept = [c for c in all_chunks
            if c.get("source_type") == "web"
            or str(c.get("note_id") or "") not in stale_notes]
    _log.warning("verify: 知识库过期，剔除 note=%s（%d 个片段）",
                 sorted(stale_notes), len(all_chunks) - len(kept))
    remaining_kb = [c for c in kept if c.get("source_type") != "web"]
    return {**base,
            "web_verify_status": "kb_stale" if remaining_kb else "web_only",
            "kb_stale_note_ids": sorted(stale_notes),
            "web_verify_note": note or "知识库内容已过期，未采用",
            "retrieved_chunks": kept}

  if status == "consistent":
    _log.info("verify: 知识库与联网一致")
    return {**base, "web_verify_status": "consistent",
            "web_verify_note": note,
            "retrieved_chunks": all_chunks}

  # 模型给了没见过的 status：当作未核对，别猜。
  _log.warning("verify: 未知 status=%r，按未核对处理", status)
  return {**base, "web_verify_status": "unverified",
          "web_verify_note": note,
          "retrieved_chunks": all_chunks}


def _normalize_conflicts(raw: Any, kb_chunks: list[dict],
                         web_chunks: list[dict]) -> list[dict]:
  """规整冲突条目，并把 refs 翻成可读的来源信息（供前端直接展示）。"""
  out: list[dict] = []
  if not isinstance(raw, list):
    return out
  for item in raw:
    if not isinstance(item, dict):
      continue
    claim = str(item.get("claim") or "").strip()
    if not claim:
      continue
    kb_idx = _refs_to_indices(item.get("kb_refs"), len(kb_chunks))
    web_letter = item.get("web_refs")
    web_idx: list[int] = []
    if isinstance(web_letter, list):
      for lv in web_letter:
        s = str(lv).strip().upper()
        if len(s) == 1 and "A" <= s <= "Z":
          i = ord(s) - ord("A")
          if 0 <= i < len(web_chunks) and i not in web_idx:
            web_idx.append(i)
    out.append({
      "claim": claim[:400],
      "kb_says": str(item.get("kb_says") or "")[:500],
      "web_says": str(item.get("web_says") or "")[:500],
      "kb_sources": [{
        "note_id": kb_chunks[i].get("note_id"),
        "title": kb_chunks[i].get("title"),
      } for i in kb_idx],
      "web_sources": [{
        "title": web_chunks[i].get("title"),
        "url": web_chunks[i].get("source_url"),
      } for i in web_idx],
    })
  return out
