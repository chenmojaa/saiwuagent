# -*- coding: utf-8 -*-
"""Planner node: decompose a research question into sub-queries (plan-and-execute).

设计（任务规划 / plan-and-execute）:
  - 在 router 之后、research 之前运行，把复杂问题分解为最多 planner_max_steps
    个子查询步骤，research 节点按计划逐步执行。
  - 使用 router 级廉价模型 + 原生 JSON 解析（沿用 router 的教训：不依赖
    structured output，避免 thinking 包裹导致的解析失败）。
  - 任何失败都返回空计划，research 自动回退到原有的启发式 follow-up 循环，
    规划永不阻塞主流程。
"""
from __future__ import annotations

import logging
import re

from app.agent.state import AgentState
from app.config import settings
from app.llm.factory import _build_model

_log = logging.getLogger(__name__)

PLANNER_PROMPT = """你是研究任务的规划器。阅读问题,只输出一个 JSON 对象,不要任何其他文字:

{"plan_summary": "<一句话描述整体思路>", "steps": [{"query": "<子问题检索语句>"}, ...]}

规划规则:
- 把问题拆成 1-4 个具体子问题,一起覆盖完整范围
- 每个 step 的 query 是知识库的独立检索句,语言与原问题一致
- 子问题必须互补(不同角度),不能是同义改写
- 第一个 step 用最直接的原问题形式
- 不要回答问题,只规划检索

【关键】步数要匹配问题复杂度,默认只出 1 步:
- 单点事实问题 -> 只出 1 个 step。问"X 是什么"、"X 有哪些"、"X 的内容"、
  "介绍一下 X"、"总结一下 X",一次检索就能覆盖,不要拆成"定义/原理/应用/
  优势"这种教科书式展开——那是过度拆分,会让同一批文档被重复检索多次。
- 只有问题明确要求跨对象比较、多来源汇总、或多个独立子主题时,才拆成 2-4 步。

示例:
输入: "RAG 是什么"
输出: {"plan_summary": "直接检索 RAG 的定义与说明", "steps": [{"query": "RAG 是什么"}]}

输入: "安徽文旅有哪些景点"
输出: {"plan_summary": "检索安徽文旅景点相关信息", "steps": [{"query": "安徽文旅有哪些景点"}]}

输入: "对比一下我笔记里 A 方案和 B 方案的优缺点"
输出: {"plan_summary": "分别检索两个方案再对比", "steps": [{"query": "A 方案的优点和缺点"}, {"query": "B 方案的优点和缺点"}, {"query": "A 方案与 B 方案的对比总结"}]}
"""

_JSON_RE = re.compile(r"\{[\s\S]*\}")
_THINK_RE = re.compile(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", re.IGNORECASE)

# 明确表示"复合问题"的线索词。命中才允许拆多步；否则默认单步。
# 注意:纯长度阈值不够——"对比 A 和 B 的优缺点"只有 12 字但确实要多步。
_MULTIPART_HINTS = (
    "对比", "比较", "对照", "区别", "差异", "vs", "versus",
    "各自", "分别", "优缺点", "优劣", "异同",
    "汇总", "综合", "总结", "梳理", "盘点", "归纳",
    "哪些方面", "几个方面", "多个", "列举", "分别说明",
)
# 超过这个长度的问题通常包含多个诉求，不再强制收敛为单步
_SIMPLE_MAX_LEN = 24
# 两个子查询的 token 重合度达到此值视为重复，折叠掉后者
_DUP_JACCARD = 0.75


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _tokens(s: str) -> set[str]:
    """粗粒度分词：CJK 逐字 + ASCII 小写词。只用于相似度比较。"""
    s = (s or "").lower()
    out: set[str] = set()
    buf: list[str] = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                out.add("".join(buf))
                buf = []
            out.add(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.add("".join(buf))
                buf = []
    if buf:
        out.add("".join(buf))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _is_simple(question: str) -> bool:
    """问题是否属于"一次检索就够"的单点问题。"""
    q = (question or "").strip()
    if not q:
        return True
    if len(q) > _SIMPLE_MAX_LEN:
        return False
    low = q.lower()
    return not any(h in low for h in _MULTIPART_HINTS)


def _collapse_redundant(queries: list[str], original: str) -> list[str]:
    """后置收敛：折叠近重复子查询；简单问题强制降为单步。

    动机：实测模型会无视 prompt 里的"简单问题只出 1 步"，把
    "RAG 是什么" 拆成 定义/原理/应用/优势 四步。四步都会打到同一批
    文档上，纯属浪费（1 次 planner + 4 次检索 + 4 次 embedding 调用）。
    这里做确定性兜底，不依赖模型自觉。
    """
    kept: list[str] = []
    for q in queries:
        tq = _tokens(q)
        if any(_jaccard(tq, _tokens(k)) >= _DUP_JACCARD for k in kept):
            _log.info("planner: 折叠近重复子查询 %r", q[:50])
            continue
        kept.append(q)
    if len(kept) > 1 and _is_simple(original):
        _log.info("planner: 判定为单点问题，%d 步收敛为 1 步", len(kept))
        kept = kept[:1]
    return kept


def _parse_plan(text: str) -> dict:
    """Parse planner JSON; raises ValueError on any structural problem."""
    text = _strip_think(text)
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("no JSON object in planner response")
    import json
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("planner JSON is not an object")
    steps_raw = obj.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ValueError("planner JSON has no steps")
    queries = []
    for s in steps_raw:
        q = (s.get("query") if isinstance(s, dict) else s) or ""
        q = str(q).strip()
        if q and q not in queries:
            queries.append(q)
        if len(queries) >= max(1, int(settings.planner_max_steps)):
            break
    if not queries:
        raise ValueError("planner steps contain no usable query")
    summary = str(obj.get("plan_summary") or "").strip()
    return {"plan_summary": summary, "queries": queries}


def planner_node(state: AgentState) -> dict:
    """Decompose the rewritten query into a search plan. NEVER raises.

    Returns {"plan": [...], "plan_summary": str}. Empty plan on any failure
    lets research fall back to its heuristic follow-up loop.
    """
    query = (state.get("rewritten_query") or state.get("query") or "").strip()
    base_result = {
        "plan": [],
        "plan_summary": "",
        "step_count": state.get("step_count", 0) + 1,
    }
    # 请求级覆盖（前端 Plan 开关）优先于服务端 HD_PLANNER_ENABLED
    override = state.get("use_planner")
    enabled = settings.planner_enabled if override is None else bool(override)
    if not enabled or not query:
        return base_result

    try:
        chat = _build_model(
            provider=None,
            model=settings.router_model or state.get("model_override"),
            api_key=state.get("api_key_override"),
            base_url=settings.router_base_url or state.get("base_url_override") or None,
            reasoning_level=None,
        )
    except Exception as e:
        _log.warning("planner: model init failed: %s", e)
        return base_result

    payload = PLANNER_PROMPT + "\nQuestion: " + query + "\nOutput JSON:"
    try:
        resp = chat.invoke(payload)
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        plan = _parse_plan(str(content or ""))
    except Exception as e:
        _log.warning("planner: JSON plan parsing failed, research will fall back: %s", e)
        return base_result

    # 后置收敛：模型常无视"简单问题只出 1 步"的指示，这里做确定性兜底。
    queries = _collapse_redundant(plan["queries"], query)
    if not queries:
        return base_result

    _log.info("planner: %d step(s): %s", len(queries), queries)
    return {
        "plan": [{"query": q} for q in queries],
        "plan_summary": plan["plan_summary"],
        "step_count": state.get("step_count", 0) + 1,
    }
