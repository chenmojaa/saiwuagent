# -*- coding: utf-8 -*-
"""Research nodes: plan execution + dynamic replanning (plan-and-execute).

图结构（graph.py）：
  planner → execute_plan (循环，每步一个子查询) → replan (循环，动态补缺) → END

拆成两个 LangGraph 节点的原因：循环节点每执行一步就向 astream 产出一次
事件，chat.py 因此能逐步流式推送「计划 2/4：xxx」进度，而不是等整个
研究完成后才发一次汇总（旧版黑盒问题）。

  - ``execute_plan_node``: 执行 plan[plan_cursor] 的一次混合检索，给每个
    新 chunk 标注 ``matched_query``（命中的子查询），推进 cursor。
  - ``replan_node``: 计划执行完仍不足 target 时的动态补缺——基于已有
    材料让廉价模型生成下一个检索角度（原启发式 follow-up 的泛化）。
    生成不出新查询时置 ``replan_stalled`` 终止循环。

预算设计（修复旧版冲突）：
  - 计划步数上限 = planner_max_steps（默认 4），不再被 research_max_iter
    截断——旧版 plan_queries[:max_iter] 会把 4 步计划砍成 3 步。
  - replan 轮数上限 = research_max_iter（默认 3）。
"""
from __future__ import annotations

import logging
from typing import Any

from app.agent.state import AgentState
from app.agent.context import strip_think
from app.config import settings
from app.storage.hybrid import hybrid_search
from app.llm.factory import _build_model

_log = logging.getLogger(__name__)

FOLLOWUP_PROMPT = """You are a research assistant. You have already gathered these reference snippets from the knowledge base:

<<COLLECTED>>

Original question: <<ORIG>>

Suggest ONE follow-up search query (in the SAME language as the original question, in a complete sentence) that would help find the missing angle. Output ONLY the query, no preamble."""


def _snippet(c: dict[str, Any]) -> str:
    return (c.get("text") or c.get("snippet") or "")[:120]


def _seen_key(c: dict[str, Any]) -> tuple[str, int]:
    return (str(c.get("note_id") or ""), int(c.get("chunk_index") or -1))


def _search(q: str, api_key, base_url) -> list[dict[str, Any]]:
    """hybrid_search wrapper: never raises, logs and returns [] on failure."""
    try:
        return hybrid_search(q, top_k=5, api_key=api_key, base_url=base_url, model=None)
    except Exception as e:
        _log.warning("research: hybrid_search failed for q=%r: %s", q[:60], e)
        return []


def _generate_followup(collected: list[dict[str, Any]], original: str,
                       chat) -> str | None:
    """Ask the cheap router-tier model for the next angle to search."""
    if not collected:
        return None
    titles = [c.get("title") or c.get("note_id") for c in collected[:8]]
    snippets = [_snippet(c) for c in collected[:8]]
    collected_str = "\n".join("- [%s] %s" % (t, s) for t, s in zip(titles, snippets))
    payload = FOLLOWUP_PROMPT.replace("<<COLLECTED>>", collected_str).replace("<<ORIG>>", original)
    try:
        resp = chat.invoke(payload)
        text = getattr(resp, "content", None) or str(resp)
        if isinstance(text, list):   # LangChain 的 content 可能是分片列表
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text
            )
        # 必须剥掉思维链：推理模型（MiniMax 等）的响应以 <think> 开头，
        # 直接取第一行会拿到字面量 "<think>" 当作检索词 —— 拿它去搜只会召回
        # 一堆无关切片，最终被当成"来源"展示给用户（实测真实发生过）。
        cleaned = strip_think(str(text or ""))
        if not cleaned:
            return None
        q = cleaned.splitlines()[0][:200].strip()
        # 兜底：检索词至少要有两个字符且含实义字符，否则视为没生成出角度
        if len(q) < 2:
            _log.info("research: follow-up query too short, treating as stalled: %r", q)
            return None
        return q
    except Exception as e:
        _log.warning("research: follow-up generation failed: %s", e)
        return None


def _followup_model(state: AgentState):
    """Build the cheap follow-up model, or None if init fails."""
    try:
        return _build_model(
            provider=None,
            model=settings.router_model or state.get("model_override"),
            api_key=state.get("api_key_override"),
            base_url=settings.router_base_url or state.get("base_url_override") or None,
            reasoning_level=None,
        )
    except Exception as e:
        _log.warning("research: follow-up model init failed: %s", e)
        return None


def execute_plan_node(state: AgentState) -> dict:
    """Execute ONE plan step: hybrid search for plan[plan_cursor].

    Each new chunk is tagged with ``matched_query`` so the answer prompt can
    tell which sub-question each piece of reference material answers.

    Fast path: when parallel_plan_enabled is on and we are at the start of
    a multi-step plan, hand off to parallel_plan_node which gathers all
    sub-queries with a ThreadPoolExecutor. The serial path stays as the
    safe fallback.
    """
    if (
      settings.parallel_plan_enabled
      and int(state.get("plan_cursor") or 0) == 0
      and len(state.get("plan") or []) > 1
    ):
        return parallel_plan_node(state)

    plan = state.get("plan") or []
    cursor = int(state.get("plan_cursor") or 0)
    if cursor >= len(plan):
        # Defensive: routing should never send us here with a spent plan.
        return {"step_count": state.get("step_count", 0) + 1}

    q = str(plan[cursor].get("query") or "").strip()
    if not q:
        # Skip empty step but still advance the cursor so the loop terminates.
        return {"plan_cursor": cursor + 1,
                "step_count": state.get("step_count", 0) + 1}

    collected = list(state.get("retrieved_chunks") or [])
    seen = {_seen_key(c) for c in collected}
    hits = _search(q, state.get("api_key_override"), state.get("base_url_override"))
    new_chunks = []
    for c in hits:
        if _seen_key(c) in seen:
            continue
        seen.add(_seen_key(c))
        c["matched_query"] = q      # 标注：该 chunk 由哪个子查询命中
        new_chunks.append(c)
    collected.extend(new_chunks)

    plan_status = list(state.get("plan_status") or [])
    plan_status.append({"query": q, "hits": len(new_chunks)})

    _log.info("research: plan step %d/%d q=%r hits=%d total=%d",
              cursor + 1, len(plan), q[:60], len(new_chunks), len(collected))
    return {
        "retrieved_chunks": collected,
        "plan_cursor": cursor + 1,
        "plan_status": plan_status,
        "research_iterations": int(state.get("research_iterations") or 0) + 1,
        "step_count": state.get("step_count", 0) + 1,
    }


def replan_node(state: AgentState) -> dict:
    """Dynamic replanning: one follow-up retrieval round when material is short.

    Also serves as the no-plan fallback path (planner disabled / failed):
    the first round searches the original query directly, subsequent rounds
    ask the cheap model for the missing angle.
    """
    query = (state.get("rewritten_query") or state.get("query") or "").strip()
    if not query:
        return {"replan_stalled": True,
                "step_count": state.get("step_count", 0) + 1}

    collected = list(state.get("retrieved_chunks") or [])
    notes = list(state.get("research_notes") or [])
    executed = {str(s.get("query") or "") for s in (state.get("plan_status") or [])}
    executed |= set(notes)

    # Pick the next query: original query when we have nothing yet (no-plan
    # fallback path), otherwise a model-generated follow-up angle.
    q = query if not collected else None
    if q is None:
        chat = _followup_model(state)
        if chat is None:
            return {"replan_stalled": True,
                    "step_count": state.get("step_count", 0) + 1}
        q = _generate_followup(collected, query, chat)
    if not q or q in executed:
        # Nothing new to search -> stop the loop (routing checks this flag).
        _log.info("research: replan stalled (no new query)")
        return {"replan_stalled": True,
                "step_count": state.get("step_count", 0) + 1}

    seen = {_seen_key(c) for c in collected}
    hits = _search(q, state.get("api_key_override"), state.get("base_url_override"))
    new_chunks = []
    for c in hits:
        if _seen_key(c) in seen:
            continue
        seen.add(_seen_key(c))
        c["matched_query"] = q
        new_chunks.append(c)
    collected.extend(new_chunks)
    notes.append(q)

    _log.info("research: replan q=%r hits=%d total=%d", q[:60], len(new_chunks), len(collected))
    return {
        "retrieved_chunks": collected,
        "research_notes": notes,
        "research_iterations": int(state.get("research_iterations") or 0) + 1,
        "replan_stalled": False,
        "step_count": state.get("step_count", 0) + 1,
    }



def parallel_plan_node(state: AgentState) -> dict:
  """Run all plan steps concurrently with a ThreadPoolExecutor.

  Note: this node is a *synchronous* LangGraph node, so it cannot use
  ``asyncio.gather`` — the concurrency comes from a thread pool running the
  blocking ``hybrid_search`` calls in parallel.

  Falls back to the serial execute_plan_node when:
    - parallel_plan_enabled is False
    - the plan has 0 or 1 step (no parallelism to exploit)
    - the cursor is already mid-plan (legacy loop support)

  Each step performs an independent hybrid_search, then we merge the
  results, dedupe by (note_id, chunk_index), and tag every chunk with
  matched_query = the step's query string so the answer prompt can show
  "step N found X" attributions.

  Failure isolation: if one step's hybrid_search raises, we log it and
  return whatever the other steps produced, plus a step_count +=1.
  """
  from concurrent.futures import ThreadPoolExecutor
  from app.agent.nodes.research import _search

  plan = state.get("plan") or []
  cursor = int(state.get("plan_cursor") or 0)
  if (
    not settings.parallel_plan_enabled
    or len(plan) <= 1
    or cursor != 0
  ):
    # Defer to the serial executor.
    return execute_plan_node(state)

  api_key = state.get("api_key_override")
  base_url = state.get("base_url_override")

  queries = [str(s.get("query") or "").strip() for s in plan]
  workers = min(len(queries), max(1, int(settings.parallel_plan_max_workers)))

  def _run_one(args):
    idx, q = args
    if not q:
      return idx, []
    try:
      hits = _search(q, api_key, base_url)
    except Exception as e:
      _log.warning("parallel_plan: step %d failed (q=%r): %s", idx + 1, q[:60], e)
      return idx, []
    return idx, hits

  with ThreadPoolExecutor(max_workers=workers) as pool:
    results = list(pool.map(_run_one, list(enumerate(queries))))

  seen: set = {_seen_key(c) for c in (state.get("retrieved_chunks") or [])}
  collected: list = list(state.get("retrieved_chunks") or [])
  plan_status: list = list(state.get("plan_status") or [])
  for idx, hits in results:
    q = queries[idx]
    new_chunks: list = []
    for c in hits:
      if _seen_key(c) in seen:
        continue
      seen.add(_seen_key(c))
      c["matched_query"] = q
      new_chunks.append(c)
    collected.extend(new_chunks)
    plan_status.append({"query": q, "hits": len(new_chunks)})
    _log.info("parallel_plan: step %d/%d q=%r hits=%d",
              idx + 1, len(plan), q[:60], len(new_chunks))

  return {
    "retrieved_chunks": collected,
    "plan_cursor": len(plan),
    "plan_status": plan_status,
    "research_iterations": int(state.get("research_iterations") or 0) + 1,
    "step_count": int(state.get("step_count") or 0) + 1,
  }
