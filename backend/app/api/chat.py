"""Chat API: graph-driven orchestration with HD_USE_GRAPH fallback.

When HD_USE_GRAPH=true (default), LangGraph handles router + sub-agent dispatch,
and chat.py consumes the node events as SSE `stage` / domain events. After the
graph terminates, chat.py drives the LLM token stream via answer_node_stream.

When HD_USE_GRAPH=false, the legacy direct-call path is preserved (hybrid_search
+ answer_node_stream) for emergency rollback.
"""
from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import settings
from app.storage.hybrid import hybrid_search
from app.storage.db import create_session, append_message, get_messages, get_profile
from app.agent.graph import get_graph
from app.agent.nodes.answer import answer_node_stream
from app.agent.memory import summarize_overflow

router = APIRouter(tags=["chat"])

# Heuristic for "compare / contrast / multi-source" queries. When the user
# message obviously calls for a multi-step plan (compare X vs Y, tabulate
# differences, summarize across N sources), auto-flip use_planner so the
# planner + parallel_plan_node kicks in instead of the cheap single-shot
# retrieve path. The user can still force use_planner=false to disable.
_AUTO_PLAN_KEYWORDS = (
  "对比", "比较", "对照", "区别", "差异", "汇总", "综合", "总结",
  "vs",
  "compare", "compared", "tabulate", "summarise across", "summarize across",
  "三家", "3家", "三个云", "几家", "几款",
)
_AUTO_PLAN_MIN_LEN = 6  # "对比 A" alone is 4 chars; pad a bit so single-word queries don't trigger


def _auto_plan_triggered(query: str) -> bool:
  if not query:
    return False
  if len(query.strip()) < _AUTO_PLAN_MIN_LEN:
    return False
  q = query.lower()
  import re
  for kw in _AUTO_PLAN_KEYWORDS:
    kl = kw.lower()
    if kl in q:
      # Word-ish guard for the short ASCII keywords only. Chinese 2-char
      # keywords (对比/比较/总结/汇总/综合/区别/差异/对照) are unambiguous
      # inside CJK text, so substring match is fine. ASCII short tokens
      # like "vs" get a boundary check so "vscode" doesn't false-fire.
      if kl.isascii() and len(kl) <= 2:
        if re.search(r"(?:^|\s|[^A-Za-z0-9])" + re.escape(kl) + r"(?:[^A-Za-z0-9]|$)", q):
          return True
      else:
        return True
  return False

_log = logging.getLogger("chat")


class Message(BaseModel):
  role: str
  content: str


class ChatRequest(BaseModel):
  messages: list[Message]
  provider: str | None = None
  model: str | None = None
  use_rag: bool = True
  session_id: str | None = None
  base_url: str | None = None
  api_key: str | None = None
  reasoning_level: str | None = None
  embedding_model: str | None = None
  embedding_base_url: str | None = None
  agent_permission: str = "default"   # default=本地访问需询问 | full=完全访问
  use_planner: bool | None = None     # None=跟随服务端 HD_PLANNER_ENABLED；true/false=本轮强制开/关
  subagent: str | None = None         # Optional sub-agent profile (explore|plan|general) to dispatch in parallel with the main answer.


class PermissionDecision(BaseModel):
  request_id: str
  approve: bool


class ClarifyDecision(BaseModel):
  """用户对歧义澄清弹窗的回答：点选项或自由输入，空串=跳过。"""
  request_id: str
  answer: str = ""


def _extract_query(messages):
  for m in reversed(messages):
    if m.role == "user" and m.content.strip():
      return m.content
  return ""


def _sse(event: str, payload) -> str:
  return "event: " + event + chr(10) + "data: " + json.dumps(payload, ensure_ascii=False) + chr(10) + chr(10)


def _schedule_fact_extraction(session_id: str, user_query: str, assistant_reply: str,
                              api_key: str | None = None) -> None:
  """本轮对话结束后后台抽取用户事实（长期记忆）。不抛错、不阻塞 SSE。

  用 to_thread 跑同步 LLM 调用，避免阻塞事件循环；fire-and-forget。
  api_key 必须从当前请求透传：.env 不配 LLM_API_KEY 时凭证只存在于请求里。
  """
  import asyncio
  from app.agent.memory import extract_facts

  history = [
    {"role": "user", "content": user_query or ""},
    {"role": "assistant", "content": (assistant_reply or "")[:1500]},
  ]
  try:
    asyncio.get_running_loop().create_task(
      asyncio.to_thread(extract_facts, history, session_id, api_key)
    )
  except RuntimeError:
    pass  # 无事件循环（理论不可达）：跳过，下一轮再抽


def _build_initial_state(body: ChatRequest, query: str, session_id: str,
                         api_key, base_url, emb_key, emb_base, emb_model) -> dict:
  # Phase 2: server is the source of truth for conversation history.
  # We overwrite whatever the client sent with the messages we just persisted
  # to DB. Frontend may still send `messages` for legacy reasons (Phase 3
  # drops it), but we ignore it here.
  # chat.py persists the user turn to DB *before* this call, so the trailing
  # entry from get_messages is usually the current question. Append only when
  # the last message is not already the current query (defensive: covers the
  # rare case where the last inbound message was not a user role).
  history = get_messages(session_id, limit=16) if session_id else []
  if not history or history[-1] != {"role": "user", "content": query}:
    history.append({"role": "user", "content": query})
  # 剥掉历史里的思维链再交给下游。
  # 存储层保留 <think>（前端要折叠展示），但下游三处都不需要它：
  #   1) prompt（context.build_messages）
  #   2) 历史摘要（summarize_overflow）
  #   3) 长期记忆抽取（extract_facts）
  # 实测单条消息里思维链能占 46% 的字符，带着它算 token 预算会白白挤掉有效历史。
  from app.agent.context import strip_think as _strip_think
  history = [
    {**m, "content": _strip_think(m.get("content") or "")}
    for m in history
  ]
  history = [m for m in history if (m.get("content") or "").strip()]

  # Long-term memory (§6.5): load the cross-session profile for this session's
  # user. Single-user local app -> one shared profile keyed by "default".
  profile = {}
  try:
    profile = get_profile("default")
  except Exception:
    pass

  # 长期记忆召回（§6.6）：按与当前问题的相关性召回已抽取的跨会话事实，
  # 注入 state 供 answer 阶段拼进 system prompt。Best-effort。
  memory_facts = []
  try:
    from app.storage.db import recall_facts
    memory_facts = recall_facts(query or "", limit=int(settings.memory_max_facts))
  except Exception as e:
    _log.debug("memory recall failed (ignored): %s", e)

  # AGENTS.md / project-rules (Codex CLI / Claude Code convention):
  # standing instructions are loaded into the answer system prompt.
  project_rules = ""
  try:
    from app.agent.agents_md import project_rules as _pr
    rules = _pr()
    # project_rules() 返回 RuleSet 对象（.text 为合并文本）；兼容旧版返回 str
    project_rules = rules.text if hasattr(rules, "text") else (rules or "")
  except Exception as e:
    _log.debug("agents_md load failed (ignored): %s", e)

  # History summary (§6.3): only when the window overflows, compress the cut-off
  # older turns so follow-ups still see the gist. Best-effort, never blocks.
  summary = ""
  if len(history) > 12:
    try:
      summary = summarize_overflow(history, api_key=api_key)
    except Exception:
      summary = ""

  return {
    "messages": history,
    "session_id": session_id,
    "query": query,
    "retrieved_chunks": [],
    "provider_override": body.provider,
    "model_override": body.model,
    "base_url_override": base_url,
    "api_key_override": api_key,
    "reasoning_level_override": body.reasoning_level,
    "embedding_model_override": emb_model,
    "step_count": 0,
    "profile": profile,
    "summary": summary,
    "memory_facts": memory_facts,
    "project_rules": project_rules,
    # Per-turn fields must be reset: with a persistent SQLite checkpointer any
    # field absent from this input would leak from the previous turn's checkpoint
    # (e.g. a stale intent="ingest" re-routing a plain chat turn to the ingest node).
    "intent": "",
    "rewritten_query": "",
    "plan": [],
    "plan_summary": "",
    "plan_cursor": 0,
    "plan_status": [],
    "replan_stalled": False,
    "skip_retrieval": False,
    "research_iterations": 0,
    "research_notes": [],
    "ingest_result": {},
    "report_result": {},
    "answer": None,
    "citations": [],
    "agent_permission": (body.agent_permission or "default").lower(),
    "use_planner": body.use_planner,
    # 歧义澄清：每轮显式重置（checkpointer 泄露防护）；
    # skip_clarify 只在拿到用户回答后的二次执行中由 chat.py 置 True。
    "clarify_request": None,
    "skip_clarify": False,
    "subagent_mode": body.subagent,
  }


@router.post("/chat/permission")
async def resolve_permission(body: PermissionDecision):
  """Frontend answers a pending agent permission request (allow/deny)."""
  from app.agent.tools import permissions as _perm
  ok = _perm.resolve(body.request_id, body.approve)
  return {"ok": ok}


@router.post("/chat/clarify")
async def resolve_clarify(body: ClarifyDecision):
  """用户回答歧义澄清（点选项 / 自由输入 / 跳过），唤醒挂起的 /chat SSE 流。"""
  from app.agent import clarify as _clarify
  ok = _clarify.resolve(body.request_id, body.answer)
  return {"ok": ok}


@router.post("/chat")
async def chat(body: ChatRequest, x_api_key: str | None = Header(None, alias="X-API-Key")):
  if not body.messages:
    raise HTTPException(status_code=400, detail="messages is empty")

  query = _extract_query(body.messages)
  # Memory shortcut: "记住 X" / "forget X" / "/记住 X" handled inline so we
  # do not consume an LLM call just to acknowledge.
  if query:
    cmd, payload = _detect_memory_command(query)
    if cmd == "remember":
      from app.storage.db import list_facts as _lf, save_facts as _sf
      content = payload
      norm = content.lower().strip()
      if len(content) > 500:
        ack = "内容太长了（最多 500 字），无法记住。"
      elif any((f.get("content", "").lower().strip() == norm or norm in f.get("content", "").lower() or f.get("content", "").lower() in norm) for f in _lf(limit=500)):
        ack = "这条已经在记忆里了，不用重复记住。"
      elif _sf([content], session_id=body.session_id or "memory_cmd") > 0:
        ack = "记下了：" + content
      else:
        ack = "记住失败，请稍后再试。"
      async def _ack_stream():
        yield "event: session\ndata: " + json.dumps({"session_id": body.session_id or "memory_cmd"}) + "\n\n"
        yield "event: stage\ndata: " + json.dumps({"stage": "memory_shortcut", "action": "remember"}) + "\n\n"
        yield "event: answer\ndata: " + json.dumps({"text": ack}) + "\n\n"
        yield "event: done\ndata: {}\n\n"
      return StreamingResponse(_ack_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    if cmd == "forget":
      fid, fcontent = _pick_fact_to_forget(payload)
      if fid is None:
        ack = "没找到匹配的记忆。可以打开「记忆」页面看看有哪些条目。"
      else:
        from app.storage.db import delete_fact as _df
        _df(fid)
        ack = "已忘记：" + fcontent
      async def _ack_stream2():
        yield "event: session\ndata: " + json.dumps({"session_id": body.session_id or "memory_cmd"}) + "\n\n"
        yield "event: stage\ndata: " + json.dumps({"stage": "memory_shortcut", "action": "forget"}) + "\n\n"
        yield "event: answer\ndata: " + json.dumps({"text": ack}) + "\n\n"
        yield "event: done\ndata: {}\n\n"
      return StreamingResponse(_ack_stream2(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

  # /api/custom-models 出于安全只返回脱敏 apiKey（sk****xx），前端会把列表值
  # 原样回传。脱敏 key 非空会覆盖服务端全部凭证回退链，导致上游全线 401
  # （MiniMax 1004 login fail）。真实 key 不含 '*'，脱敏回显一律视为未携带。
  _req_key = (body.api_key or x_api_key or "").strip()
  if "*" in _req_key:
    _req_key = ""
  api_key = _req_key or None
  if not api_key:
    # 前端刷新后只持有脱敏 key（列表接口不回明文），这里从服务端落盘的
    # models.json 里取「当前选中条目」的原始 key —— 它才是用户在界面上
    # 选定的模型配置的真正凭证。
    try:
      from app.storage.models_store import list_models, get_selected_id
      _sid = get_selected_id()
      for _m in list_models():
        if _m.get("id") == _sid:
          _k = (_m.get("apiKey") or "").strip()
          if _k and "*" not in _k:
            api_key = _k
          break
    except Exception:
      pass
  base_url = (body.base_url or "").strip() or None
  emb_key = api_key
  emb_base = (body.embedding_base_url or body.base_url or "").strip() or None
  emb_model = (body.embedding_model or "").strip() or None

  session_id = body.session_id
  if not session_id:
    title = (query[:50] + ("..." if len(query) > 50 else "")) if query else "New chat"
    sess = create_session(title=title)
    session_id = sess.id

  if query and body.messages[-1].role == "user":
    append_message(session_id, "user", query)

  # MCP context: tag every subsequent log row with this session_id, then
  # reset the session pool so stale sessions from previous turns do not
  # leak across users.
  try:
    from app.agent.tools.mcp_tools import set_session_id, set_permission_mode, reset_mcp_sessions
    set_session_id(session_id)
    set_permission_mode(getattr(body, "agent_permission", "default") or "default")
    reset_mcp_sessions()
  except Exception:
    pass

  initial_state = _build_initial_state(body, query, session_id, api_key, base_url,
                                       emb_key, emb_base, emb_model)
  t_req = time.perf_counter()

  async def generate():
    yield _sse("session", {"session_id": session_id})

    final_state = dict(initial_state)
    intent = "chat"
    rag_hits = 0
    early_done = False

    async def _graph_pass(state_in: dict):
      """跑一轮图（router -> 各节点），逐事件产出 SSE；结果写回闭包变量。

      歧义澄清场景会跑两轮：第一轮 router 判定歧义即终止（clarify -> END），
      拿到用户回答后 chat.py 用精炼问题重跑第二轮（skip_clarify 防重复追问）。
      """
      nonlocal final_state, intent, rag_hits, early_done
      final_state = dict(state_in)
      t_router = time.perf_counter()
      # Phase 3.3: thread_id config makes the checkpointer in graph.py save
      # final state per session, so a later call with the same session_id can
      # resume from where this one left off. chat.py still seeds state from
      # DB (Phase 2); the checkpointer supplements it for LangGraph-level
      # state continuity.
      graph_config = {"configurable": {"thread_id": session_id}} if session_id else None
      try:
        graph = await get_graph()
        async for event in graph.astream(state_in, config=graph_config):
          for node_name, delta in (event or {}).items():
            if not isinstance(delta, dict):
              continue
            # Phase 3.1: state.messages uses Annotated[list, operator.add].
            # When a node returns {"messages": [...]} LangGraph accumulates
            # it under the hood, but the per-node delta in the stream event
            # is the new fragment only -- merging with update() would clobber
            # history. Extend in place for messages, update the rest.
            if "messages" in delta:
              msgs = delta.pop("messages") or []
              final_state.setdefault("messages", []).extend(msgs)
            final_state.update(delta)
            if node_name == "router":
              intent = delta.get("intent") or intent
              router_ms = (time.perf_counter() - t_router) * 1000
              yield _sse("stage", {"stage": "router", "status": "done",
                                   "intent": intent,
                                   "rewritten_query": delta.get("rewritten_query"),
                                   "ms": round(router_ms, 1)})
            elif node_name == "planner":
              plan = delta.get("plan") or []
              queries = [str(s.get("query") or "") for s in plan if isinstance(s, dict)]
              yield _sse("stage", {"stage": "agent", "status": "done",
                                   "agent": "planner",
                                   "steps": len(plan),
                                   "plan_summary": delta.get("plan_summary") or ""})
              # 计划创建：前端渲染计划进度条（summary + 每步一个 chip）
              if queries:
                yield _sse("plan", {"phase": "created",
                                    "summary": delta.get("plan_summary") or "",
                                    "queries": queries})
            elif node_name == "execute_plan":
              # 计划逐步执行：astream 每循环一次产生一个事件 -> 逐步推送进度
              rag_hits = len(delta.get("retrieved_chunks") or [])
              plan_status = delta.get("plan_status") or []
              cursor = int(delta.get("plan_cursor") or 0)
              total_steps = len(final_state.get("plan") or [])
              if plan_status:
                last = plan_status[-1]
                yield _sse("stage", {"stage": "agent", "status": "done",
                                     "agent": "research",
                                     "step": cursor, "total_steps": total_steps,
                                     "query": last.get("query") or "",
                                     "hits": rag_hits})
                yield _sse("plan", {"phase": "step_done", "index": cursor - 1,
                                    "query": last.get("query") or "",
                                    "hits": int(last.get("hits") or 0)})
            elif node_name == "replan":
              # 动态补缺一轮
              rag_hits = len(delta.get("retrieved_chunks") or [])
              notes = delta.get("research_notes") or []
              if notes and not delta.get("replan_stalled"):
                yield _sse("stage", {"stage": "agent", "status": "done",
                                     "agent": "research",
                                     "step": None, "query": notes[-1],
                                     "hits": rag_hits})
                yield _sse("plan", {"phase": "replan", "query": notes[-1],
                                    "hits": rag_hits})
            elif node_name == "retrieve":
              rag_hits = len(delta.get("retrieved_chunks") or [])
              yield _sse("stage", {"stage": "rag_search", "status": "started"})
              yield _sse("stage", {"stage": "rag_search", "status": "done", "hits": rag_hits})
            elif node_name == "ingest":
              yield _sse("stage", {"stage": "agent", "status": "done", "agent": "ingest"})
              yield _sse("ingest", delta.get("ingest_result") or {})
              early_done = True
            elif node_name == "report":
              yield _sse("stage", {"stage": "agent", "status": "done", "agent": "report"})
              yield _sse("report", delta.get("report_result") or {})
              early_done = True
      except Exception as e:
        _log.warning("graph.astream failed: %s", e)
        yield _sse("error", {"detail": "graph: %s: %s" % (type(e).__name__, e)})
        # fall through to a minimal legacy answer
        early_done = False
        intent = "chat"

    if settings.use_graph:
      yield _sse("stage", {"stage": "router", "status": "started"})
      async for sse in _graph_pass(initial_state):
        yield sse

      # ---- 歧义澄清（模型驱动，始终开启，无需用户开关）----
      # router 判定问题有歧义 -> 图先终止 -> SSE clarify 事件弹窗等用户回答
      # （点选项 / 自由输入 / 超时跳过）-> 回答并入原问题后带 skip_clarify 重跑图。
      if (not early_done and final_state.get("clarify_request")
              and not final_state.get("skip_clarify")):
        from app.agent import clarify as _clarify
        spec = final_state.get("clarify_request") or {}
        rid, _fut = _clarify.create_request()
        yield _sse("clarify", {"request_id": rid,
                               "question": spec.get("question") or "",
                               "options": spec.get("options") or []})
        answer = (await _clarify.wait_answer(rid) or "").strip()
        base_q = (query or final_state.get("query") or "").strip()
        refined = ("%s（用户澄清：%s）" % (base_q, answer)) if answer else base_q
        # 二次执行：重置一次性字段（checkpointer / 首轮残留防护）
        second_state = dict(final_state)
        second_state.update({
            "query": refined,
            "rewritten_query": refined,
            "intent": "",
            "clarify_request": None,
            "skip_clarify": True,
            "skip_retrieval": False,
            "plan": [], "plan_summary": "", "plan_cursor": 0, "plan_status": [],
            "replan_stalled": False,
            "research_iterations": 0, "research_notes": [],
            "retrieved_chunks": [],
            "answer": None, "citations": [],
            "ingest_result": {}, "report_result": {},
        })
        yield _sse("stage", {"stage": "router", "status": "started"})
        async for sse in _graph_pass(second_state):
          yield sse

      if early_done:
        yield "data: [DONE]" + chr(10) + chr(10)
        return

      # 研究循环结束：发一次汇总（前端把进行中的 chips 收尾）
      if intent == "research":
        yield _sse("plan", {"phase": "done",
                            "iterations": int(final_state.get("research_iterations") or 0),
                            "hits": len(final_state.get("retrieved_chunks") or [])})
        yield _sse("stage", {"stage": "agent", "status": "done",
                             "agent": "research",
                             "iterations": int(final_state.get("research_iterations") or 0),
                             "hits": len(final_state.get("retrieved_chunks") or [])})

      # Phase 2: LLM token stream from final graph state (chat / research).
      yield _sse("stage", {"stage": "llm_stream", "status": "started"})
      text_parts: list[str] = []
      citations: list = []
      errored = False
      first_delta_logged = False
      try:
        async for kind, payload in answer_node_stream(final_state):
          if kind == "text_delta":
            text_parts.append(payload)
            if not first_delta_logged:
              first_delta_logged = True
              t_first = (time.perf_counter() - t_req) * 1000
              _log.info("chat ttft: graph=%s rag_hits=%d session=%s model=%s",
                        intent, rag_hits, session_id, body.model or "")
              _log.info("chat ttft_ms=%.0f", t_first)
            for line in payload.split(chr(10)):
              yield "data: " + line + chr(10)
            yield chr(10)
          elif kind == "tool_calls_batch":
            yield _sse("tool", {"phase": "start", "calls": payload.get("names") or []})
          elif kind == "tool_call":
            yield _sse("tool", {"phase": "call", "name": payload.get("name"),
                                "args": payload.get("args")})
          elif kind == "tool_result":
            yield _sse("tool", {"phase": "result", "name": payload.get("name"),
                                "ok": payload.get("ok"),
                                "snippet": payload.get("snippet")})
          elif kind == "permission_request":
            yield _sse("permission", {"phase": "request", **payload})
          elif kind == "permission_result":
            yield _sse("permission", {"phase": "result", **payload})
          elif kind == "done":
            citations = payload.get("citations") or []
          elif kind == "error":
            errored = True
            yield _sse("error", {"detail": payload})
      except Exception as e:
        errored = True
        yield _sse("error", {"detail": str(e)})

      yield _sse("stage", {"stage": "llm_stream", "status": "done"})
      if citations:
        yield _sse("citations", citations)
      # ---- Optional sub-agent dispatch (Codex CLI / Claude Code parity) ----
      # If the request set subagent=<mode>, fire a parallel read-only or
      # general-purpose sub-agent run *after* the main answer stream completes
      # and surface its reasoning + tool calls as ``subagent`` SSE events.
      # Disabled unless the user explicitly opts in; defaults to None so
      # the existing behaviour is preserved.
      sub_mode = final_state.get("subagent_mode") or body.subagent
      if sub_mode:
        try:
          from app.agent.subagents import run_subagent_stream, SUBAGENT_MODES
          mode = sub_mode if sub_mode in SUBAGENT_MODES else "general"
          sub_history = list(final_state.get("messages") or [])[-8:]
          async for ev in run_subagent_stream(
            mode=mode, query=query,
            api_key=api_key, base_url=base_url,
            history=sub_history,
            extra_context={
              "summary": final_state.get("summary") or "",
              "memory_facts": final_state.get("memory_facts") or [],
              "project_rules": final_state.get("project_rules") or "",
              "agent_permission": (body.agent_permission or "default"),
            },
          ):
            yield _sse("subagent", ev)
        except Exception as e:
          yield _sse("subagent", {"phase": "error", "detail": str(e)[:300]})
      yield "data: [DONE]" + chr(10) + chr(10)

      if not errored and text_parts:
        try:
          append_message(session_id, "assistant", "".join(text_parts),
                         citations if citations else None)
        except Exception:
          pass
        # 长期记忆：本轮结束后后台抽取用户事实（不阻塞 SSE 响应）
        _schedule_fact_extraction(session_id, query, "".join(text_parts), api_key)
      return

    # ---- Legacy path (HD_USE_GRAPH=false) ----
    yield _sse("stage", {"stage": "rag_search", "status": "started"})
    t_rag = time.perf_counter()
    retrieved = []
    if body.use_rag and query:
      try:
        retrieved = hybrid_search(query, top_k=5, api_key=emb_key,
                                  base_url=emb_base, model=emb_model)
      except Exception as e:
        _log.warning("hybrid_search failed: %s", e)
    initial_state["retrieved_chunks"] = retrieved
    rag_ms = (time.perf_counter() - t_rag) * 1000
    yield _sse("stage", {"stage": "rag_search", "status": "done",
                        "ms": round(rag_ms, 1), "hits": len(retrieved)})

    yield _sse("stage", {"stage": "llm_stream", "status": "started"})
    text_parts = []
    citations = []
    errored = False
    first_delta_logged = False
    try:
      async for kind, payload in answer_node_stream(initial_state):
        if kind == "text_delta":
          text_parts.append(payload)
          if not first_delta_logged:
            first_delta_logged = True
            t_first = (time.perf_counter() - t_req) * 1000
            _log.info("chat ttft (legacy): rag=%.0fms llm_first_delta=%.0fms hits=%d session=%s",
                      rag_ms, t_first, len(retrieved), session_id)
          for line in payload.split(chr(10)):
            yield "data: " + line + chr(10)
          yield chr(10)
        elif kind == "tool_calls_batch":
          yield _sse("tool", {"phase": "start", "calls": payload.get("names") or []})
        elif kind == "tool_call":
          yield _sse("tool", {"phase": "call", "name": payload.get("name"),
                              "args": payload.get("args")})
        elif kind == "tool_result":
          yield _sse("tool", {"phase": "result", "name": payload.get("name"),
                              "ok": payload.get("ok"),
                              "snippet": payload.get("snippet")})
        elif kind == "permission_request":
          yield _sse("permission", {"phase": "request", **payload})
        elif kind == "permission_result":
          yield _sse("permission", {"phase": "result", **payload})
        elif kind == "done":
          citations = payload.get("citations") or []
        elif kind == "error":
          errored = True
          yield _sse("error", {"detail": payload})
    except Exception as e:
      errored = True
      yield _sse("error", {"detail": str(e)})

    yield _sse("stage", {"stage": "llm_stream", "status": "done"})
    if citations:
      yield _sse("citations", citations)
    yield "data: [DONE]" + chr(10) + chr(10)

    if not errored and text_parts:
      try:
        append_message(session_id, "assistant", "".join(text_parts),
                       citations if citations else None)
      except Exception:
        pass

  return StreamingResponse(generate(), media_type="text/event-stream",
                          headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
# Helper called before the LangGraph pipeline. If the user message is a
# "remember this" or "forget that" command, handle it inline (so it does
# not get answered as a normal chat question) and yield a synthetic SSE
# reply.
import re as _re

_MEM_REMEMBER_PATTERNS = (
  _re.compile(r"^\s*(?:(?:请|麻烦|please)\s*)?(?:记(?:住|录|下)|remember(?:\s+that)?)\s*[::]?\s*(.+)$", _re.IGNORECASE),
  _re.compile(r"^\s*/remember\s+(.+)$", _re.IGNORECASE),
  _re.compile(r"^\s*/记住\s*(.*)$"),
)
_MEM_FORGET_PATTERNS = (
  _re.compile(r"^\s*(?:(?:请|麻烦|please)\s*)?(?:忘(?:记|了)|forget(?:\s+about)?)\s*[::]?\s*(.+)$", _re.IGNORECASE),
  _re.compile(r"^\s*/forget\s+(.+)$", _re.IGNORECASE),
  _re.compile(r"^\s*/忘记\s*(.*)$"),
)


def _detect_memory_command(text: str):
  """Return ("remember" | "forget", content_str) or (None, None)."""
  if not text:
    return None, None
  t = text.strip()
  for pat in _MEM_REMEMBER_PATTERNS:
    m = pat.match(t)
    if m:
      content = (m.group(1) or "").strip().rstrip("�?!?!?")
      if content:
        return "remember", content
      return "remember", t
  for pat in _MEM_FORGET_PATTERNS:
    m = pat.match(t)
    if m:
      content = (m.group(1) or "").strip().rstrip("�?!?!?")
      return "forget", content
  return None, None


def _pick_fact_to_forget(content: str):
  """Find the best-matching fact for a forget command. Returns (id, content) or (None, None)."""
  from app.storage.db import list_facts
  facts = list_facts(limit=200)
  if not facts:
    return None, None
  content_l = (content or "").lower().strip()
  if not content_l:
    return None, None
  # 1) Exact substring match
  for f in facts:
    fc = (f.get("content") or "").lower()
    if content_l == fc or content_l in fc or fc in content_l:
      return f.get("id"), f.get("content")
  # 2) Token overlap (>50% of tokens of the query appear in the fact)
  tokens = [t for t in content_l.split() if len(t) >= 2]
  if tokens:
    best = None
    best_score = 0
    for f in facts:
      fc = (f.get("content") or "").lower()
      hits = sum(1 for t in tokens if t in fc)
      if hits > best_score:
        best_score = hits
        best = f
    if best and best_score >= max(1, len(tokens) // 2):
      return best.get("id"), best.get("content")
  return None, None
