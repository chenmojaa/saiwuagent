"""Answer node - LangChain chat model with optional tool-calling.

Two entry points are kept on purpose:

  - ``answer_node`` (non-streaming, structured output via AnswerResult). This is
    only used by legacy code paths that want a JSON answer; tool-calling is
    not combined with ``with_structured_output`` because most providers will not
    return tool_calls when constrained to a JSON schema.

  - ``answer_node_stream`` is what chat.py drives. When ``settings.tools_enabled``
    is on AND there is at least one registered tool (skills or MCP), we bind
    the tools to the chat model and run a tool-call loop:

      1. stream the model with the current message list (token 级 text_delta)
      2. if the response carries ``tool_calls`` -> execute them, append a
         ``ToolMessage`` for each, continue
      3. the first response *without* tool_calls is the final answer; its
         body has already been streamed out as ``text_delta`` events, so we
         finish with a ``done`` event carrying citations

    A small capability inventory is appended to the answer system prompt so the
    model knows which MCP servers / skills exist without having to call the
    registry first.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage

from app.llm.factory import _build_model
from app.agent.state import AgentState
from app.agent.prompts import get_answer_instructions
from app.agent.context import build_messages, format_verify_block
from app.agent.tools import load_tools, inventory_text
from app.config import settings


_CITE_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _coerce(content) -> str:
  """Normalise LangChain message content (str | list[dict] | None) to str."""
  if content is None:
    return ""
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    parts = []
    for part in content:
      if isinstance(part, dict):
        if part.get("type") == "text" and isinstance(part.get("text"), str):
          parts.append(part["text"])
        else:
          parts.append(str(part))
      else:
        parts.append(str(part))
    return "".join(parts)
  return str(content)


def _build_messages(state: AgentState):
  chat = _build_model(
    provider=state.get("provider_override"),
    model=state.get("model_override"),
    api_key=state.get("api_key_override"),
    base_url=state.get("base_url_override"),
    reasoning_level=state.get("reasoning_level_override"),
  )
  chunks = state.get("retrieved_chunks") or []
  question = state.get("query", "") or "(no question)"
  history = [
    m for m in (state.get("messages") or [])
    if m.get("role") in ("user", "assistant") and m.get("content")
  ]
  tools = load_tools() if settings.tools_enabled else []
  inventory = inventory_text() if settings.tools_enabled else ""
  # Sub-agent mode gets a restricted tool palette. The general profile
  # keeps every tool; explore/plan shed obviously-mutating tools so the
  # helper cannot accidentally fs_write / delete a note while summarising.
  subagent_mode = state.get("subagent_mode")
  if subagent_mode and tools:
    try:
      from app.agent.subagents import _filter_tools_for_mode
      allowed = set(_filter_tools_for_mode([t.name for t in tools], subagent_mode))
      tools = [t for t in tools if t.name in allowed]
    except Exception:
      pass
  msgs = build_messages(
    # Read per-request (mtime-cached) so config.yaml prompt edits hot-reload
    # without a backend restart; a module-level constant froze the YAML.
    instructions=get_answer_instructions(),
    chunks=chunks,
    history=history,
    question=question,
    summary=state.get("summary") or "",
    profile=state.get("profile") or None,
    memory_facts=state.get("memory_facts") or None,
    project_rules=state.get("project_rules") or "",
    inventory=inventory,
    verify_block=format_verify_block(
      state.get("web_verify_status") or "",
      state.get("web_verify_note") or "",
      state.get("web_verify_conflicts") or [],
      state.get("kb_stale_note_ids") or [],
    ),
  )
  # (inventory now flows through build_messages directly)
  return chat, msgs, chunks, tools, inventory


def _citations_from_text(text: str, chunks: list) -> list:
  if not text or not chunks:
    return []
  seen: set = set()
  indices: list[int] = []
  for m in _CITE_RE.finditer(text):
    idx = int(m.group(1)) - 1
    if idx < 0 or idx >= len(chunks) or idx in seen:
      continue
    seen.add(idx)
    indices.append(idx)
  indices.sort()
  out: list = []
  for idx in indices:
    c = chunks[idx]
    out.append({
      "note_id": c.get("note_id"),
      "title": c.get("title"),
      "chunk_index": c.get("chunk_index"),
      "snippet": (c.get("text") or "")[:240],
      "score": c.get("final_score"),
      "source_type": c.get("source_type", ""),
      "source_url": c.get("source_url", ""),
    })
  return out


def _tool_by_name(tools, name):
  for t in tools:
    if getattr(t, "name", None) == name:
      return t
  return None


class _ToolStreamAgg:
  """聚合一次带工具绑定模型的流式响应。

  - 文本增量随到随吐（add 返回待转发的 delta），实现 token 级流式；
  - tool_call 分片按 index 聚合，流结束后统一 json.loads 出完整参数；
  - 一旦出现 tool_call 分片即停止转发文本，避免工具调用前后的杂讯
    泄漏进最终回答。
  """

  def __init__(self):
    self.text = ""
    self.tool_calls: list[dict] = []
    self._agg: dict[int, dict] = {}
    self._saw_tool = False
    self._last_idx: int | None = None

  def add(self, chunk) -> str:
    """喂入一个 AIMessageChunk，返回应转发给用户的文本增量（可为 ''）。"""
    tccs = list(getattr(chunk, "tool_call_chunks", None) or [])
    # 部分供应商直接给解析好的完整 tool_calls（无分片）
    parsed = list(getattr(chunk, "tool_calls", None) or [])
    if tccs or parsed:
      self._saw_tool = True
    for tcc in tccs:
      get = tcc.get if isinstance(tcc, dict) else (lambda k, d=None: getattr(tcc, k, d))
      idx = get("index")
      if idx is None:
        # 无 index 的分片视为上一个调用的续传（部分供应商不回传 index）
        idx = self._last_idx if self._last_idx is not None else 0
      else:
        self._last_idx = idx
      entry = self._agg.setdefault(idx, {"name": "", "args": "", "id": ""})
      if get("name"):
        entry["name"] += get("name")
      if get("args"):
        entry["args"] += get("args")
      if get("id"):
        entry["id"] = get("id")
    for tc in parsed:
      # langchain 会从单个流式分片派生 parsed tool_calls：name 只在首个分片
      # 出现时，后续 args 分片派生出空名条目；且与 _agg 分片聚合重复。
      # 空名/重复条目进入 AIMessage(tool_calls=...) 历史后回传上游，会触发
      # MiniMax 400 invalid tool calls count (2013)。parsed 仅在供应商完全
      # 不发原始分片（直接给完整调用）时才可信。
      if tccs:
        break
      name = (tc.get("name") or "").strip()
      if not name:
        continue
      if self._has_call(name, tc.get("id") or ""):
        continue
      self.tool_calls.append({
        "name": name,
        "args": tc.get("args") or {},
        "id": tc.get("id") or "",
      })
    content = getattr(chunk, "content", "") or ""
    if content and not self._saw_tool:
      delta = _coerce(content)
      self.text += delta
      return delta
    return ""

  def _has_call(self, name: str, call_id: str) -> bool:
    """判断某工具调用是否已收集（按 id 精确匹配，无 id 时按名匹配）。"""
    for t in self.tool_calls:
      if call_id and (t.get("id") or "") == call_id:
        return True
      if not call_id and t.get("name") == name:
        return True
    return False

  def finish(self) -> list[dict]:
    """流结束后解析聚合出的 tool_calls（可多次调用，幂等）。"""
    for i in sorted(self._agg):
      e = self._agg[i]
      if not e["name"]:
        continue
      try:
        args = json.loads(e["args"]) if e["args"] else {}
      except Exception:
        args = {}
      # 与 parsed 通道（若供应商混发两种形态）按 id/名去重，避免同一调用
      # 被执行两次并回传重复历史。
      if self._has_call(e["name"], e["id"]):
        continue
      self.tool_calls.append({"name": e["name"], "args": args, "id": e["id"]})
    self._agg.clear()
    return self.tool_calls


def _forced_notice(state: AgentState) -> str:
  """冲突 / 知识库过期时，由服务端**强制**插到答案最前面的提示块。

  为什么不能只靠提示词：用户要的是「强制挂冲突警告」。提示词是软约束，
  模型完全可以不照做（尤其是它自认为已经解释清楚的时候）。服务端拼接是
  硬保证 —— 哪怕模型完全不配合，用户也一定会看到这条警告。

  只在 conflict / kb_stale 两个 status 下输出：
    - conflict：两边说法不一致，用户绝不能被蒙在鼓里当成「已确认」。
    - kb_stale：知识库内容被判定过期并从上下文剔除了，用户必须知道
      「这次回答没有依据你的旧资料」，否则会误以为知识库仍然生效。
  unverified / web_only 不加横幅 —— 它们是常态（联网不可用、知识库本来就没有
  相关材料），每轮都挂横幅会让人对横幅脱敏，反而削弱 conflict 的警示作用。
  """
  status = (state.get("web_verify_status") or "").strip().lower()
  if status == "conflict":
    lines = [
      "> ⚠️ **检测到知识库与联网结果存在冲突，以下内容未经裁定，请人工复核。**",
    ]
    conflicts = state.get("web_verify_conflicts") or []
    if conflicts:
      lines.append(">")
      for c in conflicts[:5]:
        if not isinstance(c, dict):
          continue
        claim = str(c.get("claim") or "").strip()
        if claim:
          lines.append("> - **%s**" % claim)
        kb_says = str(c.get("kb_says") or "").strip()
        web_says = str(c.get("web_says") or "").strip()
        if kb_says:
          lines.append(">   - 知识库：%s" % kb_says)
        if web_says:
          lines.append(">   - 联网：%s" % web_says)
    return "\n".join(lines) + "\n\n"

  if status == "kb_stale":
    stale = state.get("kb_stale_note_ids") or []
    hint = ("（已剔除来源：%s）" % "、".join(str(x) for x in stale[:5])) if stale else ""
    return (
      "> ⚠️ **知识库中的相关资料已过期，本次回答未采用旧内容。**%s\n\n" % hint
    )
  return ""


def answer_node(state: AgentState) -> dict:
  """Non-streaming variant: single structured call. No tool-calling here."""
  chat, msgs, chunks, _tools, _inventory = _build_messages(state)
  from langchain_core.output_parsers import PydanticOutputParser
  from app.agent.schemas import AnswerResult
  structured = chat.with_structured_output(AnswerResult)
  result = structured.invoke(msgs)
  citations = [c.model_dump() for c in (result.citations or [])] or _citations_from_text(result.text, chunks)
  return {
    "answer": _forced_notice(state) + (result.text or ""),
    "citations": citations,
    "step_count": state.get("step_count", 0) + 1,
  }


async def answer_node_stream(state: AgentState, instructions_override=None):
  """Stream the answer as one LLM call, then derive citations from [n] markers.

  Tool-calling path: when tools are available we bind them and run a tool-call
  loop. ``tool_call`` and ``tool_result`` SSE events surface every invocation
  (the frontend can ignore them silently). Every model turn is streamed
  (``astream``): the final answer arrives as incremental ``text_delta``
  events, followed by ``done`` with citations.

  ``instructions_override`` lets sub-agents (app/agent/subagents.py)
  substitute the default system prompt with their profile-specific text.
  Defaults to None which preserves the existing behaviour.
  """
  chat, msgs, chunks, tools, _inventory = _build_messages(state)
  if instructions_override:
    from langchain_core.messages import SystemMessage as _SM
    if msgs and isinstance(msgs[0], _SM):
      msgs = [_SM(content=instructions_override + "\n\n" + (msgs[0].content or ""))] + list(msgs[1:])
    else:
      msgs = [_SM(content=instructions_override)] + list(msgs)
  full_text = ""
  # 冲突 / 知识库过期时，先把强制警告作为首个 text_delta 推出去。
  # 单独存一份而不是并进 full_text：下面工具路径用的是 `full_text = agg.text`
  # （赋值而非累加），并进去会被整个覆盖掉，警告就丢了。
  notice = _forced_notice(state)
  if notice:
    yield ("text_delta", notice)

  # 权限模式：default=本地能力调用需用户逐次批准；full=不询问。
  # 同时把模式同步给 mcp_tools（限定 filesystem server 的授权目录）。
  from app.agent.tools import mcp_tools as _mcp_tools
  from app.agent.tools import permissions as _perm
  perm_mode = (state.get("agent_permission") or "default").lower()
  _mcp_tools.set_permission_mode(perm_mode)
  turn_approved_targets: set[str] = set()  # 同一目标已批准过则不再询问

  if tools and settings.tools_enabled:
    from langchain_core.messages import ToolMessage
    bound = chat.bind_tools(tools)
    max_steps = max(1, int(getattr(settings, "tools_max_steps", 4) or 4))
    # 必须在进入循环前初始化：下面 `if not tcs` 分支会读它来判断"是否已经
    # 执行过工具"。首轮如果模型直接作答（没有 tool_calls）——这是最常见的情
    # 况——就会读到未定义变量并抛 UnboundLocalError，整条回答流中断。
    executed_any = False

    for _step in range(max_steps):
      agg = _ToolStreamAgg()
      try:
        async for chunk in bound.astream(msgs):
          delta = agg.add(chunk)
          if delta:
            yield ("text_delta", delta)
      except Exception as e:
        yield ("error", "%s: %s" % (type(e).__name__, e))
        return

      tcs = agg.finish()
      if not tcs:
        # After tool execution the LLM sometimes streams a short transitional
        # sentence with no tool calls ("let me first see what tools are
        # available..."). That is NOT the final answer -- force another
        # iteration with a stronger FINAL-ANSWER nudge so the model keeps
        # going until it actually answers the user.
        if executed_any and _step < max_steps - 1:
          from langchain_core.messages import SystemMessage as _SM
          msgs = msgs + [_SM(content=(
            "Tool results are in the messages above. The previous response was "
            "just transitional chatter. Produce the FINAL ANSWER to the user's "
            "original question using those results. Do NOT call more tools. Do "
            "NOT describe what you would do. Output the answer directly, keep "
            "it concise and use [n] markers to cite the relevant sources."
          ))]
          continue
        full_text = agg.text
        break
      # 上游（MiniMax 2013）要求 assistant.tool_calls[].id 非空且与后续
      # ToolMessage 一一对应；供应商漏发 id 时补一个稳定值。
      for _i, _tc in enumerate(tcs):
        if not (_tc.get("id") or "").strip():
          _tc["id"] = "call_s%d_%d" % (_step, _i)

      yield ("tool_calls_batch", {
        "count": len(tcs),
        "names": [tc.get("name") for tc in tcs],
      })
      resp = AIMessage(content=agg.text, tool_calls=tcs)
      msgs = msgs + [resp]
      executed_any = False
      for tc in tcs:
        name = tc.get("name") or "unknown_tool"
        args = tc.get("args") or {}
        tc_id = tc.get("id") or ""
        yield ("tool_call", {"name": name, "args": args})
        tool = _tool_by_name(tools, name)

        # ---- 权限门控（默认模式）----
        # mcp_invoke 意味着访问本地能力（文件/命令/网络），默认模式下
        # 先经用户批准：yield permission_request -> 前端弹窗 -> POST 决定。
        #
        # 注意 tool 是 LangChain 的 StructuredTool **对象**（_tool_by_name 用
        # getattr(t, "name") 匹配后返回的对象），不是 dict。mcp_invoke 是通用
        # 工具，真正的目标 server 在 args["server_id"] 里。早期这里写成
        # tool.get("server")，导致每次 mcp_invoke 都抛
        # AttributeError: 'StructuredTool' object has no attribute 'get'。
        # 取不到 server_id 时 target 为空串，永远不等于已批准集合 —— 即失败时
        # 倾向再问一次（fail-closed），符合权限门控的预期。
        denied_by_permission = False
        target = (args.get("server_id") or "") if isinstance(args, dict) else ""
        if (perm_mode != "full" and name == "mcp_invoke" and tool is not None
                and target not in turn_approved_targets):
          req_id, _fut = _perm.create_request()
          yield ("permission_request", {
            "request_id": req_id,
            "tool": name,
            "args": args,
          })
          approved = await _perm.wait_decision(req_id)
          yield ("permission_result", {"request_id": req_id, "approved": approved})
          if approved:
            # Per-target broker: record this exact server so
            # subsequent same-server calls in this turn skip the modal.
            if target:
                turn_approved_targets.add(target)
            # 把本次请求涉及的盘符根加入授权范围（后续同轮调用不再被
            # filesystem server 的 allowed-dirs 拦截）
            args = args if isinstance(args, dict) else {}
            _mcp_tools.add_approved_dirs_from(args)
          else:
            denied_by_permission = True

        if denied_by_permission:
          observation = (
            "[权限拒绝] 用户未授权本次本地访问。请直接告诉用户：如需让助手"
            "访问本地文件，可以在聊天输入框下方开启「完全访问」权限后重试。"
          )
          ok = False
        elif tool is None:
          observation = "[tool error] unknown tool %r" % name
          ok = False
        else:
          try:
            out = await tool.ainvoke(args)
            observation = out if isinstance(out, str) else str(out)
            ok = True
            executed_any = True
          except Exception as e:
            observation = "[tool error] %s: %s" % (type(e).__name__, e)
            ok = False
        snippet = observation[:300] if isinstance(observation, str) else str(observation)[:300]
        yield ("tool_result", {"name": name, "ok": ok, "snippet": snippet})
        msgs = msgs + [ToolMessage(content=observation, tool_call_id=tc_id)]
      # Nudge the model to keep going: without this, models tend to reply with
      # "here are the steps I would take" instead of actually calling the next
      # tool (observed with MiniMax-Text-01).
      from langchain_core.messages import SystemMessage as _SM
      msgs = msgs + [_SM(content=(
        "Tool call(s) executed — the results are in the tool messages above. "
        "If the user's task is NOT yet fully answered with real data, "
        "immediately call the next tool (e.g. mcp_discover_tools or "
        "mcp_invoke); do NOT describe what you would do. Only give the final "
        "answer once you have the actual results."
      ))] if executed_any else msgs
    else:
      # ran out of steps without a text answer; force one last pass
      agg = _ToolStreamAgg()
      try:
        async for chunk in bound.astream(msgs):
          delta = agg.add(chunk)
          if delta:
            yield ("text_delta", delta)
        full_text = agg.text
      except Exception as e:
        yield ("error", "%s: %s" % (type(e).__name__, e))
        return
  else:
    try:
      async for chunk in chat.astream(msgs):
        delta = getattr(chunk, "content", "") or ""
        if delta:
          full_text += delta
          yield ("text_delta", delta)
    except Exception as e:
      yield ("error", "%s: %s" % (type(e).__name__, e))
      return

  citations = _citations_from_text(full_text, chunks)
  # 这里**不再**做"模型没写 [n] 就把所有检索切片当引用"的兜底。
  #
  # 旧实现的问题：提示词明确要求「参考材料为空或不相关时，改用你自己的知识回答
  # 且不要提参考材料」。模型据此作答时不会写 [n]，但兜底仍会把全部检索切片塞进
  # citations，前端 validSourceTokens 又把它们全渲染成「来源：[n]」——
  # 用户看到"有引用"，会以为答案有知识库依据，而模型其实压根没参考它们。
  #
  # 现在语义是：没有 [n] = 模型没有引用任何切片 = 不显示来源。
  # 这正好也是用户判断"答案是否来自我的资料"的唯一可靠信号。
  # （「检索到了什么」属于调试信息，不应伪装成引用来源。）
  # answer 带上 notice：流式期间它已经单独推给前端了，这里补上是为了让
  # 落库的对话记录与最终 answer 字段也包含警告 —— 否则历史回看时警告消失。
  yield ("done", {"answer": notice + full_text, "citations": citations})