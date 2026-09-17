"""LangGraph state schema for the HEAR agent."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
  # ---- conversation ----
  # Replace semantics (no reducer): chat.py seeds the full history from the DB
  # on every call (server is the source of truth), so the input must overwrite
  # whatever the checkpointer stored. With an operator.add reducer the seeded
  # history would be appended to the checkpointed copy and duplicate every turn.
  messages: list
  session_id: str

  # ---- long-term memory (§6.5) ----
  profile: dict                     # user profile facts, injected by chat.py
  summary: str                      # compressed summary of history outside the window
  memory_facts: list                # recalled cross-session facts (long-term memory)

  # ---- current request ----
  query: str
  intent: str                       # chat | research | ingest | report (router output)
  rewritten_query: str              # router output: original query + 3-turn history resolved

  # ---- task planning (plan-and-execute) ----
  plan: list                        # planner output: [{"query": <sub-question>}, ...]
  plan_summary: str                 # one-sentence description of the overall approach
  plan_cursor: int                  # index of the NEXT plan step to execute
  plan_status: list                 # per-step execution records: [{query, hits, new_chunks}]
  replan_stalled: bool              # replan loop could not produce a new query -> stop
  use_planner: bool | None          # per-request override; None = follow HD_PLANNER_ENABLED

  # ---- query clarification (model-driven, always on) ----
  clarify_request: dict | None      # router output: {"question": str, "options": [str]} when ambiguous
  skip_clarify: bool                # second-pass guard: answer collected (or skipped), never re-ask

  # ---- retrieval results ----
  retrieved_chunks: list
  skip_retrieval: bool              # fast-path: skip retrieve node for greetings
  research_iterations: int          # how many rounds research agent ran
  research_notes: list              # intermediate follow-up queries produced during research
  # 联网兜底是否已尝试过（本地完全无结果时才会走）。
  # 只标记"试过没有"，避免 replan 反复联网；联网结果只进 retrieved_chunks，
  # **不写回知识库**。
  web_search_used: bool

  # ---- 联网校验（策略 A：优先知识库 + 联网做校验）----
  # 与 web_search_used 的区别：那是「兜底」（知识库空才联网），
  # 这是「校验」（知识库有结果也联网核对时效性与事实冲突）。
  # web_verify_status 取值：
  #   consistent         知识库与联网一致，以知识库为主作答
  #   conflict           存在事实性冲突，作答但强制挂冲突警告
  #   kb_stale           知识库明显过期，过期片段已从上下文剔除
  #   web_only           知识库无相关材料，仅凭联网结果作答
  #   unverified         联网失败或裁决不可用，未能核对（不阻断作答）
  #   skipped            校验被关闭 / 非时效敏感问题跳过
  #   disabled           总开关关闭
  web_verify_status: str
  web_verify_conflicts: list        # [{claim, kb_says, web_says, kb_refs, web_refs}]
  web_verify_note: str              # 裁决模型给的一句话说明（用于日志/提示）
  kb_stale_note_ids: list           # 判定过期的知识库 note_id，作答时剔除
  pending_candidates: list          # 待审核候选：联网结果，绝不自动并入主知识库

  # ---- ingest agent output ----
  ingest_result: dict               # {title, tags, summary, note_id, duplicate_of, ...}

  # ---- report agent output ----
  report_result: dict               # {note_id, period, counts, summary}

  # ---- final answer ----
  answer: str | None
  citations: list

  # ---- per-request overrides ----
  provider_override: str | None
  model_override: str | None
  base_url_override: str | None
  api_key_override: str | None
  reasoning_level_override: str | None
  embedding_model_override: str | None
  step_count: int

  # ---- local-access permission ----
  # 'default': agent must ask the user before mcp_invoke (local file/cmd access);
  # 'full':    no asking, filesystem scope = all drives.
  agent_permission: str
