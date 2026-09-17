"""LangGraph graph: router -> (retrieve | planner -> execute_plan* -> replan* | ingest | report).

Per OPTIMIZATION.md section 2.4. The router classifies intent and rewrites the
query for downstream nodes. research intent runs the plan-and-execute loop:

  planner -> execute_plan (one sub-query per pass, loops until the plan is
  spent or enough material is collected) -> replan (dynamic follow-up rounds
  when still short) -> END

Because execute_plan / replan are loop nodes, graph.astream emits one event
per step, which chat.py forwards as per-step SSE progress (no more black-box
research). ingest/report terminate at END; the retrieve node (plain chat
intent) populates ``retrieved_chunks`` and terminates at END; chat.py then
streams the answer with ``answer_node_stream`` so reasoning models do not
have to pass through a separate structured-output call.

HD_USE_GRAPH=false (env) keeps the legacy direct-call path in chat.py active;
default is graph-driven.
"""
from __future__ import annotations

import os

import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.state import AgentState
from app.config import settings
from app.agent.nodes.ingest import ingest_node
from app.agent.nodes.planner import planner_node
from app.agent.nodes.report import report_node
from app.agent.nodes.research import execute_plan_node, replan_node
from app.agent.nodes.retrieve import retrieve_node
from app.agent.nodes.router import route_by_intent, router_node
from app.agent.nodes.verify import verify_node


def route_after_planner(state: AgentState) -> str:
  """有计划 -> 逐步执行；空计划（规划失败/关闭）-> 直接 replan 兜底。"""
  return "execute_plan" if (state.get("plan") or []) else "replan"


def _need_more(state: AgentState) -> bool:
  """材料不足且 replan 还有预算且没卡住。

  联网兜底一旦做过就直接收尾：本地已证明没有相关内容，
  再让模型换角度去搜本地只是空转（联网也不会重复触发）。
  """
  if state.get("web_search_used"):
    return False
  target = max(1, int(settings.research_target_chunks))
  collected = len(state.get("retrieved_chunks") or [])
  replans = len(state.get("research_notes") or [])
  max_replans = max(0, int(settings.research_max_iter))
  return (collected < target and replans < max_replans
          and not state.get("replan_stalled"))


def route_after_step(state: AgentState) -> str:
  """计划步执行后的去向：继续执行下一步 / 转 replan / 转校验。"""
  plan = state.get("plan") or []
  cursor = int(state.get("plan_cursor") or 0)
  target = max(1, int(settings.research_target_chunks))
  collected = len(state.get("retrieved_chunks") or [])
  if cursor < len(plan) and collected < target:
    return "execute_plan"
  if _need_more(state):
    return "replan"
  return "verify"


def route_after_replan(state: AgentState) -> str:
  """replan 一轮后的去向：继续补缺 / 转校验。"""
  return "replan" if _need_more(state) else "verify"


def _build_workflow() -> StateGraph:
  g = StateGraph(AgentState)
  g.add_node("router", router_node)
  g.add_node("planner", planner_node)
  g.add_node("execute_plan", execute_plan_node)
  g.add_node("replan", replan_node)
  g.add_node("retrieve", retrieve_node)
  g.add_node("verify", verify_node)
  g.add_node("ingest", ingest_node)
  g.add_node("report", report_node)

  g.add_edge(START, "router")
  g.add_conditional_edges("router", route_by_intent, {
    "chat": "retrieve",
    "chat_no_rag": END,
    "clarify": END,               # 歧义澄清：图先终止，chat.py 拦截并等用户回答后带 skip_clarify 重跑
    "research": "planner",       # 任务规划：复杂问题先分解再检索
    "ingest": "ingest",
    "report": "report",
  })
  g.add_conditional_edges("planner", route_after_planner, {
    "execute_plan": "execute_plan",
    "replan": "replan",
  })
  # 循环节点：execute_plan -> execute_plan / replan / verify
  g.add_conditional_edges("execute_plan", route_after_step, {
    "execute_plan": "execute_plan",
    "replan": "replan",
    "verify": "verify",
  })
  # replan -> replan / verify
  g.add_conditional_edges("replan", route_after_replan, {
    "replan": "replan",
    "verify": "verify",
  })
  # 策略 A：检索完不直接作答，先联网核对时效性与事实冲突。
  # 两条路径（chat 的 retrieve、research 的循环收尾）都必须经过它。
  g.add_edge("retrieve", "verify")
  g.add_edge("verify", END)
  g.add_edge("ingest", END)
  g.add_edge("report", END)
  return g


# ---- Lazy async graph singleton ----
# AsyncSqliteSaver.__init__ calls asyncio.get_running_loop(), so the saver (and
# therefore the compiled graph) cannot be constructed at import time. We build it
# once on first async use and cache it. The aiosqlite connection is opened lazily
# by the saver and lives for the process lifetime.
_graph = None
_conn = None


async def get_graph():
  """Return the compiled graph, building the SQLite checkpointer on first call."""
  global _graph, _conn
  if _graph is None:
    ckpt_path = os.path.join(settings.data_dir, "checkpoints.sqlite")
    os.makedirs(settings.data_dir, exist_ok=True)
    _conn = aiosqlite.connect(ckpt_path)
    checkpointer = AsyncSqliteSaver(_conn)
    _graph = _build_workflow().compile(checkpointer=checkpointer)
  return _graph
