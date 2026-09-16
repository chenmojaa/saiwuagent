"""Retrieve node - hybrid search + optional cross-encoder rerank.

Stage 1: hybrid_search() recalls a wider pool from the SQLite store.
Stage 2: cross-encoder rerank() trims that pool to top 5. Rerank failure is
caught and the node falls back to the hybrid top-5 so a missing model never
breaks chat.

召回深度是自适应的（见 `_hybrid_topk`）：只有在 CrossEncoder 真的可用时
才多召回，否则"召回 50 再截断到 5"等于白做 10 倍向量 + FTS 工作。
"""
import logging

from app.storage.hybrid import hybrid_search
from app.agent.state import AgentState

log = logging.getLogger(__name__)

# 最终送给 answer 节点的引用条数
FINAL_TOP_K = 5
# 重排可用时的召回池大小（重排需要富候选才有意义）
RECALL_WITH_RERANK = 50
# 重排不可用时的召回深度。略高于 FINAL_TOP_K，留一点余量给阈值过滤后
# 可能不足 5 条的情况，但不做无谓的大池召回。
RECALL_PLAIN = 8

_rerank_available_cache: bool | None = None


def _rerank_available() -> bool:
  """CrossEncoder 重排是否真的可用（依赖 sentence_transformers）。

  结果缓存：import 探测一次即可，不必每次检索都试。
  """
  global _rerank_available_cache
  if _rerank_available_cache is None:
    try:
      import sentence_transformers  # noqa: F401
      _rerank_available_cache = True
    except Exception:
      _rerank_available_cache = False
      log.info("retrieve: sentence_transformers 未安装，重排不可用；"
               "召回深度降为 %d（避免召回 %d 再截断的无谓开销）",
               RECALL_PLAIN, RECALL_WITH_RERANK)
  return _rerank_available_cache


def _read_content(m) -> str:
  if isinstance(m, dict):
    return m.get("content", "") or ""
  return getattr(m, "content", "") or ""


def _read_role(m) -> str:
  if isinstance(m, dict):
    return m.get("role", "") or ""
  return getattr(m, "role", "") or ""


def _hybrid_topk() -> int:
  """How many candidates to recall from hybrid before the final trim.

  重排可用 -> 宽召回（重排负责精选）；
  重排不可用 -> 窄召回，直接拿 top-N，不做无意义的 50 条大池。
  """
  return RECALL_WITH_RERANK if _rerank_available() else RECALL_PLAIN


def retrieve_node(state: AgentState) -> dict:
  messages = state.get("messages") or []
  # Prefer the router-rewritten query when present; fall back to the last user
  # message verbatim (legacy behavior).
  rewritten = (state.get("rewritten_query") or "").strip()
  raw = (state.get("query") or "").strip()
  query = rewritten or raw
  if not query:
    for m in reversed(messages):
      if _read_role(m) == "user":
        c = _read_content(m).strip()
        if c:
          query = c
          break

  if not query:
    return {"retrieved_chunks": [], "step_count": state.get("step_count", 0) + 1}

  api_key = state.get("api_key_override")
  base_url = state.get("base_url_override")
  emb_model = state.get("embedding_model_override")
  try:
    chunks = hybrid_search(query, top_k=_hybrid_topk(),
                           api_key=api_key, base_url=base_url, model=emb_model)
  except Exception as e:
    log.warning("hybrid_search failed: %s", e)
    chunks = []

  # Stage 2: cross-encoder rerank（仅在依赖可用时尝试）。Best-effort；
  # 任何异常都回退到 hybrid top-N，保证 answer 节点仍有东西可引用。
  if chunks and _rerank_available():
    try:
      from app.agent.rerank import rerank
      cands = [dict(c) for c in chunks]  # shallow copy so original chunks stay intact
      chunks = rerank(query, cands, top_k=FINAL_TOP_K)
    except Exception as e:
      log.warning("rerank failed, falling back to hybrid top-%d: %s", FINAL_TOP_K, e)
      chunks = chunks[:FINAL_TOP_K]
  else:
    chunks = chunks[:FINAL_TOP_K]

  return {
    "retrieved_chunks": chunks,
    "step_count": state.get("step_count", 0) + 1,
  }
