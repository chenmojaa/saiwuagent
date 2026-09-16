"""Hybrid search: vector (Chroma) + keyword (FTS5) -> dedupe + rerank."""
from __future__ import annotations

import logging
import re
from sqlmodel import text

from app.embeddings.factory import embed_texts
from app.storage.vector import search as vector_search
from app.storage.db import get_engine
from app.config import settings
# dropped before the LLM sees it, so turns like "my name is X" no
# longer get a fake citation card.
#
# 阈值改为可配置（HD_RETRIEVAL_MIN_SCORE / HD_RETRIEVAL_MIN_DIM_SCORE）。
# 原值 0.18 过于宽松：实测「推荐合肥蜀山区野钓的地点」会召回一批只是词面
# 沾了"安徽/合肥"的文旅切片（0.24~0.41），随后被模型当成"来源"引用展示。
# 实测相关查询最低 ≈0.42、不相关最高 ≈0.41，故默认提到 0.45。
MIN_FINAL_SCORE = float(settings.retrieval_min_score)   # 0.7 * vec + 0.3 * kw
MIN_DIM_SCORE = float(settings.retrieval_min_dim_score)  # max(vec_score, kw_score)
from app.storage.db import fts_search


def _make_key(note_id: str, chunk_index: int) -> str:
  return f"{note_id}#{chunk_index}"


def _fts_escape(query: str) -> str:
  """Make a string safe for SQLite FTS5 MATCH.

  FTS5 treats double quotes and a few ASCII chars as syntax. Strip them and
  wrap the cleaned string in double quotes so the engine treats it as a phrase
  and tokenizes CJK characters correctly even for 2-3 char queries.
  """
  if not query:
    return ""
  s = query.strip()
  if not s:
    return ""
  # FTS5 reserved chars: " ( ) * ^ : AND OR NOT NEAR
  for ch in (chr(34), '(', ')', '*', '^', ':'):
    s = s.replace(ch, ' ')
  s = s.strip()
  if not s:
    return ""
  return chr(34) + s + chr(34)



def hybrid_search(
  query: str,
  top_k: int = 5,
  api_key: str | None = None,
  base_url: str | None = None,
  model: str | None = None,
  min_score: float = MIN_FINAL_SCORE,
  min_dim_score: float = MIN_DIM_SCORE,
  source_type: str | None = None,
  tag: str | None = None,
  date_from: str | None = None,
  date_to: str | None = None,
  rerank: bool = False,
  expand: bool = False,
) -> list[dict]:
  """Combine vector + keyword search, dedupe, return top_k.

  base_url lets the caller reuse the same OpenAI-compatible endpoint that
  chat is using, so embeddings stay consistent with whatever was at ingest.
  """
  # vector search. MiniMax embo-01 does not accept mode="query"
  # (status_code=2013) so we pick the mode up-front from the resolved URL,
  # avoiding the round-trip of "try query, fail, retry db" on every search.
  vec_hits: list[dict] = []
  try:
    from app.embeddings.factory import _resolve_url as _ru
    _resolved_url = _ru(base_url)
    _mode = "db" if "minimax" in _resolved_url.lower() else "query"
    emb = embed_texts([query], api_key=api_key, base_url=base_url, model=model, mode=_mode)[0]
    vec_hits = vector_search(emb, top_k=top_k * 2)
    for h in vec_hits:
      d = h.get("distance")
      h["vec_score"] = max(0.0, 1.0 - (d if d is not None else 1.0))
  except Exception as e:
    logging.getLogger(__name__).warning("hybrid_search vector embed failed: %s", e)

  # keyword search (FTS5 / BM25)
  kw_hits = []
  try:
    safe_q = _fts_escape(query)
    if safe_q:
      kw_hits = fts_search(safe_q, top_k=top_k * 2)
      for h in kw_hits:
        s = h.get("score") or 0.0
        h["kw_score"] = 1.0 / (1.0 + abs(s))
  except Exception as e:
    logging.getLogger(__name__).warning("hybrid_search fts failed: %s", e)

  # merge + dedupe by (note_id, chunk_index)
  merged = {}
  for h in vec_hits:
    k = _make_key(h["note_id"], h["chunk_index"])
    merged[k] = {
      "note_id": h["note_id"],
      "chunk_index": h["chunk_index"],
      "text": h["text"],
      "vec_score": h.get("vec_score", 0.0),
      "kw_score": 0.0,
    }
  for h in kw_hits:
    k = _make_key(h["note_id"], h["chunk_index"])
    if k in merged:
      merged[k]["kw_score"] = h.get("kw_score", 0.0)
    else:
      merged[k] = {
        "note_id": h["note_id"],
        "chunk_index": h["chunk_index"],
        "text": h["text"],
        "vec_score": 0.0,
        "kw_score": h.get("kw_score", 0.0),
      }

  # combined: vector 0.7 + keyword 0.3
  scored = [
    (
      0.7 * v["vec_score"] + 0.3 * v["kw_score"],
      max(v["vec_score"], v["kw_score"]),
      v,
    )
    for v in merged.values()
  ]
  ranked = sorted(scored, key=lambda t: t[0], reverse=True)

  from app.storage.db import get_note_title, get_note_meta
  results = []
  for final_s, dim_s, r in ranked:
    r["title"] = get_note_title(r["note_id"])
    r["final_score"] = round(final_s, 4)
    # Relevance gate: drop pure-noise chunks so conversational queries
    # (e.g. "my name is X") no longer get fake citation cards.
    if final_s < min_score:
      continue
    if dim_s < min_dim_score:
      continue
    # Attach source metadata for frontend display (uploaded vs feishu)
    meta = get_note_meta(r["note_id"])
    if meta:
      r["source_type"] = meta["source_type"]
      r["source_url"] = meta["source_url"]
    results.append(r)
    if len(results) >= top_k:
      break
  return results


def filter_by_meta(results, source_type=None, tag=None, date_from=None, date_to=None):
  """Post-filter merged results by note metadata. No-op if no filters given."""
  if not any([source_type, tag, date_from, date_to]):
    return results
  from datetime import datetime
  from app.storage.db import get_note_meta
  out = []
  df = None
  dt = None
  try:
    if date_from: df = datetime.fromisoformat(date_from)
    if date_to:   dt = datetime.fromisoformat(date_to)
  except Exception:
    pass
  for r in results:
    meta = get_note_meta(r["note_id"]) or {}
    if source_type and meta.get("source_type") != source_type:
      continue
    if tag:
      tags = (meta.get("tags") or "").split(",")
      if tag not in [t.strip() for t in tags if t.strip()]:
        continue
    if df or dt:
      ts = meta.get("created_at")
      if ts:
        try:
          d = datetime.fromisoformat(str(ts).replace("Z", ""))
          if df and d < df: continue
          if dt and d > dt: continue
        except Exception:
          pass
    out.append(r)
  return out


def hybrid_search_with_expansion(query, top_k=5, **kwargs):
  """hybrid_search + optional query expansion + optional rerank."""
  from app.agent.retrieval_quality import expand_query, rerank
  # Caller-supplied credentials must reach expand_query + rerank so the
  # LLM helper uses the requester's provider instead of silently falling
  # back to settings.router_* (per-request override was previously ignored).
  _rq_api_key = kwargs.get("api_key")
  _rq_base_url = kwargs.get("base_url")
  if kwargs.get("expand"):
    variants = expand_query(query, api_key=_rq_api_key, base_url=_rq_base_url)
    merged_acc = []
    seen = set()
    for v in variants:
      sub = hybrid_search(v, top_k=top_k, **{k: vv for k, vv in kwargs.items() if k not in ("expand", "rerank")})
      for r in sub:
        k = "%s#%s" % (r["note_id"], r["chunk_index"])
        if k not in seen:
          seen.add(k)
          r["_expansion_variant"] = v
          merged_acc.append(r)
    results = merged_acc[:top_k * 2]
  else:
    results = hybrid_search(query, top_k=top_k, **{k: vv for k, vv in kwargs.items() if k not in ("expand", "rerank")})

  if kwargs.get("rerank") and len(results) > 1:
    results = rerank(query, results, api_key=_rq_api_key, base_url=_rq_base_url)
  # Parent-child dedup + context window (always on; pass expand_window=0 to opt out).
  expand_window = kwargs.pop("expand_window", 2)
  if expand_window != 0 and results:
    results = expand_search_results(results, window=expand_window)
  return results[:top_k]

# === Parent-child context expansion ===
def _fetch_adjacent_chunks_from_fts(note_id: str, center_index: int, window: int) -> dict[int, str]:
  """Pull adjacent chunks (chunk_index in [center-window, center+window]) from FTS5.

  Used by `merge_needs` to expand the visible context around a matched child chunk
  so the LLM sees the surrounding text instead of a single 500-char slice. The
  FTS5 store is the source of truth for chunk content because it does not require
  a re-embed roundtrip the way the Chroma store would.
  """
  if window <= 0:
    return {center_index: ""}
  try:
    with get_engine().begin() as conn:
      rows = conn.execute(text(
        "SELECT chunk_index, content FROM chunk_fts "
        "WHERE note_id = :nid AND chunk_index BETWEEN :lo AND :hi"
      ), {"nid": note_id, "lo": max(0, center_index - window), "hi": center_index + window}).all()
  except Exception:
    return {center_index: ""}
  return {int(r.chunk_index): str(r.content or "") for r in rows}


def _bucket_hits_by_note(hits: list[dict]) -> dict[str, list[dict]]:
  out: dict[str, list[dict]] = {}
  for h in hits:
    out.setdefault(h.get("note_id") or "?", []).append(h)
  for v in out.values():
    v.sort(key=lambda x: int(x.get("chunk_index") or 0))
  return out


def merge_neighboring_hits(hits: list[dict], window: int = 2, max_keep: int = 50) -> list[dict]:
  """Merge hits whose chunks sit within `window` of each other inside the same note.

  Effect: a single user-facing citation card now carries the surrounding context
  (parent-window) around the matched child chunk, instead of returning N near-
  duplicate slices from the same document. The highest-scoring child's score wins,
  so dedup never buries a strongly-matched child behind a weakly-matched sibling.

  Args:
    hits: hybrid_search output, sorted by final_score desc.
    window: number of adjacent chunks to expand on each side.
    max_keep: hard cap so we never materialize more than this many expanded hits.

  Returns a new list; the input is not mutated.
  """
  if not hits:
    return []
  if len(hits) <= 1:
    # still expand the lone hit so the LLM gets a wider window
    out = []
    for h in hits[:max_keep]:
      out.append(_expand_single(h, window))
    return out

  # First pass: coalesce neighbors inside each note.
  coalesced: list[dict] = []
  for note_id, group in _bucket_hits_by_note(hits).items():
    used = [False] * len(group)
    for i, anchor in enumerate(group):
      if used[i]:
        continue
      used[i] = True
      cluster = [anchor]
      for j in range(i + 1, len(group)):
        if used[j]:
          continue
        if int(group[j]["chunk_index"]) - int(cluster[-1]["chunk_index"]) <= window:
          cluster.append(group[j])
          used[j] = True
      # collapse cluster into one record; highest final_score wins as the canonical child
      cluster.sort(key=lambda x: x.get("final_score") or 0, reverse=True)
      winner = cluster[0]
      siblings = cluster[1:]
      winner = dict(winner)
      winner["sibling_chunk_indices"] = [int(s["chunk_index"]) for s in siblings]
      winner["merged_children"] = len(cluster)
      coalesced.append(winner)

  # Second pass: rank and trim.
  coalesced.sort(key=lambda h: h.get("final_score") or 0, reverse=True)
  coalesced = coalesced[:max_keep]

  # Third pass: actually expand with sibling context.
  return [_expand_single(h, window) for h in coalesced]


def _expand_single(hit: dict, window: int) -> dict:
  """Replace hit['text'] with the child text + adjacent chunks from FTS5.

  The original matched text is kept under hit['matched_text'] so the frontend
  can still highlight what was actually relevant if it wants to.
  """
  hit = dict(hit)
  note_id = hit.get("note_id") or ""
  try:
    center = int(hit.get("chunk_index") or 0)
  except Exception:
    center = 0
  original = hit.get("text") or ""
  hit["matched_text"] = original
  if window <= 0 or not note_id:
    return hit
  adj = _fetch_adjacent_chunks_from_fts(note_id, center, window)
  if not adj:
    return hit
  ordered = sorted(adj.items())
  parts = [content for _, content in ordered if content]
  hit["text"] = "\n\n".join(parts)
  hit["context_window"] = window
  hit["context_chunk_count"] = len(adj)
  return hit


def expand_search_results(hits: list[dict], window: int = 2) -> list[dict]:
  """Public entry: dedupe + expand child hits with their surrounding context.

  Tuned defaults: window=2 means each returned citation shows ~5 chunks total
  (~2500 chars of context) which is the right scale for an LLM to ground
  an answer. Callers can opt out by passing window=0.
  """
  return merge_neighboring_hits(hits, window=window)
