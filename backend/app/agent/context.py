"""Context assembly: token budgeting + unified message building (§7, §8).

This module is the single place where "what goes into the LLM prompt" is
decided. It owns:

  1. ``format_context``  - render retrieved chunks as a numbered reference block
     with a token budget (drop lowest-score chunks when over budget).
  2. ``trim_history``    - sliding-window history trim; overflow goes to a
     summary string instead of being silently dropped.
  3. ``build_messages``  - assemble the final LangChain message list:
     system (persona + profile + summary + references) -> history -> question.

Token counting is a cheap heuristic (CJK chars ~1.5 tokens each, ASCII words
~1.3 tokens each) rather than a tiktoken dependency; it is only used to decide
when to truncate, never for billing.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Iterable

_log = logging.getLogger(__name__)

_EMPTY_CONTEXT = "(no reference material available)"
_SEPARATOR = "\n\n---\n\n"

# 推理模型会把思维链写在正文里（<think>...</think> 或 <thinking>）。
# 存储层必须保留它（前端要折叠展示），但喂给模型时必须剥掉：
# 思维链对模型理解对话几乎没有价值，实测单条消息里它能占 46% 的字符，
# 而且会随会话累积——历史窗口被大量无效内容挤占。
_THINK_RE = re.compile(r"<think(?:ing)?>[\s\S]*?</think(?:ing)?>", re.IGNORECASE)


def strip_think(text: str) -> str:
    """剥掉正文里的思维链，只留真正的回答。

    未闭合的 <think>（流式截断等）也一并处理：从标记处截断到结尾。
    """
    if not text:
        return text or ""
    out = _THINK_RE.sub("", text)
    # 兜底：未闭合的 <think>（模型偶尔不写 </think>），丢弃其后全部内容
    m = re.search(r"<think(?:ing)?>", out, re.IGNORECASE)
    if m:
        out = out[: m.start()]
    return out.strip()

# ---- Token budgets (§8) ----
# Rough CJK-aware estimate: 1 CJK char ~ 1.5 tokens, 1 ASCII word ~ 1.3 tokens.
CONTEXT_TOKEN_BUDGET = 3000   # total budget for the reference block
MAX_CHUNK_CHARS = 800         # hard cap per chunk regardless of budget
HISTORY_TOKEN_BUDGET = 2000   # budget for the conversation history block


def estimate_tokens(text: str) -> int:
  """Cheap token estimate without pulling in tiktoken."""
  if not text:
    return 0
  cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
  rest = len(text) - cjk
  return int(cjk * 1.5 + rest * 0.35)


def format_context(chunks: Iterable[dict] | None,
                   token_budget: int = CONTEXT_TOKEN_BUDGET) -> str:
  """Render retrieved_chunks as a numbered reference block, within token budget.

  Chunks are kept in their original (score-descending) order. If the block
  exceeds the budget, the lowest-score chunks at the tail are dropped first.
  Each chunk is also hard-capped at MAX_CHUNK_CHARS. Numbers are 1-based and
  match the [n] citation tokens the answer prompt expects.
  """
  chunk_list = list(chunks) if chunks else []
  if not chunk_list:
    return _EMPTY_CONTEXT

  # Pre-truncate each chunk, then accumulate until budget is hit.
  rendered: list[tuple[int, str]] = []  # (original index, text)
  used = 0
  for i, c in enumerate(chunk_list):
    text = (c.get("text") or "")[:MAX_CHUNK_CHARS]
    title = c.get("title") or c.get("note_id", "?")
    # 任务规划：标注该 chunk 由哪个子查询命中，模型可按子问题组织回答
    matched = str(c.get("matched_query") or "").strip()
    if matched:
      block = "[%d] source: %s (sub-question: %s)\n%s" % (i + 1, title, matched[:80], text)
    else:
      block = "[%d] source: %s\n%s" % (i + 1, title, text)
    cost = estimate_tokens(block)
    if used + cost > token_budget and rendered:
      _log.info("context: dropped %d/%d chunks over token budget (%d/%d tokens)",
                len(chunk_list) - len(rendered), len(chunk_list), used, token_budget)
      break
    rendered.append((i, block))
    used += cost

  if not rendered:
    return _EMPTY_CONTEXT
  return _SEPARATOR.join(block for _, block in rendered)


def trim_history(messages: list[dict],
                 max_messages: int = 12,
                 token_budget: int = HISTORY_TOKEN_BUDGET) -> tuple[list[dict], list[dict]]:
  """Split history into (kept_recent, overflow).

  Keeps the most recent ``max_messages`` messages that also fit within
  ``token_budget``. Returns (recent, overflow) where overflow is the older
  messages that were cut (oldest first) - callers may summarize them.
  """
  if not messages:
    return [], []

  recent: list[dict] = []
  used = 0
  for m in reversed(messages[-max_messages:]):
    cost = estimate_tokens(m.get("content") or "")
    if used + cost > token_budget and recent:
      break
    recent.append(m)
    used += cost
  recent.reverse()

  # `recent` is always a suffix of `messages`, so the overflow is the prefix.
  overflow = messages[: len(messages) - len(recent)]
  return recent, overflow


def _profile_line(profile: dict) -> str:
  """Compress the user profile dict into one short line for the system prompt."""
  if not profile:
    return ""
  try:
    return "[user profile] " + json.dumps(profile, ensure_ascii=False)[:300]
  except Exception:
    return ""


def _strip_citation_rules(text: str) -> str:
  """Remove citation-related rules from the system prompt when no chunks exist.

  Prevents the LLM from faithfully following "末尾输出 来源：[n][m]..." and
  emitting a bogus "来源：无" line when there is nothing to cite.
  """
  lines = text.split("\n")
  out: list[str] = []
  skip = False
  for line in lines:
    stripped = line.strip()
    low = stripped.lower()
    # Skip numbered rule "Cite sources by index [n]..."
    if low.startswith("cite sources"):
      continue
    # Skip the "Citations:" bullet under Output format (and its continuation)
    if low.startswith("citations:"):
      skip = True
      continue
    # Continuation of the citations bullet (indented or starts with "listing")
    if skip and (line.startswith("    ") or line.startswith("\t") or low.startswith("listing") or low.startswith("do not")):
      continue
    # Stop skipping when we hit a non-indented line that is not empty
    if skip and stripped and not line[0].isspace():
      skip = False
    if skip:
      continue
    out.append(line)
  return "\n".join(out)


def build_messages(instructions: str,
                   chunks: list,
                   history: list[dict],
                   question: str,
                   summary: str = "",
                   profile: dict | None = None,
                   memory_facts: list[str] | None = None,
                   project_rules: str = "",
                   inventory: str = "") -> list:
  """Assemble the final LangChain message list (§7).

  Order is fixed: system (instructions + summary + memory facts + profile +
  references) -> trimmed history -> current question. The ``<<CONTEXT>>`` and
  ``<<QUESTION>>`` placeholders in ``instructions`` are filled here.

  When there are no reference chunks, citation-related rules are stripped
  from the instructions so the LLM does not output a bogus "来源：无" line.
  """
  from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

  context_block = format_context(chunks)
  has_chunks = bool(chunks)
  sys_text = instructions.replace("<<CONTEXT>>", context_block)
  sys_text = sys_text.replace("<<QUESTION>>", question)

  # Strip citation rules when there is nothing to cite — otherwise the model
  # faithfully follows the prompt and emits "来源：无" at the end.
  if not has_chunks:
    sys_text = _strip_citation_rules(sys_text)

  parts = [sys_text]
  if summary:
    parts.append("[history summary] " + summary[:400])
  # 长期记忆：召回的跨会话事实，模型应自然利用（不要求主动提及）
  facts = [str(f).strip() for f in (memory_facts or []) if str(f).strip()]
  if facts:
    parts.append("[long-term memory about this user]\n" +
                 "\n".join("- " + f[:200] for f in facts))
  line = _profile_line(profile or {})
  if line:
    parts.append(line)

  # AGENTS.md / project standing instructions (Codex CLI / Claude Code).
  # 兼容 RuleSet 对象（.text 为合并文本）与旧版 str
  rules = project_rules if isinstance(project_rules, str) else (
      getattr(project_rules, "text", "") or "")
  rules = rules.strip()
  if rules:
    parts.append("[project rules - AGENTS.md]\n" + rules[:6000])
  inv = (inventory or "").strip()
  if inv:
    parts.append(inv)

  msgs = [SystemMessage(content="\n\n".join(parts))]

  # History: drop the trailing entry if it duplicates the current question so
  # the question appears exactly once at the end.
  hist = list(history or [])
  if hist and hist[-1].get("role") == "user" and hist[-1].get("content") == question:
    hist = hist[:-1]
  # 关键：先剥掉思维链，再裁剪窗口。
  # 顺序不能反——trim_history 按 token 估算裁窗口，若带着 <think> 去算，
  # 大量预算会被思维链吃掉，真正有用的历史反而被提前挤出去。
  # 整条只剩思维链（剥完为空）的消息直接丢弃。
  hist = [
    {"role": m.get("role"), "content": strip_think(m.get("content") or "")}
    for m in hist
  ]
  hist = [m for m in hist if m["content"]]
  recent, _overflow = trim_history(hist)
  for m in recent:
    if m.get("role") == "user":
      msgs.append(HumanMessage(m.get("content") or ""))
    else:
      msgs.append(AIMessage(m.get("content") or ""))

  msgs.append(HumanMessage(content=question))

  total = sum(estimate_tokens(getattr(m, "content", "") or "") for m in msgs)
  _log.info("context: built %d messages, ~%d tokens (history=%d, chunks=%d)",
            len(msgs), total, len(recent), len(chunks))
  return msgs
