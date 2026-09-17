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
    # 联网兜底来的材料要显式标注：模型需要知道这段不是用户的知识库内容，
    # 回答时才能说明来源（否则会当成用户资料来引用）。
    if (c.get("source_type") or "").strip().lower() == "web":
      title = "【网络搜索结果】" + str(title)
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


def format_verify_block(status: str, note: str = "",
                        conflicts: list | None = None,
                        stale_note_ids: list | None = None) -> str:
  """把联网校验结论渲染成注入系统提示的指令块。

  与 answer.py 里 `_forced_notice` 的分工：这里管**软约束**（希望模型怎么写），
  那里管**硬保证**（无论模型怎么写，用户一定会看到的横幅）。
  conflict / kb_stale 两条走硬保证，所以这里只给补充指引，不重复贴警告。
  """
  st = (status or "").strip().lower()
  if not st or st in ("disabled", "skipped"):
    return ""

  note_line = ("\n核对说明：" + note[:200]) if note else ""

  if st == "consistent":
    return (
      "[web verification] 本轮已联网核对：知识库与联网结果**一致**。\n"
      "以知识库材料为主作答；联网结果仅用于印证，不要用它替换知识库的说法。"
      + note_line
    )

  if st == "conflict":
    return (
      "[web verification] ⚠️ 本轮检测到知识库与联网结果**存在事实性冲突**。\n"
      "冲突点：\n" + _conflict_lines(conflicts) + "\n"
      "要求：\n"
      "1. 已有一条服务端强制插入的冲突提示位于回答最前面，**不要重复它**。\n"
      "2. 正文中把冲突双方的说法分别陈述，明确标注哪句来自知识库、哪句来自联网。\n"
      "3. **不要自行裁定谁对谁错**，也不要只讲一方。若无法判断，就直说需要人工复核。"
      + note_line
    )

  if st == "kb_stale":
    stale = "、".join(str(x) for x in (stale_note_ids or [])[:10])
    return (
      "[web verification] 知识库中的部分内容已被判定**过期**，相关片段已从参考资料中剔除"
      + ("（来源：%s）" % stale if stale else "") + "。\n"
      "要求：不要依据已剔除的内容作答；以联网结果为准，并在回答中说明"
      "「原有资料已过期，以下为最新信息」。若联网结果不足以回答，就明说资料不足。"
      + note_line
    )

  if st == "web_only":
    return (
      "[web verification] 本轮知识库**没有**相关材料，参考资料全部来自联网检索。\n"
      "要求：明确告诉用户「你的知识库中没有相关资料，以下内容来自联网检索」，"
      "让用户知道这不是基于自己的资料回答的。"
      + note_line
    )

  if st == "unverified":
    return (
      "[web verification] 本轮**未能完成**联网核对（联网检索失败或核对未完成）。\n"
      "要求：正常作答，但不要声称内容已经过时效性核对。若回答涉及可能变化的"
      "数字、政策或时效信息，提醒用户自行确认。"
      + note_line
    )

  return ""


def _conflict_lines(conflicts: list | None) -> str:
  out: list[str] = []
  for i, c in enumerate((conflicts or [])[:5], start=1):
    if not isinstance(c, dict):
      continue
    out.append("- 争议点%d：%s" % (i, str(c.get("claim") or "")[:200]))
    out.append("    知识库：%s" % str(c.get("kb_says") or "")[:300])
    out.append("    联网：  %s" % str(c.get("web_says") or "")[:300])
  return "\n".join(out) if out else "- （裁决未给出具体条目）"


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
                   inventory: str = "",
                   verify_block: str = "") -> list:
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

  # 联网校验指令（策略 A）。放在参考资料之后：模型先看到材料，
  # 再看到"该怎么用这些材料"的约束，比反过来更符合阅读顺序。
  vb = (verify_block or "").strip()
  if vb:
    parts.append(vb)

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
