"""SQLite metadata store (SQLModel) + FTS5 + ChatSession/ChatMessage."""
from __future__ import annotations

import os
import json
import re
import threading
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import Field, SQLModel, create_engine, Session, select, text
from app.config import settings


class Note(SQLModel, table=True):
  __tablename__ = "notes"
  id: str = Field(primary_key=True)
  title: str
  source_type: str
  source_url: Optional[str] = None
  content_path: Optional[str] = None
  summary: Optional[str] = None
  tags: Optional[str] = None
  word_count: int = 0
  chunk_count: int = 0
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  embedded: bool = False
  source_revision: Optional[str] = None   # remote obj_edit_time / etag for incremental sync
  source_updated_at: Optional[datetime] = None   # when we last saw a remote update
  # 入库时正文（content_path 那份 .md）的 sha1。用于判断「磁盘上的正文被改过没有」——
  # 直接编辑 data/notes/<id>.md 之后，后台扫描靠它发现变化并自动重建索引。
  # 飞书那条走 source_revision，本地文件没有远端版本号，只能靠内容 hash。
  content_hash: Optional[str] = None


class ChatSession(SQLModel, table=True):
  __tablename__ = "chat_sessions"
  id: str = Field(primary_key=True)
  title: str = "新对话"
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMessage(SQLModel, table=True):
  __tablename__ = "chat_messages"
  id: Optional[int] = Field(default=None, primary_key=True)
  session_id: str = Field(index=True)
  role: str  # user | assistant | system
  content: str
  citations_json: Optional[str] = None
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserProfile(SQLModel, table=True):
  """Long-term user profile (§6.5): cross-session facts/preferences as JSON."""
  __tablename__ = "user_profiles"
  user_id: str = Field(primary_key=True)
  facts_json: str = "{}"
  updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MemoryFact(SQLModel, table=True):
  """长期记忆事实：从对话中自动抽取的用户偏好/背景/约束，跨会话召回。

  与 UserProfile（手动 JSON 画像）互补：这里是逐条事实 + 来源会话 + 时间，
  支持按与当前问题的相关性召回（字符重叠启发式，见 recall_facts）。
  """
  __tablename__ = "memory_facts"
  id: Optional[int] = Field(default=None, primary_key=True)
  content: str                      # 一条完整事实，如「用户偏好简洁的中文回答」
  norm_key: str = Field(index=True) # 规范化文本 hash，用于去重
  session_id: Optional[str] = None  # 事实来源会话
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PendingCandidate(SQLModel, table=True):
  """待审核候选库（策略 A 的防污染边界）。

  联网结果**永远先落这里**，绝不自动写进主知识库。只有人工 approve 之后，
  才由 api/candidates.py 走正常 ingest 链路生成 Note + chunk + 向量。

  与 notes 的关系：这里是「候选」，不是知识；approve 成功后用 note_id 回指
  生成的 Note，便于审计「这条知识是从哪个网页来的、谁批的」。
  """
  __tablename__ = "pending_candidates"
  id: Optional[int] = Field(default=None, primary_key=True)
  query: str                        # 触发本次联网的问题
  source_url: str = Field(index=True)
  title: Optional[str] = None
  snippet: str                      # 网页原文片段（已截断，见 web_search._MAX_SNIPPET）
  # 去重键：url + 片段内容的规范化 hash。同一网页在不同轮次被搜到时只留一条，
  # 否则候选库会被重复条目淹没，人工审核没法用。
  norm_key: str = Field(index=True)
  status: str = Field(default="pending", index=True)   # pending | approved | rejected
  session_id: Optional[str] = None
  note_id: Optional[str] = None     # approve 后生成的 Note id
  review_note: Optional[str] = None # 审核备注（驳回理由等）
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  reviewed_at: Optional[datetime] = None


class User(SQLModel, table=True):
  """Login account: phone is the account identifier, password stored as PBKDF2 hash."""
  __tablename__ = "users"
  id: Optional[int] = Field(default=None, primary_key=True)
  phone: str = Field(unique=True, index=True)
  password_salt: str
  password_hash: str
  token_version: int = Field(default=0)
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
  updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

_engine = None
# 懒加载必须加锁：get_engine() 会被线程池线程调用（parallel_plan_node ->
# hybrid_search -> fts_search）。无锁时多线程可能同时进入初始化分支，
# 各自 create_engine 并重复跑建表 / 迁移 DDL —— 轻则建出多个 engine，
# 重则 SQLite 报 table already exists / database is locked。
# 与 vector.get_collection() 保持同一写法（双重检查锁）。
_engine_lock = threading.Lock()


def get_engine():
  global _engine
  if _engine is None:
    with _engine_lock:
      if _engine is None:   # 等锁期间可能已被其他线程建好
        os.makedirs(os.path.dirname(settings.sqlite_path), exist_ok=True)
        engine = create_engine(
          f"sqlite:///{settings.sqlite_path}",
          echo=False,
          connect_args={"check_same_thread": False},
        )
        SQLModel.metadata.create_all(engine)
        _init_fts(engine)
        _migrate_notes(engine)
        _migrate_token_version(engine)
        # 全部初始化成功后才发布，避免别的线程拿到半成品 engine
        _engine = engine
  return _engine


def _init_fts(engine):
  """Create the FTS5 virtual table if it doesn't exist yet."""
  with engine.begin() as conn:
    conn.execute(text("""
      CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
        note_id UNINDEXED,
        chunk_index UNINDEXED,
        content,
        tokenize = "unicode61 remove_diacritics 2"
      )
    """))


def _migrate_notes(engine):
  """Idempotent column adds for existing Note tables.

  SQLModel.metadata.create_all only creates missing tables, not missing columns.
  Swallow 'duplicate column' / 'already exists' errors so this is safe to call
  on every startup. Re-raise on any other failure.
  """
  statements = [
    "ALTER TABLE notes ADD COLUMN source_revision TEXT",
    "ALTER TABLE notes ADD COLUMN source_updated_at DATETIME",
    "ALTER TABLE notes ADD COLUMN content_hash TEXT",
  ]
  with engine.begin() as conn:
    for stmt in statements:
      try:
        conn.execute(text(stmt))
      except Exception as e:
        msg = str(e).lower()
        if "duplicate column" not in msg and "already exists" not in msg:
          raise

def _migrate_token_version(engine):
  """Idempotent column add for users.token_version.

  The User model has carried ``token_version`` since the password-
  rotation fix, but get_engine() calls this before any SELECT hits the
  users table, so a missing function (or a missing column on an old DB)
  surfaced as 500 "no such column: users.token_version" on /api/auth/
  login. ALTER TABLE ADD COLUMN ... DEFAULT 0 is safe to re-run: the
  "duplicate column" / "already exists" path is swallowed the same way
  as _migrate_notes above.
  """
  statement = "ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0"
  with engine.begin() as conn:
    try:
      conn.execute(text(statement))
    except Exception as e:
      msg = str(e).lower()
      if "duplicate column" not in msg and "already exists" not in msg:
        raise


def get_session() -> Session:
  return Session(get_engine())


# === FTS5 helpers ===
def add_fts(note_id: str, chunk_index: int, content: str):
  with get_engine().begin() as conn:
    conn.execute(
      text("INSERT INTO chunk_fts (note_id, chunk_index, content) VALUES (:nid, :idx, :c)"),
      {"nid": note_id, "idx": chunk_index, "c": content},
    )


def delete_fts(note_id: str) -> int:
  with get_engine().begin() as conn:
    result = conn.execute(
      text("DELETE FROM chunk_fts WHERE note_id = :nid"),
      {"nid": note_id},
    )
  return result.rowcount or 0


# FTS5 operator characters we must strip from user input before it can become
# a MATCH expression. Anything outside this set is treated as ordinary text and
# gets double-quoted so FTS5 does not interpret it as syntax.
_FTS5_BAD = set('"()*:^-+')

def _fts_sanitize_phrase(raw: str) -> str:
  """Return raw with FTS5 operators replaced by spaces, then collapsed."""
  if not raw:
    return ""
  cleaned = "".join(" " if ch in _FTS5_BAD else ch for ch in raw)
  return " ".join(cleaned.split())

def _fts_tokenize(raw: str) -> list[str]:
  """Split a query into safe per-token terms for an OR-expression."""
  cleaned = _fts_sanitize_phrase(raw)
  if not cleaned:
    return []
  out: list[str] = []
  for tok in cleaned.split():
    if all(0x4E00 <= ord(c) <= 0x9FFF for c in tok):
      out.extend(tok)
    else:
      out.append(tok)
  seen: set[str] = set()
  uniq: list[str] = []
  for t in out:
    if t and t not in seen:
      seen.add(t)
      uniq.append(t)
  return uniq

def _fts_quote(term: str) -> str:
  """Quote a single term so FTS5 treats it as literal text."""
  return '"' + term.replace(chr(34), chr(34) * 2) + '"'

def fts_search(query: str, top_k: int = 5) -> list[dict]:
  """Two-pass FTS5 search that survives both ASCII and CJK queries.

  Pass 1: phrase match against the whole query (works for ASCII tokens and
  for CJK when the query spans a whole content phrase).
  Pass 2: per-token OR match -- splits CJK into single chars because
  unicode61 produces single-char tokens for CJK; ASCII tokens are kept
  whole. Falls back to LIKE if FTS5 still refuses to parse.

  Always returns at least an empty list, never raises on a bad query.
  """
  if not query or not query.strip():
    return []
  cleaned = _fts_sanitize_phrase(query.strip())
  if not cleaned:
    return []
  uniq_tokens = _fts_tokenize(query.strip())
  base_sql = (
    "SELECT note_id, chunk_index, content, bm25(chunk_fts) AS score "
    "FROM chunk_fts "
    "WHERE chunk_fts MATCH :match "
    "ORDER BY score "
    "LIMIT :k"
  )
  by_key = {}

  def _absorb(rows):
    for r in rows:
      key = (r.note_id, r.chunk_index)
      hit = {"note_id": r.note_id, "chunk_index": r.chunk_index, "text": r.content, "score": float(r.score) if r.score is not None else 0.0}
      if key in by_key:
        by_key[key]["score"] = min(by_key[key]["score"], hit["score"])
      else:
        by_key[key] = hit

  with get_engine().begin() as conn:
    try:
      rows = conn.execute(text(base_sql), {"match": _fts_quote(cleaned), "k": top_k}).all()
      _absorb(rows)
    except Exception:
      pass
    if len(by_key) < top_k and uniq_tokens:
      try:
        or_expr = " OR ".join(_fts_quote(t) for t in uniq_tokens)
        rows = conn.execute(text(base_sql), {"match": or_expr, "k": top_k}).all()
        _absorb(rows)
      except Exception:
        pass
    if not by_key:
      try:
        like_q = "%" + cleaned + "%"
        for r in conn.execute(text("SELECT note_id, chunk_index, content, 0.0 AS score FROM chunk_fts WHERE content LIKE :q LIMIT :k"), {"q": like_q, "k": top_k}).all():
          _absorb([r])
      except Exception:
        pass

  ranked = sorted(by_key.values(), key=lambda h: h["score"])[:top_k]
  return ranked



def get_note_title(note_id: str) -> str | None:
  with get_session() as s:
    n = s.get(Note, note_id)
    return n.title if n else None


def get_note_meta(note_id: str) -> dict | None:
  """Return {source_type, source_url} for a note, or None if not found."""
  with get_session() as s:
    n = s.get(Note, note_id)
    if not n:
      return None
    return {"source_type": n.source_type, "source_url": n.source_url or ""}


def update_note_revision(note_id: str, revision: str | None,
                         updated_at: datetime | None = None) -> bool:
  """Update a note's source_revision + source_updated_at. Returns True on hit."""
  with get_session() as s:
    n = s.get(Note, note_id)
    if not n:
      return False
    n.source_revision = revision
    n.source_updated_at = updated_at or datetime.now(timezone.utc)
    s.add(n)
    s.commit()
    s.refresh(n)
    return True


# === Chat Session helpers ===
def list_sessions(limit: int = 100) -> list[dict]:
  with get_session() as s:
    rows = s.exec(
      select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(limit)
    ).all()
    out = []
    for r in rows:
      # 取最后一条消息作为 preview
      last = s.exec(
        select(ChatMessage).where(ChatMessage.session_id == r.id).order_by(ChatMessage.id.desc()).limit(1)
      ).first()
      msg_count = s.exec(
        select(ChatMessage).where(ChatMessage.session_id == r.id)
      ).all()
      # Some models (e.g. DeepSeek-class) sometimes leave <think>... unclosed
      # in the streamed text, which then leaks into the sidebar preview. Strip
      # the leading <think> block so the preview reads as the actual answer.
      preview_text = _strip_think(last.content) if last else ""
      out.append({
        "id": r.id,
        "title": r.title,
        "created_at": r.created_at.isoformat(),
        "updated_at": r.updated_at.isoformat(),
        "message_count": len(msg_count),
        "preview": (preview_text[:60] + ("..." if preview_text and len(preview_text) > 60 else "")),
      })
  return out


_THINK_RE = re.compile(r"<think>[\s\S]*?(</think>|$)", re.IGNORECASE)
def _strip_think(s: str) -> str:
  """Drop a leading <think>...</think> block (paired OR unclosed trailing).

  Mirrors the frontend's stripThink so the sidebar preview agrees with what
  the user actually sees inside MessageBubble."""
  if not s:
    return ""
  return _THINK_RE.sub("", s, count=1).strip()


def get_session_with_messages(session_id: str) -> dict | None:
  with get_session() as s:
    sess = s.get(ChatSession, session_id)
    if not sess:
      return None
    msgs = s.exec(
      select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    ).all()
  return {
    "id": sess.id,
    "title": sess.title,
    "created_at": sess.created_at.isoformat(),
    "updated_at": sess.updated_at.isoformat(),
    "messages": [
      {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "citations": json.loads(m.citations_json) if m.citations_json else None,
        "created_at": m.created_at.isoformat(),
      }
      for m in msgs
    ],
  }

def get_messages(session_id: str, limit: int = 16) -> list[dict]:
  """Return the last ``limit`` messages for a session as {role, content}.

  Phase 2 server-side context source of truth (see CONTEXT_UPGRADE.md
  Phase 2.1). Ordered oldest -> newest so callers can feed the result
  straight into a LangChain message list. Excludes the 'system' role
  because the answer node injects its own system prompt.
  """
  with get_session() as s:
    rows = s.exec(
      select(ChatMessage)
      .where(ChatMessage.session_id == session_id)
      .order_by(ChatMessage.id.desc())
      .limit(limit)
    ).all()
  return [
    {"role": m.role, "content": m.content}
    for m in reversed(rows)
    if m.role in ("user", "assistant")
  ]


def create_session(title: str = "新对话") -> ChatSession:
  from uuid import uuid4
  sid = "s_" + uuid4().hex[:12]
  sess = ChatSession(id=sid, title=title[:100])
  with get_session() as s:
    s.add(sess)
    s.commit()
    s.refresh(sess)
  return sess


def delete_session(session_id: str) -> bool:
  with get_session() as s:
    sess = s.get(ChatSession, session_id)
    if not sess:
      return False
    msgs = s.exec(select(ChatMessage).where(ChatMessage.session_id == session_id)).all()
    for m in msgs:
      s.delete(m)
    s.delete(sess)
    s.commit()
  return True


def rename_session(session_id: str, title: str) -> bool:
  with get_session() as s:
    sess = s.get(ChatSession, session_id)
    if not sess:
      return False
    sess.title = title[:100]
    sess.updated_at = datetime.now(timezone.utc)
    s.add(sess)
    s.commit()
  return True


def fork_session(session_id: str, new_title: str | None = None) -> dict | None:
  """Clone an existing session + all of its messages into a brand new session.

  Returns the new session dict {id, title, created_at, updated_at, parent_id,
  source_message_count} or None if the source session does not exist. Used by
  /api/sessions/{id}/fork (Codex CLI / Claude Code "fork session" parity)."""
  from uuid import uuid4
  with get_session() as s:
    src = s.get(ChatSession, session_id)
    if not src:
      return None
    src_msgs = s.exec(
      select(ChatMessage)
      .where(ChatMessage.session_id == session_id)
      .order_by(ChatMessage.id)
    ).all()
    new_id = "s_" + uuid4().hex[:12]
    title = (new_title or ("Fork of " + (src.title or "Untitled"))).strip()[:100] or "Fork"
    new_sess = ChatSession(id=new_id, title=title)
    s.add(new_sess)
    s.flush()
    for m in src_msgs:
      s.add(ChatMessage(
        session_id=new_id,
        role=m.role,
        content=m.content,
        citations_json=m.citations_json,
      ))
    s.commit()
    s.refresh(new_sess)
    return {
      "id": new_id,
      "title": new_sess.title,
      "created_at": new_sess.created_at.isoformat() if new_sess.created_at else None,
      "updated_at": new_sess.updated_at.isoformat() if new_sess.updated_at else None,
      "parent_id": session_id,
      "source_message_count": len(src_msgs),
    }


def touch_session(session_id: str) -> None:
  with get_session() as s:
    sess = s.get(ChatSession, session_id)
    if sess:
      sess.updated_at = datetime.now(timezone.utc)
      s.add(sess)
      s.commit()


def append_message(session_id: str, role: str, content: str, citations: list | None = None) -> ChatMessage:
  from sqlmodel import Session as SqlSession
  msg = ChatMessage(
    session_id=session_id,
    role=role,
    content=content,
    citations_json=json.dumps(citations, ensure_ascii=False) if citations else None,
  )
  with get_session() as s:
    s.add(msg)
    s.commit()
    s.refresh(msg)
    # 同时 touch session.updated_at
    sess = s.get(ChatSession, session_id)
    if sess:
      sess.updated_at = datetime.now(timezone.utc)
      s.add(sess)
      s.commit()
  return msg


def update_message_citations(message_id: int, citations: list) -> None:
  with get_session() as s:
    msg = s.get(ChatMessage, message_id)
    if msg:
      msg.citations_json = json.dumps(citations, ensure_ascii=False)
      s.add(msg)
      s.commit()


# === Long-term user profile (§6.5) ===
def get_profile(user_id: str) -> dict:
  """Return the stored profile facts for a user ({} when absent)."""
  with get_session() as s:
    row = s.get(UserProfile, user_id)
    if not row:
      return {}
    try:
      return json.loads(row.facts_json)
    except Exception:
      return {}


def save_profile(user_id: str, facts: dict) -> None:
  """Upsert the profile facts for a user."""
  with get_session() as s:
    row = s.get(UserProfile, user_id)
    payload = json.dumps(facts, ensure_ascii=False)
    if row:
      row.facts_json = payload
      row.updated_at = datetime.now(timezone.utc)
      s.add(row)
    else:
      s.add(UserProfile(user_id=user_id, facts_json=payload))
    s.commit()


# === Long-term memory facts (自动抽取的跨会话事实) ===

def _norm_fact(text: str) -> str:
  """规范化事实文本：去空白/标点后取 hash，用于精确去重。"""
  import hashlib
  cleaned = re.sub(r"[\s，。、；：？！,.:;?!\"'（）()\[\]【】\-—_]", "", (text or "").lower())
  return hashlib.md5(cleaned.encode("utf-8")).hexdigest() if cleaned else ""


def save_facts(facts: list[str], session_id: str | None = None) -> int:
  """写入一批抽取事实，norm_key 去重。返回新增条数。"""
  added = 0
  with get_session() as s:
    for f in facts:
      f = (f or "").strip()
      if not f or len(f) < 4 or len(f) > 300:
        continue
      key = _norm_fact(f)
      if not key:
        continue
      exists = s.exec(select(MemoryFact).where(MemoryFact.norm_key == key)).first()
      if exists:
        # 重复出现的事实视为再次确认：刷新时间，提高召回优先级
        exists.updated_at = datetime.now(timezone.utc)
        s.add(exists)
        continue
      s.add(MemoryFact(content=f, norm_key=key, session_id=session_id))
      added += 1
    s.commit()
  return added


def list_facts(limit: int = 200) -> list[dict]:
  """列出全部事实（管理用），时间倒序。"""
  with get_session() as s:
    rows = s.exec(
      select(MemoryFact).order_by(MemoryFact.updated_at.desc()).limit(limit)
    ).all()
  return [{"id": r.id, "content": r.content, "session_id": r.session_id,
           "created_at": r.created_at.isoformat(), "updated_at": r.updated_at.isoformat()}
          for r in rows]


def delete_fact(fact_id: int) -> bool:
  with get_session() as s:
    row = s.get(MemoryFact, fact_id)
    if not row:
      return False
    s.delete(row)
    s.commit()
  return True


def update_fact(fact_id: int, content: str) -> bool:
  """Edit an existing fact in place. Re-normalises so the dedup key stays consistent."""
  content = (content or "").strip()
  if not content:
    return False
  key = _norm_fact(content)
  with get_session() as s:
    row = s.get(MemoryFact, fact_id)
    if not row:
      return False
    row.content = content
    row.norm_key = key
    row.updated_at = datetime.now(timezone.utc)
    s.add(row)
    s.commit()
  return True


def recall_facts(query: str, limit: int = 8) -> list[str]:
  """跨会话召回：按与当前问题的字符重叠启发式排序 + 时间倒序。

  事实总量小（个人知识助手场景 < 数百条），全量读入后排序比建向量索引
  更可靠（不依赖 EMBEDDING_API_KEY，且对 CJK 无分词问题）。重叠>0 的
  相关事实优先，其余名额留给最近的事实（保证基础背景常驻）。
  """
  with get_session() as s:
    rows = s.exec(
      select(MemoryFact).order_by(MemoryFact.updated_at.desc()).limit(500)
    ).all()
  if not rows:
    return []

  # 字符 bigram 重叠计分：CJK 无空格分词，bigram 是最稳的相关性近似
  def _bigrams(t: str) -> set[str]:
    t = re.sub(r"\s", "", t)
    return {t[i:i+2] for i in range(len(t) - 1)} if len(t) > 1 else {t}

  qgrams = _bigrams(query or "")
  scored = []
  for r in rows:
    overlap = len(qgrams & _bigrams(r.content)) if qgrams else 0
    scored.append((overlap, r.updated_at, r.content))
  scored.sort(key=lambda x: (-x[0], -x[1].timestamp()))

  # 相关事实优先，剩余名额给最近事实（背景兜底）
  relevant = [c for o, _, c in scored if o > 0]
  recent = [c for o, _, c in scored if o == 0]
  half = max(1, limit // 2)
  out = (relevant[:limit] + recent[:half])[:limit]
  return out
# === MCP call history (cross-session visibility) ===


class MCPCallLog(SQLModel, table=True):
  __tablename__ = "mcp_call_log"
  id: Optional[int] = Field(default=None, primary_key=True)
  session_id: Optional[str] = Field(default=None, index=True)
  server_id: str = Field(index=True)
  tool_name: str
  arguments_json: Optional[str] = None
  status: str  # ok | error | timeout | denied
  latency_ms: int = 0
  result_preview: Optional[str] = None  # first 200 chars
  error_message: Optional[str] = None
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)

_MCP_LOG_MAX_ROWS = 5000


def log_mcp_call(session_id, server_id, tool_name, arguments, status, latency_ms=0, result_preview=None, error_message=None):
  try:
    import json as _json
    args_str = _json.dumps(arguments, ensure_ascii=False)[:4000] if arguments is not None else None
    row = MCPCallLog(
      session_id=session_id, server_id=server_id, tool_name=tool_name,
      arguments_json=args_str, status=status, latency_ms=int(latency_ms),
      result_preview=(result_preview or "")[:300] or None,
      error_message=(error_message or "")[:500] or None,
    )
    with Session(get_engine()) as s:
      s.add(row); s.commit(); s.refresh(row)
    if row.id and row.id % 100 == 0:
      with get_engine().begin() as c:
        c.execute(text(
          "DELETE FROM mcp_call_log WHERE id NOT IN ("
          "  SELECT id FROM mcp_call_log ORDER BY id DESC LIMIT :cap)"
        ), {"cap": _MCP_LOG_MAX_ROWS})
    return int(row.id or 0)
  except Exception as e:
    import logging as _log
    _log.getLogger(__name__).warning("mcp_call_log insert failed: %s", e)
    return 0


def list_mcp_calls(limit=100, server_id=None, session_id=None, status=None):
  limit = min(max(limit, 1), 500)
  stmt = select(MCPCallLog).order_by(MCPCallLog.id.desc()).limit(limit)
  if server_id: stmt = stmt.where(MCPCallLog.server_id == server_id)
  if session_id: stmt = stmt.where(MCPCallLog.session_id == session_id)
  if status: stmt = stmt.where(MCPCallLog.status == status)
  with Session(get_engine()) as s:
    rows = s.exec(stmt).all()
  out = []
  for r in rows:
    out.append({
      "id": r.id,
      "session_id": r.session_id,
      "server_id": r.server_id,
      "tool_name": r.tool_name,
      "arguments": _safe_json(r.arguments_json),
      "status": r.status,
      "latency_ms": r.latency_ms,
      "result_preview": r.result_preview,
      "error_message": r.error_message,
      "created_at": r.created_at.isoformat() if r.created_at else None,
    })
  return out


def clear_mcp_calls(server_id=None):
  with get_engine().begin() as c:
    if server_id:
      r = c.execute(text("DELETE FROM mcp_call_log WHERE server_id = :s"), {"s": server_id})
    else:
      r = c.execute(text("DELETE FROM mcp_call_log"))
  return r.rowcount or 0


def _safe_json(s):
  if not s: return None
  import json as _json
  try: return _json.loads(s)
  except Exception: return None
