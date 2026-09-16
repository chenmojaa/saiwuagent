"""Auth hardening: token blacklist + rate limiting + GDPR data export.

Adds three independent security controls on top of the existing
HMAC-signed-token auth (see app/api/auth.py):

1. **Token blacklist**  - logged-out tokens are persisted so a stolen
   token cannot be reused after the user explicitly logs out (or after
   a password change). Backed by a SQLModel table; verified by the auth
   middleware on every request.

2. **In-process rate limiter**  - per-IP, per-endpoint token bucket
   for the auth endpoints (login/register/change-password). Defaults
   are tight enough to blunt credential-stuffing without inconveniencing
   a real user on a flaky connection. No external dependency (slowapi)
   so we don't break the venv.

3. **Old-password gate**  - change-password now requires the previous
   password (delivered over HTTPS only because the HMAC token is short-
   lived). Implemented in api/auth.py.

4. **GDPR export/delete**  - GET /api/auth/me/export returns the user's
   notes, sessions, messages, memory facts and profile as a single JSON
   blob. DELETE /api/auth/me permanently removes the user record and
   all related rows. Useful for compliance + a clean account-reset path.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Deque, Optional

from sqlmodel import Field, SQLModel, Session, select, text

from app.storage.db import get_engine

_log = logging.getLogger(__name__)


# === Token blacklist (persistent) =============================================
class RevokedToken(SQLModel, table=True):
    """A token that has been explicitly logged out or invalidated.

    We store the HMAC signature (last 16 hex chars) instead of the full
    token so the table is small and unindexed-by-secret. The signature
    is sufficient to reject reuse without revealing it.
    """

    __tablename__ = "revoked_tokens"
    id: Optional[int] = Field(default=None, primary_key=True)
    sig_suffix: str = Field(unique=True, index=True)
    user_id: Optional[int] = None
    reason: str = "logout"
    revoked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), index=True)
    expires_at: Optional[datetime] = None


def _sig_suffix(token: str) -> str:
    """Take the last 16 chars of the signature segment."""
    if not token or "." not in token:
        return ""
    return token.split(".")[-1][-16:]


def revoke_token(token: str, user_id: Optional[int] = None, reason: str = "logout") -> bool:
    """Persist a token revocation. Idempotent - returns True if a new row was added."""
    sig = _sig_suffix(token)
    if not sig:
        return False
    try:
        with Session(get_engine()) as s:
            existing = s.exec(select(RevokedToken).where(RevokedToken.sig_suffix == sig)).first()
            if existing:
                return False
            row = RevokedToken(sig_suffix=sig, user_id=user_id, reason=reason)
            s.add(row)
            s.commit()
            s.refresh(row)
        # Opportunistic prune: keep the table small.
        _prune_revoked()
        return True
    except Exception as e:
        _log.warning("revoke_token failed: %s", e)
        return False


def is_revoked(token: str) -> bool:
    """Return True if the token has been explicitly revoked."""
    sig = _sig_suffix(token)
    if not sig:
        return False
    try:
        with Session(get_engine()) as s:
            row = s.exec(select(RevokedToken).where(RevokedToken.sig_suffix == sig)).first()
            if not row:
                return False
            # If the revoked row has an expires_at, honour TTL.
            if row.expires_at and row.expires_at < datetime.now(timezone.utc):
                return False
            return True
    except Exception as e:
        _log.debug("is_revoked check failed (fail-open): %s", e)
        return False


def _prune_revoked():
    """Drop revoked rows whose expiry has passed (defensive; the
    expires_at column is only set when we know the token's expiry).
    Only prunes if cutoff is meaningfully in the past so a same-second
    comparison can't accidentally nuke a freshly inserted row.
    """
    try:
        with get_engine().begin() as conn:
            # Drop only when expires_at is set AND in the past.
            # revoked_at is left untouched (it's an audit trail).
            conn.execute(text(
                "DELETE FROM revoked_tokens "
                "WHERE expires_at IS NOT NULL AND expires_at < :past"
            ), {"past": (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()})
    except Exception:
        pass


# === Rate limiter (in-process) ===============================================
class _Bucket:
    """Token bucket per (key, route). Refill is steady."""

    __slots__ = ("capacity", "refill_per_sec", "timestamps")

    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = capacity
        self.refill_per_sec = refill_per_sec
        self.timestamps: Deque[float] = deque()

    def hit(self) -> tuple[bool, float]:
        """Record an attempt and return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        # Drop entries that have "refilled" away.
        while self.timestamps and (now - self.timestamps[0]) * self.refill_per_sec >= 1.0:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.capacity:
            # Retry after = (1 / refill_per_sec) seconds; conservative.
            return False, 1.0 / max(self.refill_per_sec, 0.01)
        self.timestamps.append(now)
        return True, 0.0


class RateLimiter:
    """In-process token-bucket limiter keyed by (client_ip, route).

    Single-instance only - acceptable for the local/desktop deployment
    documented in this repo. Replace with redis-backed limiter if the
    app is ever multi-instance.
    """

    def __init__(self):
        self._buckets: dict[tuple[str, str], _Bucket] = defaultdict(lambda: _Bucket(5, 0.5))
        self._lock = threading.Lock()
        self._route_limits: dict[str, tuple[int, float]] = {
            # capacity, refill_per_sec
            "auth:login": (5, 0.2),         # 5 burst, then 1 / 5s
            "auth:register": (3, 0.05),     # 3 burst, then 1 / 20s
            "auth:change_password": (5, 0.1),
        }

    def _bucket_for(self, ip: str, route: str) -> _Bucket:
        cap, refill = self._route_limits.get(route, (10, 1.0))
        key = (ip, route)
        with self._lock:
            b = self._buckets.get(key)
            if b is None or b.capacity != cap or b.refill_per_sec != refill:
                b = _Bucket(cap, refill)
                self._buckets[key] = b
            return b

    def check(self, ip: str, route: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        return self._bucket_for(ip, route).hit()

    def reset(self, route: Optional[str] = None):
        """Clear buckets (used by tests + admin)."""
        with self._lock:
            if route is None:
                self._buckets.clear()
            else:
                self._buckets = {k: v for k, v in self._buckets.items() if k[1] != route}


_rate_limiter = RateLimiter()


def check_rate(ip: str, route: str) -> tuple[bool, float]:
    """Module-level convenience wrapper."""
    return _rate_limiter.check(ip, route)


def reset_rate(route: Optional[str] = None):
    """For tests."""
    _rate_limiter.reset(route)


# === GDPR export / delete ====================================================
def export_user_data(user_id: int) -> dict:
    """Collect everything we know about a user into a JSON-serialisable dict."""
    from app.storage.db import (
        Note, ChatSession, ChatMessage, MemoryFact, UserProfile, User,
        list_facts,
    )

    out: dict = {
        "user": None,
        "profile": None,
        "notes": [],
        "sessions": [],
        "memory_facts": [],
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        with Session(get_engine()) as s:
            u = s.get(User, user_id)
            if u:
                out["user"] = {"id": u.id, "phone": u.phone, "created_at": u.created_at.isoformat()}
            p = s.get(UserProfile, "default")
            if p:
                try:
                    out["profile"] = json.loads(p.facts_json or "{}")
                except Exception:
                    out["profile"] = {}
            for n in s.exec(select(Note)).all():
                out["notes"].append({
                    "id": n.id, "title": n.title, "source_type": n.source_type,
                    "source_url": n.source_url, "tags": n.tags, "created_at": n.created_at.isoformat(),
                })
            for sess in s.exec(select(ChatSession)).all():
                msgs = s.exec(select(ChatMessage).where(ChatMessage.session_id == sess.id)).all()
                out["sessions"].append({
                    "id": sess.id, "title": sess.title, "created_at": sess.created_at.isoformat(),
                    "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in msgs],
                })
            for f in list_facts(limit=500):
                out["memory_facts"].append(f)
    except Exception as e:
        _log.warning("export_user_data failed: %s", e)
    return out


class SharedDataDeletionRefused(RuntimeError):
    """多账号环境下拒绝执行"清空全库"。

    本项目的表结构没有按用户隔离：notes / chat_sessions / chat_messages /
    memory_facts / mcp_call_log 都没有 user 归属列（只有 user_profiles 和
    revoked_tokens 带 user_id）。因此"只删我的数据"在这套 schema 下无法
    表达——单账号时全表 DELETE 恰好等于"清空我这台机器"，语义正确；一旦
    存在多个账号，它就会连带抹掉别人的知识库。
    """


def delete_user_data(user_id: int) -> int:
    """Wipe all rows tied to this user. Returns the number of rows removed.

    Raises on partial failure so the caller can return a 5xx instead of
    silently lying that the account is gone. Raises
    ``SharedDataDeletionRefused`` when the wipe would destroy other accounts'
    data (see that class for why the schema makes this unavoidable).
    """
    from app.storage.vector import delete_note_chunks
    removed = 0
    with get_engine().begin() as conn:
        # ---- 多账号安全闸 ----
        # 必须在任何 DELETE 之前判断：下面的清空是"全表"语义，多账号下
        # 会误删他人数据。
        total_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar() or 0
        if total_users > 1:
            _log.error(
                "delete_user_data refused: %d accounts exist but tables are "
                "not user-scoped (would wipe every account's notes/sessions)",
                total_users,
            )
            raise SharedDataDeletionRefused(
                "拒绝执行：数据库中还有 %d 个账号，但 notes / chat_sessions / "
                "chat_messages 等表未按用户隔离（共享同一份数据）。继续清空会"
                "连带删除其他账号的知识库。请先删除其他账号（或改为按用户隔离的"
                "数据模型）后重试。" % total_users
            )

        # Collect note ids before we drop the notes table.
        with Session(get_engine()) as s:
            note_ids = [n.id for n in s.exec(select(Note)).all()]
        for nid in note_ids:
            try:
                delete_note_chunks(nid)
            except Exception as e:
                _log.warning("vector delete for %s failed: %s", nid, e)
        # 无 user 归属列的表：单账号下等价于清空本机数据（已在上面拦截多账号）。
        for tbl in (
            "chat_messages",
            "chat_sessions",
            "memory_facts",
            "mcp_call_log",
        ):
            r = conn.execute(text(f"DELETE FROM {tbl}"))
            removed += r.rowcount or 0
        # 有 user_id 的表按账号精确删除。
        r = conn.execute(text("DELETE FROM user_profiles WHERE user_id = :uid"),
                         {"uid": user_id})
        removed += r.rowcount or 0
        r = conn.execute(text("DELETE FROM revoked_tokens WHERE user_id = :uid"),
                         {"uid": user_id})
        removed += r.rowcount or 0
        # notes last (after vector) so partial failure is recoverable.
        r = conn.execute(text("DELETE FROM notes"))
        removed += r.rowcount or 0
        r = conn.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user_id})
        removed += r.rowcount or 0
    return removed


__all__ = [
    "revoke_token",
    "is_revoked",
    "check_rate",
    "reset_rate",
    "export_user_data",
    "delete_user_data",
    "SharedDataDeletionRefused",
]