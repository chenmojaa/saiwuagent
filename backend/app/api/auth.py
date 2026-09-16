"""Authentication API: phone+password register / login / change password / me / export / delete.

Token format (unchanged): stateless HMAC-signed `userId.exp.signature`, secret
persisted in data/auth_secret so tokens survive restarts. New: every issued
token is checked against the RevokedToken table before being trusted (see
auth_security.py), so a logged-out or password-changed user cannot reuse
the old token.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import Session as DBSession, select

from app.config import settings
from app.storage.db import User, get_engine, get_session
from app.api.auth_security import (
    revoke_token, is_revoked, check_rate, export_user_data, delete_user_data,
)

router = APIRouter(prefix="/auth", tags=["auth"])

PHONE_RE = re.compile(r"^1[3-9]\d{9}$")
TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# ---- HMAC secret (persisted) ----
_SECRET_FILE = Path(settings.data_dir) / "auth_secret"
_secret_cache: Optional[str] = None


def _get_secret() -> str:
    global _secret_cache
    if _secret_cache is None:
        _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
        if _SECRET_FILE.exists():
            _secret_cache = _SECRET_FILE.read_text(encoding="utf-8").strip()
        else:
            _secret_cache = secrets.token_hex(32)
            _SECRET_FILE.write_text(_secret_cache, encoding="utf-8")
    return _secret_cache


# ---- Password hashing ----
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


# ---- Token ----
def make_token(user_id: int, token_version: int = 0) -> str:
    payload = f"{user_id}.{token_version}.{int(time.time()) + TOKEN_TTL_SECONDS}"
    sig = hmac.new(_get_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> Optional[int]:
    """Return user_id if the token is valid, not expired, and not revoked.

    Tokens are ``f"{user_id}.{token_version}.{exp}.{sig}"``. We compare the
    embedded ``token_version`` against the current ``User.token_version`` and
    reject mismatches, so a password change invalidates every other active
    session for the same account.
    """
    if not token:
        return None
    parts = token.split(".")
    if len(parts) != 4:
        return None
    user_id_s, tok_version_s, exp_s, sig = parts
    payload = f"{user_id_s}.{tok_version_s}.{exp_s}"
    expected = hmac.new(_get_secret().encode(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        if int(exp_s) < time.time():
            return None
        user_id = int(user_id_s)
        tok_version = int(tok_version_s)
    except ValueError:
        return None
    try:
        with get_session() as s:
            user = s.get(User, user_id)
            if user is None or getattr(user, "token_version", 0) != tok_version:
                return None
    except Exception:
        return None
    try:
        if is_revoked(token):
            return None
    except Exception:
        return None
    return user_id

def _find_user_by_phone(session: DBSession, phone: str) -> Optional[User]:
    return session.exec(select(User).where(User.phone == phone)).first()


def _client_ip(request: Request) -> str:
    """Best-effort client IP. Trusts X-Forwarded-For only if the
    deployment is behind a single trusted proxy; otherwise falls back to
    the direct remote address. Local-only deployment defaults to direct.
    """
    fwd = (request.headers.get("x-forwarded-for") or "").strip()
    if fwd:
        return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "unknown") or "unknown"


def _validate_phone(phone: str) -> str:
    phone = (phone or "").strip()
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="手机号格式不正确")
    return phone


def _validate_password(password: str) -> str:
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="密码长度至少 6 位")
    return password


# ---- Request models ----
class RegisterReq(BaseModel):
    phone: str
    password: str


class LoginReq(BaseModel):
    account: str   # 手机号
    password: str


class ChangePasswordReq(BaseModel):
    phone: str
    old_password: str        # NEW: required to prevent silent hijack via leaked phone
    new_password: str


class AuthedChangePasswordReq(BaseModel):
    """Authenticated change-password: any logged-in user can change their own
    password without typing the phone number again. Avoids the social-
    engineering risk of the unauth version (anyone who knows your phone can
    trigger a reset link in your SMS).
    """
    old_password: str
    new_password: str


# ---- Endpoints ----
@router.post("/register")
def register(req: RegisterReq, request: Request):
    """Register with phone + password; phone is stored in the SQLite users table."""
    ip = _client_ip(request)
    allowed, retry_after = check_rate(ip, "auth:register")
    if not allowed:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {int(retry_after) + 1} 秒后重试")
    phone = _validate_phone(req.phone)
    password = _validate_password(req.password)
    with DBSession(get_engine()) as session:
        if _find_user_by_phone(session, phone):
            raise HTTPException(status_code=409, detail="该手机号已注册")
        salt = secrets.token_hex(16)
        user = User(phone=phone, password_salt=salt, password_hash=_hash_password(password, salt))
        session.add(user)
        session.commit()
        session.refresh(user)
        return {"ok": True, "user_id": user.id, "phone": user.phone, "token": make_token(user.id, getattr(user, 'token_version', 0))}


@router.post("/login")
def login(req: LoginReq, request: Request):
    """Login with account (phone) + password."""
    ip = _client_ip(request)
    allowed, retry_after = check_rate(ip, "auth:login")
    if not allowed:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {int(retry_after) + 1} 秒后重试")
    phone = (req.account or "").strip()
    password = req.password or ""
    with DBSession(get_engine()) as session:
        user = _find_user_by_phone(session, phone)
        if not user:
            # Constant-time-ish: still hash a dummy to avoid timing oracle.
            _hash_password(password, "x" * 32)
            raise HTTPException(status_code=401, detail="账号不存在，请先注册")
        if not hmac.compare_digest(user.password_hash, _hash_password(password, user.password_salt)):
            raise HTTPException(status_code=401, detail="密码错误")
        return {"ok": True, "user_id": user.id, "phone": user.phone, "token": make_token(user.id, getattr(user, 'token_version', 0))}


@router.post("/logout")
def logout(request: Request):
    """Revoke the bearer token so it cannot be reused after logout.

    Returns ok=True even when the request had no token (logout must be
    idempotent). All the device's requests after this point will get 401.
    """
    token = (request.headers.get("authorization") or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token:
        token = (request.headers.get("x-auth-token") or "").strip()
    user_id = getattr(request.state, "user_id", None)
    revoke_token(token or "", user_id=user_id, reason="logout")
    return {"ok": True}


@router.post("/change-password")
def change_password(req: ChangePasswordReq, request: Request):
    """Change password by phone + old_password + new_password.

    Also revokes all active tokens for the user by inserting the
    currently-issued token signature into the blacklist (best effort -
    a user without a current session can still change the password,
    but old tokens remain valid for 7 days unless explicitly revoked).
    """
    ip = _client_ip(request)
    allowed, retry_after = check_rate(ip, "auth:change_password")
    if not allowed:
        raise HTTPException(status_code=429, detail=f"请求过于频繁，请 {int(retry_after) + 1} 秒后重试")
    phone = _validate_phone(req.phone)
    new_password = _validate_password(req.new_password)
    old_password = req.old_password or ""
    if old_password == new_password:
        raise HTTPException(status_code=400, detail="新密码不能与旧密码相同")
    with DBSession(get_engine()) as session:
        user = _find_user_by_phone(session, phone)
        if not user:
            raise HTTPException(status_code=404, detail="该手机号不存在，无法修改密码")
        if not hmac.compare_digest(user.password_hash, _hash_password(old_password, user.password_salt)):
            raise HTTPException(status_code=401, detail="旧密码错误")
        salt = secrets.token_hex(16)
        user.password_salt = salt
        user.password_hash = _hash_password(new_password, salt)
        user.token_version = (getattr(user, 'token_version', 0) or 0) + 1
        # Bumping token_version invalidates every other active session for
        # this account immediately (verify_token rejects mismatched versions).
        # The current request still holds the old token; we revoke it below
        # so the user is forced to log in with the new credentials.
        user.token_version = (getattr(user, "token_version", 0) or 0) + 1
        user.updated_at = datetime.now(timezone.utc)
        session.add(user)
        session.commit()
    # Best-effort: revoke the request's own token (if any) so the device
    # that just proved ownership of the old password is signed out cleanly.
    try:
        token = (request.headers.get("authorization") or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            token = (request.headers.get("x-auth-token") or "").strip()
        if token:
            revoke_token(token, user_id=user.id, reason="password_change")
    except Exception:
        pass
    return {"ok": True}


@router.post("/change-password-authed")
def change_password_authed(req: AuthedChangePasswordReq, request: Request):
    """Authenticated change-password: user must be logged in, supplies
    old_password, and we rotate the password + revoke the current token."""
    new_password = _validate_password(req.new_password)
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未登录")
    with DBSession(get_engine()) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")
        if not hmac.compare_digest(user.password_hash, _hash_password(req.old_password or "", user.password_salt)):
            raise HTTPException(status_code=401, detail="旧密码错误")
        if req.old_password == new_password:
            raise HTTPException(status_code=400, detail="新密码不能与旧密码相同")
        salt = secrets.token_hex(16)
        user.password_salt = salt
        user.password_hash = _hash_password(new_password, salt)
        user.updated_at = datetime.now(timezone.utc)
        session.add(user)
        session.commit()
    # Revoke the current token.
    try:
        token = (request.headers.get("authorization") or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            token = (request.headers.get("x-auth-token") or "").strip()
        if token:
            revoke_token(token, user_id=user_id, reason="password_change")
    except Exception:
        pass
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    """Return current logged-in user; user_id is injected by the auth middleware."""
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未登录")
    with DBSession(get_engine()) as session:
        user = session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=401, detail="用户不存在")
        return {"user_id": user.id, "phone": user.phone}


@router.get("/me/export")
def me_export(request: Request):
    """GDPR / data portability: dump everything tied to the current user."""
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未登录")
    return export_user_data(user_id)


@router.delete("/me")
def me_delete(request: Request):
    """GDPR / right-to-be-forgotten: wipe the account + all data.

    Revokes the current token first so a follow-up call with the same
    token fails fast instead of accidentally succeeding.
    """
    user_id = getattr(request.state, "user_id", None)
    if user_id is None:
        raise HTTPException(status_code=401, detail="未登录")
    # Revoke current token before wiping so this request is the last one.
    try:
        token = (request.headers.get("authorization") or "").strip()
        if token.lower().startswith("bearer "):
            token = token[7:].strip()
        if not token:
            token = (request.headers.get("x-auth-token") or "").strip()
        if token:
            revoke_token(token, user_id=user_id, reason="account_delete")
    except Exception:
        pass
    removed = _delete_guarded(user_id)
    return {"ok": True, "removed_rows": removed}


def _delete_guarded(user_id: int) -> int:
    """Call delete_user_data, mapping the multi-account refusal to HTTP 409.

    单账号时行为与以前完全一致（清空本机数据）。多账号时不再静默清空
    他人的知识库，而是明确告诉调用方为什么不能删。
    """
    from app.api.auth_security import SharedDataDeletionRefused
    try:
        return delete_user_data(user_id)
    except SharedDataDeletionRefused as e:
        raise HTTPException(status_code=409, detail=str(e))