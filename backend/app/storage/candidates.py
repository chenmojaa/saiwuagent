# -*- coding: utf-8 -*-
"""待审核候选库的存取层。

**这是策略 A 的防污染边界**：联网结果永远先落这里，绝不自动写进主知识库。
只有人工 approve 之后，才由调用方走正常 ingest 链路生成 Note + chunk + 向量。

为什么单独一个模块而不是塞进 db.py：db.py 负责 schema 与迁移，这里负责
「候选」这一业务概念的去重、入库、审核状态流转。两者混在一起会让 db.py
继续膨胀，而它已经有 FTS / 迁移 / 会话三块职责了。
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone

from sqlmodel import select

from app.storage.db import PendingCandidate, get_session

_log = logging.getLogger(__name__)

# 状态取值。用字符串而非 Enum：与项目既有风格一致（Note.source_type 等也是
# 裸字符串），且 SQLite 里直接可读，便于人工排查。
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"

_WS_RE = re.compile(r"\s+")


def _norm_key(source_url: str, snippet: str) -> str:
  """去重键：url + 片段正文的规范化 hash。

  同一网页在不同轮次被搜到时内容基本一致，只留一条；否则候选库会被重复
  条目淹没，人工审核就没法用了。只按 url 去重又太粗——同一页面改版后
  内容变了应该重新提审。
  """
  text = _WS_RE.sub(" ", (snippet or "")).strip().lower()
  raw = "%s\n%s" % ((source_url or "").strip(), text)
  return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()


def add_candidates(chunks: list[dict], query: str = "",
                   session_id: str | None = None) -> list[int]:
  """把联网结果写入候选库，返回**新建**记录的 id 列表。

  已存在（norm_key 命中）的直接跳过 —— 包括已经 approve/reject 过的：
  审核结论不该被同一轮重复联网覆盖掉。想看「为什么又被搜到」应该去看
  候选的 created_at，而不是把状态重置回 pending。

  绝不抛异常：候选库是旁路审计，写失败不能影响本轮作答。
  """
  if not chunks:
    return []
  new_ids: list[int] = []
  try:
    with get_session() as s:
      keys = [
        _norm_key(c.get("source_url") or "", c.get("text") or c.get("snippet") or "")
        for c in chunks
      ]
      existing = set()
      if keys:
        rows = s.exec(select(PendingCandidate.norm_key).where(
          PendingCandidate.norm_key.in_(keys))).all()
        # SQLModel 的 exec(...).all() 在只选单列时返回标量列表
        existing = {r if isinstance(r, str) else r[0] for r in rows}

      for c, key in zip(chunks, keys):
        if key in existing:
          continue
        row = PendingCandidate(
          query=(query or "")[:500],
          source_url=(c.get("source_url") or "")[:1000],
          title=(c.get("title") or "")[:500] or None,
          snippet=(c.get("text") or c.get("snippet") or "")[:4000],
          norm_key=key,
          status=STATUS_PENDING,
          session_id=session_id,
        )
        s.add(row)
        s.flush()          # 拿到自增 id
        new_ids.append(int(row.id or 0))
        existing.add(key)
      s.commit()
  except Exception as e:
    _log.warning("candidates: 写入失败（不影响作答）: %s", e)
    return []
  if new_ids:
    _log.info("candidates: 新增 %d 条待审核（q=%r）", len(new_ids), (query or "")[:60])
  return new_ids


def list_candidates(status: str | None = STATUS_PENDING, limit: int = 100,
                    offset: int = 0) -> list[dict]:
  """按状态列候选，最新在前。status=None 表示不过滤。"""
  try:
    with get_session() as s:
      stmt = select(PendingCandidate)
      if status:
        stmt = stmt.where(PendingCandidate.status == status)
      stmt = stmt.order_by(PendingCandidate.created_at.desc()).offset(offset).limit(limit)
      rows = s.exec(stmt).all()
      return [_to_dict(r) for r in rows]
  except Exception as e:
    _log.warning("candidates: 查询失败: %s", e)
    return []


def get_candidate(cid: int) -> dict | None:
  try:
    with get_session() as s:
      row = s.get(PendingCandidate, cid)
      return _to_dict(row) if row else None
  except Exception as e:
    _log.warning("candidates: 读取 %s 失败: %s", cid, e)
    return None


def count_by_status() -> dict:
  """各状态计数，用于审核入口显示待办数。"""
  out = {STATUS_PENDING: 0, STATUS_APPROVED: 0, STATUS_REJECTED: 0}
  try:
    with get_session() as s:
      for row in s.exec(select(PendingCandidate)).all():
        st = row.status or STATUS_PENDING
        out[st] = out.get(st, 0) + 1
  except Exception as e:
    _log.warning("candidates: 计数失败: %s", e)
  return out


def mark_reviewed(cid: int, status: str, note_id: str | None = None,
                  review_note: str | None = None) -> bool:
  """把候选标记为 approved / rejected。返回是否命中记录。

  注意：**这里不写主知识库**。approve 的实际入库由调用方（api 层）在
  调用本函数前完成，这样「入库」和「改状态」两件事可以分别重试：
  先入库成功再改状态，最坏情况是知识进了库但候选仍显示 pending（可人工再点
  一次，ingest 自身幂等），而不会出现「状态已批准但知识没进去」的静默丢失。
  """
  if status not in (STATUS_APPROVED, STATUS_REJECTED):
    raise ValueError("status 必须是 approved 或 rejected")
  try:
    with get_session() as s:
      row = s.get(PendingCandidate, cid)
      if row is None:
        return False
      row.status = status
      row.note_id = note_id
      row.review_note = review_note
      row.reviewed_at = datetime.now(timezone.utc)
      s.add(row)
      s.commit()
      return True
  except Exception as e:
    _log.warning("candidates: 标记 %s 为 %s 失败: %s", cid, status, e)
    return False


def _to_dict(row: PendingCandidate) -> dict:
  return {
    "id": row.id,
    "query": row.query,
    "source_url": row.source_url,
    "title": row.title,
    "snippet": row.snippet,
    "status": row.status,
    "session_id": row.session_id,
    "note_id": row.note_id,
    "review_note": row.review_note,
    "created_at": row.created_at.isoformat() if row.created_at else None,
    "reviewed_at": row.reviewed_at.isoformat() if row.reviewed_at else None,
  }
