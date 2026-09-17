# -*- coding: utf-8 -*-
"""待审核候选库 REST API（策略 A 的防污染边界）。

流程：Agent 联网得到结果 -> 自动落 pending_candidates（status=pending）
-> 人工在这里 list 查看 -> approve 才走 ingest 写进主知识库 / reject 丢弃。

**approve 是联网内容进入主知识库的唯一通道。** 没有任何自动路径会调用它，
这是刻意的：联网内容不可信，必须有人看过原文再决定要不要变成「用户资料」。

入库顺序（重要）：**先 ingest 成功，再改状态**。反过来会出现「状态已批准
但知识没进库」的静默丢失 —— 用户以为知识已经有了，实际什么都没有。按这个
顺序最坏只是「知识进去了但候选还显示 pending」，再点一次即可（ingest 幂等），
是可发现的失败。
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.storage import candidates as store
from app.tools.ingest import ingest_web_candidate

router = APIRouter(tags=["candidates"])


class ReviewRequest(BaseModel):
  note: Optional[str] = Field(None, description="审核备注（驳回理由等）")
  base_url: Optional[str] = Field(None, description="可选 embedding 接口 URL")
  embedding_model: Optional[str] = Field(None, description="可选 embedding 模型名")


class BatchReviewRequest(BaseModel):
  ids: list[int] = Field(default_factory=list, description="要处理的候选 id 列表")
  note: Optional[str] = None
  base_url: Optional[str] = None
  embedding_model: Optional[str] = None


def _resolve_base_url(body_url: Optional[str], header_url: Optional[str]) -> Optional[str]:
  return (body_url or header_url or "").strip() or None


def _resolve_embedding_model(body_model: Optional[str], header_model: Optional[str]) -> Optional[str]:
  return (body_model or header_model or "").strip() or None


@router.get("/candidates")
def api_list_candidates(
  status: str | None = Query("pending", description="pending | approved | rejected；传 all 表示不过滤"),
  limit: int = Query(100, ge=1, le=500),
  offset: int = Query(0, ge=0),
):
  """列出候选。默认只看待审核的。"""
  st = None if (status or "").strip().lower() == "all" else status
  items = store.list_candidates(status=st, limit=limit, offset=offset)
  return {"items": items, "counts": store.count_by_status()}


@router.get("/candidates/stats")
def api_candidate_stats():
  """各状态计数，用于审核入口显示待办数。"""
  return store.count_by_status()


@router.post("/candidates/{cid}/approve")
def api_approve_candidate(
  cid: int,
  body: ReviewRequest | None = None,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """人工确认事实无误 -> 写入主知识库。"""
  body = body or ReviewRequest()
  row = store.get_candidate(cid)
  if row is None:
    raise HTTPException(status_code=404, detail="候选不存在")
  if row["status"] == store.STATUS_APPROVED:
    raise HTTPException(status_code=409, detail="该候选已批准")

  try:
    note = ingest_web_candidate(
      title=row.get("title"),
      content=row.get("snippet") or "",
      source_url=row.get("source_url") or "",
      api_key=x_api_key,
      base_url=_resolve_base_url(body.base_url, x_embedding_base_url),
      embedding_model=_resolve_embedding_model(body.embedding_model, x_embedding_model),
    )
  except Exception as e:
    # 入库失败：候选保持 pending，用户可修好 embedding 配置后重试。
    raise HTTPException(status_code=400, detail="写入知识库失败：%s" % e)

  store.mark_reviewed(cid, store.STATUS_APPROVED,
                      note_id=getattr(note, "id", None), review_note=body.note)
  return {"ok": True, "note_id": getattr(note, "id", None),
          "embedded": bool(getattr(note, "embedded", False)),
          "item": store.get_candidate(cid)}


@router.post("/candidates/{cid}/reject")
def api_reject_candidate(cid: int, body: ReviewRequest | None = None):
  """驳回：只改状态，不碰知识库。"""
  body = body or ReviewRequest()
  row = store.get_candidate(cid)
  if row is None:
    raise HTTPException(status_code=404, detail="候选不存在")
  if not store.mark_reviewed(cid, store.STATUS_REJECTED, review_note=body.note):
    raise HTTPException(status_code=404, detail="候选不存在")
  return {"ok": True, "item": store.get_candidate(cid)}


@router.post("/candidates/batch-approve")
def api_batch_approve(
  body: BatchReviewRequest,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """批量批准。逐条独立处理，单条失败不影响其它条目。"""
  return _batch(body, approve=True, api_key=x_api_key,
                base_url=_resolve_base_url(body.base_url, x_embedding_base_url),
                embedding_model=_resolve_embedding_model(body.embedding_model, x_embedding_model))


@router.post("/candidates/batch-reject")
def api_batch_reject(body: BatchReviewRequest):
  """批量驳回。"""
  return _batch(body, approve=False)


def _batch(body: BatchReviewRequest, approve: bool,
           api_key: str | None = None, base_url: str | None = None,
           embedding_model: str | None = None) -> dict:
  ok: list[int] = []
  failed: list[dict] = []
  for cid in body.ids:
    row = store.get_candidate(cid)
    if row is None:
      failed.append({"id": cid, "error": "候选不存在"})
      continue
    if not approve:
      if store.mark_reviewed(cid, store.STATUS_REJECTED, review_note=body.note):
        ok.append(cid)
      else:
        failed.append({"id": cid, "error": "标记失败"})
      continue
    if row["status"] == store.STATUS_APPROVED:
      failed.append({"id": cid, "error": "已批准"})
      continue
    try:
      note = ingest_web_candidate(
        title=row.get("title"), content=row.get("snippet") or "",
        source_url=row.get("source_url") or "", api_key=api_key,
        base_url=base_url, embedding_model=embedding_model,
      )
    except Exception as e:
      failed.append({"id": cid, "error": str(e)[:200]})
      continue
    store.mark_reviewed(cid, store.STATUS_APPROVED,
                        note_id=getattr(note, "id", None), review_note=body.note)
    ok.append(cid)
  return {"ok": ok, "failed": failed, "counts": store.count_by_status()}
