"""Notes REST API."""
from __future__ import annotations

import logging
from typing import Optional
from fastapi import Query,  APIRouter, HTTPException, UploadFile, File, Header, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import select
from sqlalchemy import text

from app.storage.db import Note, get_session
from app.tools.ingest import ingest_url, ingest_text, ingest_pdf, ingest_image, ingest_file, _ingest
from app.tools.reindex import replace_note_index
from app.storage.vector import collection_stats, delete_note_chunks

_log = logging.getLogger(__name__)

router = APIRouter(tags=["notes"])


class IngestURLRequest(BaseModel):
  url: str
  base_url: Optional[str] = Field(None, description="可选 embedding 接口 URL")
  embedding_model: Optional[str] = Field(None, description="可选 embedding 模型名 例如 embo-01")


class IngestTextRequest(BaseModel):
  text: str
  title: Optional[str] = None
  base_url: Optional[str] = Field(None, description="可选 embedding 接口 URL")
  embedding_model: Optional[str] = Field(None, description="可选 embedding 模型名 例如 embo-01")


def _to_dict(note: Note) -> dict:
  import re
  from app.storage.feishu_config_store import get_web_url

  view_url = None
  if note.source_type.startswith("feishu_") and note.source_url:
    web_base = get_web_url()
    if web_base:
      m = re.match(r"feishu://(\w+)/(\w+)/(\w+)", note.source_url)
      if m:
        scheme, space_id, token = m.group(1), m.group(2), m.group(3)
        web_base = web_base.rstrip("/")
        if scheme == "wiki":
          view_url = f"{web_base}/wiki/{token}"
        elif scheme == "bitable":
          view_url = f"{web_base}/base/{space_id}?table={token}"

  return {
    "id": note.id,
    "title": note.title,
    "source_type": note.source_type,
    "source_url": note.source_url,
    "content_path": note.content_path,
    "summary": note.summary,
    "tags": note.tags,
    "word_count": note.word_count,
    "chunk_count": note.chunk_count,
    "embedded": note.embedded,
    "created_at": note.created_at.isoformat() if note.created_at else None,
    "view_url": view_url,
  }


def _resolve_base_url(body_url: Optional[str], header_url: Optional[str]) -> Optional[str]:
  return (body_url or header_url or "").strip() or None


def _resolve_embedding_model(body_model: Optional[str], header_model: Optional[str]) -> Optional[str]:
  return (body_model or header_model or "").strip() or None


_ALLOWED_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".html", ".htm", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

async def _safe_read_upload(file, max_bytes: int) -> bytes:
    """Read an UploadFile in chunks, refusing payloads > max_bytes."""
    total = 0
    buf = bytearray()
    while True:
        chunk = await file.read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail="file too large (>%d bytes)" % max_bytes)
        buf.extend(chunk)
    return bytes(buf)


def _check_ext(filename: str) -> None:
    from pathlib import Path
    ext = Path(filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(status_code=415, detail="unsupported file type: " + (ext or "<none>"))


@router.post("/notes/url")
async def api_ingest_url(
  body: IngestURLRequest,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  try:
    note = ingest_url(
      body.url,
      api_key=x_api_key,
      base_url=_resolve_base_url(body.base_url, x_embedding_base_url),
      embedding_model=_resolve_embedding_model(body.embedding_model, x_embedding_model),
    )
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
  return _to_dict(note)


@router.post("/notes/text")
async def api_ingest_text(
  body: IngestTextRequest,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  try:
    note = ingest_text(
      body.text,
      body.title,
      api_key=x_api_key,
      base_url=_resolve_base_url(body.base_url, x_embedding_base_url),
      embedding_model=_resolve_embedding_model(body.embedding_model, x_embedding_model),
    )
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
  return _to_dict(note)


@router.post("/notes/pdf")
async def api_ingest_pdf(
  file: UploadFile = File(...),
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  import os, tempfile
  from app.config import settings
  suffix = os.path.splitext(file.filename or "")[1] or ".pdf"
  name = file.filename or "upload.pdf"
  fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=settings.data_dir)
  os.close(fd)
  try:
    content = await file.read()
    with open(tmp_path, "wb") as f:
      f.write(content)
    note = ingest_pdf(
      tmp_path,
      api_key=x_api_key,
      base_url=_resolve_base_url(None, x_embedding_base_url),
      embedding_model=_resolve_embedding_model(None, x_embedding_model),
      original_name=name,
    )
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    try: os.unlink(tmp_path)
    except Exception: pass
  return _to_dict(note)




@router.post("/notes/image")
async def api_ingest_image(
  file: UploadFile = File(...),
  lang: str = "chi_sim+eng",
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """Upload an image png/jpg/webp/bmp/tif. OCR via Tesseract."""
  import os, tempfile
  from app.config import settings
  suffix = os.path.splitext(file.filename or "")[1] or ".png"
  name = file.filename or "upload.png"
  fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=settings.data_dir)
  os.close(fd)
  try:
    content = await file.read()
    with open(tmp_path, "wb") as f:
      f.write(content)
    note = ingest_image(
      tmp_path,
      api_key=x_api_key,
      base_url=_resolve_base_url(None, x_embedding_base_url),
      lang=lang,
      embedding_model=_resolve_embedding_model(None, x_embedding_model),
      original_name=name,
    )
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    try: os.unlink(tmp_path)
    except Exception: pass
  return _to_dict(note)


@router.post("/notes/file")
async def api_ingest_file(
  file: UploadFile = File(...),
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """Generic file upload: dispatches by extension pdf/docx/txt/md/image."""
  import os, tempfile
  from app.config import settings
  name = file.filename or "upload.bin"
  suffix = os.path.splitext(name)[1] or ""
  fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=settings.data_dir)
  os.close(fd)
  try:
    content = await file.read()
    with open(tmp_path, "wb") as f:
      f.write(content)
    note = ingest_file(
      tmp_path,
      original_name=name,
      api_key=x_api_key,
      base_url=_resolve_base_url(None, x_embedding_base_url),
      embedding_model=_resolve_embedding_model(None, x_embedding_model),
    )
  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))
  finally:
    try: os.unlink(tmp_path)
    except Exception: pass
  return _to_dict(note)

@router.get("/notes")
async def api_list_notes(limit: int = Query(50, ge=1, le=500, description="1..500"), offset: int = 0):
  with get_session() as s:
    stmt = select(Note).order_by(Note.created_at.desc()).offset(offset).limit(limit)
    notes = s.exec(stmt).all()
    total = s.exec(text("SELECT COUNT(*) FROM notes")).scalar()
  return {"items": [_to_dict(n) for n in notes], "total": total, "limit": limit, "offset": offset}


@router.get("/notes/{note_id}")
async def api_get_note(note_id: str):
  with get_session() as s:
    note = s.get(Note, note_id)
    if not note:
      raise HTTPException(status_code=404, detail="Note not found")
  return _to_dict(note)


@router.get("/notes/{note_id}/download")
async def api_download_note(note_id: str):
  import os
  from urllib.parse import quote
  with get_session() as s:
    note = s.get(Note, note_id)
    if not note:
      raise HTTPException(status_code=404, detail="Note not found")
    title = note.title or "note"
    content_path = note.content_path
  ascii_name = "".join(c for c in title if c.isalnum() or c in (" ", ".", "_", "-")).strip() or "note"
  quoted_name = quote(ascii_name, safe="-_.")
  if content_path and os.path.isfile(content_path):
    return FileResponse(
      path=content_path,
      filename=ascii_name + ".md",
      media_type="text/markdown; charset=utf-8",
      headers={"Content-Disposition": "attachment; filename=" + quoted_name + ".md"},
    )
  summary = (note.summary or "").encode("utf-8")
  data = (b"# " + title.encode("utf-8") + b"\n\n" + summary + b"\n")
  return Response(
    content=data,
    media_type="text/markdown; charset=utf-8",
    headers={"Content-Disposition": "attachment; filename=" + quoted_name + ".md"},
  )
@router.post("/notes/{note_id}/reembed")
async def api_reembed_note(
  note_id: str,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """Re-run embedding for an existing note（正文不变，只重建索引）。

  重建逻辑统一在 app/tools/reindex.py —— 这里只负责读正文、翻译结果。
  顺序（先算 embedding -> 先删旧 -> 后写新）的说明见那个模块。
  """
  import os
  with get_session() as s:
    note = s.get(Note, note_id)
    if not note:
      raise HTTPException(status_code=404, detail="Note not found")
    content_path = note.content_path

  if not content_path or not os.path.isfile(content_path):
    raise HTTPException(status_code=400, detail="Note content missing on disk")

  with open(content_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

  # write_content=False：正文没变，不必重写文件。
  res = replace_note_index(
    note_id, content,
    api_key=x_api_key,
    base_url=_resolve_base_url(None, x_embedding_base_url),
    embedding_model=_resolve_embedding_model(None, x_embedding_model),
    write_content=False,
  )
  if res.note is None:
    raise HTTPException(status_code=404, detail="Note not found")
  if res.embedding_failed:
    # 索引一个字节都没动，笔记保持原样可用 —— 报错让用户修好配置再试。
    raise HTTPException(status_code=400, detail=res.reason)
  if not res.embedded and content.strip():
    raise HTTPException(
      status_code=400,
      detail="重建索引失败：向量库写入未成功，笔记当前不可检索，请重试",
    )
  return _to_dict(res.note)


class UpdateContentRequest(BaseModel):
  content: str = Field(..., description="新的正文（Markdown）")
  title: Optional[str] = Field(None, description="可选，同时改标题")


@router.patch("/notes/{note_id}/content")
async def api_update_note_content(
  note_id: str,
  body: UpdateContentRequest,
  x_api_key: str | None = Header(None, alias="X-API-Key"),
  x_embedding_base_url: str | None = Header(None, alias="X-Embedding-Base-URL"),
  x_embedding_model: str | None = Header(None, alias="X-Embedding-Model"),
):
  """改笔记正文并立即重建索引（切分 + 向量 + FTS）。

  这是「知识库内容更新了怎么生效」的入口：改几个字、加几段，调这个接口，
  新内容马上可检索。

  为什么不用「删掉重建」：那样 note_id 会变，历史回答里的引用（[n] 指向的
  note_id）就全部失效了。这里保持 id 不变，只换正文与索引。

  失败语义：embedding 失败 -> 400 且**原索引完全不动**（笔记仍可用）；
  向量写入失败 -> 400 且 embedded=False（前端显示「未索引」，可重试）。
  两种情况都不会留下「标记已索引但实际搜不到」的孤儿笔记。
  """
  import os
  with get_session() as s:
    note = s.get(Note, note_id)
    if not note:
      raise HTTPException(status_code=404, detail="Note not found")
    content_path = note.content_path

  # 正文为空也允许（用户可能先清空再写），只是会得到一条无索引的笔记，
  # embedded=False 会如实反映这一点。
  res = replace_note_index(
    note_id,
    body.content or "",
    title=body.title,
    api_key=x_api_key,
    base_url=_resolve_base_url(None, x_embedding_base_url),
    embedding_model=_resolve_embedding_model(None, x_embedding_model),
    write_content=True,
  )
  if res.note is None:
    raise HTTPException(status_code=404, detail="Note not found")
  if res.embedding_failed:
    # keep_old_on_embed_failure=False（手动编辑路径）：算 embedding 失败就整体
    # 放弃 —— 正文和索引都保持旧状态，用户修好配置后可以重试。
    raise HTTPException(status_code=400, detail=res.reason)
  if not res.embedded and (body.content or "").strip():
    raise HTTPException(
      status_code=400,
      detail="正文已保存，但索引重建失败（向量库写入未成功），当前不可检索，请重试",
    )
  return _to_dict(res.note)


@router.delete("/notes/{note_id}")
async def api_delete_note(note_id: str):
  import os
  content_path = None
  with get_session() as s:
    note = s.get(Note, note_id)
    if not note:
      raise HTTPException(status_code=404, detail="Note not found")
    content_path = note.content_path
    s.delete(note)
    s.commit()
  chunks_deleted = delete_note_chunks(note_id)
  if content_path and os.path.isfile(content_path):
    try:
      os.remove(content_path)
    except OSError:
      pass
  gone = bool(content_path and not os.path.exists(content_path))
  return {"deleted": note_id, "chunks_deleted": chunks_deleted, "file_removed": gone}
@router.get("/notes-stats")
async def api_stats():
  with get_session() as s:
    total = s.exec(text("SELECT COUNT(*) FROM notes")).scalar()
    embedded = len(s.exec(select(Note).where(Note.embedded == True)).all())  # noqa: E712
  return {
    "sqlite": {"total_notes": total, "embedded_notes": embedded},
    "chroma": collection_stats(),
  }