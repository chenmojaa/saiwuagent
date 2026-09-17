"""Ingestion orchestrator."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from sqlmodel import Session as SqlSession

from app.config import settings
from app.storage.db import Note, get_engine
from app.storage.vector import add_chunks
from app.embeddings.factory import embed_texts
from app.tools.fetch_url import fetch_url
from app.tools.parse_pdf import parse_pdf
from app.tools.chunk import chunk_text
def _resolve_title(parsed_title: str, original_name: str | None, path: str) -> str:
  """Prefer the caller's original filename; fallback to parser's title or path basename.

  Avoids leaking tempfile names like "tmpabc123" when caller has the real filename.
  """
  if original_name:
    base = os.path.splitext(original_name)[0].strip()
    if base:
      return base[:200]
  if parsed_title and not parsed_title.startswith("Untitled"):
    # parser may have used path basename already; trust it
    return parsed_title[:200]
  return parsed_title[:200] if parsed_title else os.path.splitext(os.path.basename(path))[0]


def _new_id() -> str:
  return "n_" + uuid4().hex[:12]


def _save_note(note: Note) -> Note:
  engine = get_engine()
  with SqlSession(engine) as s:
    s.add(note)
    s.commit()
    s.refresh(note)
  return note


def _ingest(
  title: str,
  content: str,
  source_type: str,
  source_url: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
) -> Note:
  note_id = _new_id()
  os.makedirs(settings.notes_dir, exist_ok=True)
  content_path = os.path.join(settings.notes_dir, note_id + ".md")
  with open(content_path, "w", encoding="utf-8") as f:
    f.write(content)

  note = Note(
    id=note_id,
    title=title[:500],
    source_type=source_type,
    source_url=source_url,
    content_path=content_path,
    word_count=len(content),
    created_at=datetime.now(timezone.utc),
  )
  note = _save_note(note)

  chunks = chunk_text(content)
  if chunks:
    try:
      embeddings = embed_texts(
        chunks,
        api_key=api_key,
        base_url=base_url,
        model=embedding_model,
      )
      n = add_chunks(note_id, chunks, embeddings)
      note.chunk_count = n
      note.embedded = True
      note.summary = chunks[0][:200]
      note = _save_note(note)
    except Exception as e:
      # Embedding 失败时仍然保留笔记（embedded=False），不阻塞入库；
      # 上层路由将返回 200 + note，前端会显示"未 embedding"标签，用户可后续重试。
      note.summary = f"[embedding failed] {type(e).__name__}: {e}"
      note.embedded = False
      note = _save_note(note)
      return note

  return note


def ingest_url(
  url: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
) -> Note:
  fetched = fetch_url(url)
  return _ingest(
    title=fetched["title"],
    content=fetched["content"],
    source_type="url",
    source_url=url,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_text(
  text: str,
  title: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
) -> Note:
  t = (title or "Note " + datetime.now().strftime("%Y-%m-%d %H:%M"))[:200]
  return _ingest(
    title=t,
    content=text,
    source_type="text",
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_web_candidate(
  title: str | None,
  content: str,
  source_url: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
) -> Note:
  """把**人工已审核通过**的联网候选写入主知识库。

  这是候选库进入主库的唯一通道（策略 A 的防污染边界）：
  联网结果先落 pending_candidates，人工 approve 后才调到这里。

  与 ingest_url 的区别：不重新抓网页，直接用审核时看到的那段原文入库。
  重新抓会引入两个问题 —— ① 内容可能已经变了，人工批的是 A 入库的是 B，
  审核就失去意义；② 抓取可能失败/被反爬拦截，让「批准」这个动作变得不可靠。
  """
  t = (title or source_url or "联网候选")[:200]
  return _ingest(
    title=t,
    content=content,
    source_type="url",
    source_url=source_url,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_image(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  lang: str = "chi_sim+eng",
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.ocr import ocr_image
  parsed = ocr_image(path, lang=lang)
  fallback_name = original_name or os.path.basename(path)
  return _ingest(
    title=parsed.get("title") or os.path.splitext(fallback_name)[0] or "Untitled Image",
    content=parsed["text"],
    source_type="image",
    source_url=None,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_docx(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.parse_doc import parse_docx
  parsed = parse_docx(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="docx",
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_plain(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  """Read .txt / .md / .markdown as UTF-8 (fallback GBK/Latin-1).

  Title resolution:
    1) first non-empty line if it looks like a heading (short, no markdown)
    2) original_name without extension
    3) path basename without extension
  """
  for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
    try:
      content = open(path, "r", encoding=enc).read()
      break
    except UnicodeDecodeError:
      continue
  else:
    content = open(path, "rb").read().decode("utf-8", errors="ignore")

  # Try first non-empty line as title (only if short + looks like a heading)
  first_line = next((l.strip() for l in content.splitlines() if l.strip()), "")
  title = ""
  if first_line and len(first_line) <= 80 and not first_line.startswith(("#", "---")):
    title = first_line[:80]

  if not title:
    if original_name:
      title = os.path.splitext(original_name)[0] or ""
    if not title:
      title = os.path.splitext(os.path.basename(path))[0] or "Untitled"

  return _ingest(
    title=title[:200],
    content=content,
    source_type="text",
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_pptx(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.parse_pptx import parse_pptx
  parsed = parse_pptx(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="pptx",
    source_url=None,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_xlsx(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.parse_xlsx import parse_xlsx
  parsed = parse_xlsx(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="xlsx",
    source_url=None,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_html(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.parse_html import parse_html
  parsed = parse_html(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="html",
    source_url=None,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_csv(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  from app.tools.parse_csv import parse_csv
  parsed = parse_csv(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="csv",
    source_url=None,
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )


def ingest_file(
  path: str,
  original_name: str | None = None,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
) -> Note:
  """Dispatch by file extension."""
  name = (original_name or os.path.basename(path)).lower()
  if name.endswith(".pdf"):
    return ingest_pdf(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith(".docx"):
    return ingest_docx(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith(".pptx"):
    return ingest_pptx(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith(".xlsx"):
    return ingest_xlsx(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith((".html", ".htm")):
    return ingest_html(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith(".csv"):
    return ingest_csv(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
    return ingest_image(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  if name.endswith((".txt", ".md", ".markdown", ".rst")):
    return ingest_plain(path, api_key=api_key, base_url=base_url, embedding_model=embedding_model, original_name=original_name)
  raise ValueError(f"Unsupported file type: {name}")


def ingest_pdf(
  path: str,
  api_key: str | None = None,
  base_url: str | None = None,
  embedding_model: str | None = None,
  original_name: str | None = None,
) -> Note:
  parsed = parse_pdf(path)
  return _ingest(
    title=_resolve_title(parsed["title"], original_name, path),
    content=parsed["content"],
    source_type="pdf",
    api_key=api_key,
    base_url=base_url,
    embedding_model=embedding_model,
  )