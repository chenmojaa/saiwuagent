"""OCR pass over every image-only note that has not been OCR'd yet.

A note counts as image-only when:
  - its source_type is image, OR
  - its body is shorter than 30 chars (almost certainly an OCR miss)

For each such note we re-run OCR against the original source file (when the
source URL still points back to a file under data/uploads/) and overwrite the
.md body. Stdout is line-per-step so the SSE consumer can render a progress
bar.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger("ocr_all")


def _emit(phase, **kw):
    print(json.dumps({"phase": phase, **kw}, ensure_ascii=False), flush=True)


def _should_ocr(text):
    if not text:
        return True
    return len(text.strip()) < 30


# 只有这些后缀才可能是可 OCR 的图片源。
# 注意 content_path 指向的是笔记的 .md 正文文件，绝不能拿去当图片喂给 OCR
# （否则会报 "cannot identify image file ... .md"）。
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def _read_body(note) -> str:
    """读取笔记的正文文本。

    `notes` 表没有 text / content 列 —— 正文落在 content_path 指向的 .md 文件里
    （chunk 级文本另存于 chunk_fts）。旧实现直接 `_attr(note, "text")`，永远拿到
    None，导致 _should_ocr("") 恒为 True，把每条笔记都误判成"需要 OCR"。
    """
    for name in ("text", "content", "body"):
        v = _attr(note, name)
        if v:
            return str(v)
    cp = _attr(note, "content_path")
    if cp:
        try:
            p = Path(cp)
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    return ""


def _attr(note, name, default=None):
    """Read SQLModel/dict attribute uniformly."""
    if isinstance(note, dict):
        return note.get(name, default)
    return getattr(note, name, default)


def _list_notes(_db):
    """Iterate every note. Tries list_all_notes, falls back to SQLModel query."""
    if hasattr(_db, "list_all_notes"):
        try:
            return list(_db.list_all_notes() or [])
        except Exception:
            pass
    try:
        from app.storage.db import Note  # type: ignore
        with _db.get_session() as s:
            return list(s.query(Note).all())
    except Exception:
        return []


def _update_text(_db, note_id, text):
    """Update note body. Tries update_note_text; otherwise raw SQLModel update."""
    if hasattr(_db, "update_note_text"):
        try:
            return bool(_db.update_note_text(note_id, text))
        except Exception:
            pass
    try:
        from app.storage.db import Note  # type: ignore
        from sqlmodel import select  # type: ignore
        with _db.get_session() as s:
            n = s.get(Note, note_id)
            if not n:
                return False
            n.text = text
            s.add(n)
            s.commit()
            return True
    except Exception:
        return False


def main():
    _emit("started", script="ocr_all")
    from app.storage import db as _db
    from app.config import settings

    try:
        from app.tools.ocr import ocr_image
    except Exception as e:
        _emit("done", processed=0, skipped=0, failed=0, reason="ocr not available: " + str(e)[:200])
        return 0

    notes = _list_notes(_db)
    _emit("info", total=len(notes), unit="note")

    uploads = Path(settings.data_dir) / "uploads"
    processed = skipped = failed = 0
    for note in notes:
        src = (_attr(note, "source_type") or "").lower()
        body = _read_body(note)
        nid = _attr(note, "id")
        if src != "image" and not _should_ocr(body):
            skipped += 1
            continue
        candidates = []
        s_url = _attr(note, "source_url")
        if s_url:
            candidates.append(Path(s_url))
        c_path = _attr(note, "content_path")
        if c_path:
            candidates.append(Path(c_path))
        if nid:
            candidates.append(uploads / str(nid))
        target = None
        for c in candidates:
            try:
                # 必须同时满足「存在」+「是图片后缀」：content_path 指向的 .md
                # 正文文件也 is_file()，但拿它去 OCR 只会报 cannot identify image。
                if c.is_file() and c.suffix.lower() in _IMAGE_EXTS:
                    target = c
                    break
            except Exception:
                continue
        if target is None:
            _emit("skip", note_id=nid, reason="no source file")
            skipped += 1
            continue
        try:
            result = ocr_image(str(target), lang="chi_sim+eng")
            new_text = (result or {}).get("text") or ""
        except Exception as e:
            _emit("error", note_id=nid, detail=str(e)[:200])
            failed += 1
            continue
        if _update_text(_db, nid, new_text):
            processed += 1
            _emit("progress", done=processed, total=len(notes), note_id=nid)
        else:
            failed += 1
            _emit("error", note_id=nid, detail="update_note_text returned False")
    _emit("done", processed=processed, skipped=skipped, failed=failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
