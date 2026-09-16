"""OCR via pytesseract (requires Tesseract binary on PATH / install path)."""
from __future__ import annotations

import os
import logging

log = logging.getLogger(__name__)


def _find_tesseract():
  """Try to locate the tesseract binary on common Windows / Unix paths."""
  candidates = [
    os.environ.get("TESSERACT_CMD"),
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    r"C:\Users\Administrator\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    "/usr/local/bin/tesseract",
    "/usr/bin/tesseract",
    "tesseract",
  ]
  for p in candidates:
    if not p:
      continue
    if os.path.isabs(p) and os.path.exists(p):
      return p
    if not os.path.isabs(p):
      # bare name: assume in PATH
      from shutil import which
      if which(p):
        return p
  return None


def ocr_image(path: str, lang: str = "chi_sim+eng") -> dict:
  """Run OCR on an image file. Returns {"text": str, "engine": str}."""
  import pytesseract
  from PIL import Image
  # Cap pixel dimension before opening to avoid decompression bomb DoS.
  Image.MAX_IMAGE_PIXELS = 16_384 * 16_384

  tcmd = _find_tesseract()
  if tcmd is None:
    raise RuntimeError(
      "Tesseract OCR 未安装。请到 https://github.com/UB-Mannheim/tesseract/wiki "
      "下载 Windows 安装包并安装；或者把 tesseract 路径加到 PATH。"
    )

  if tcmd and os.path.isabs(tcmd):
    pytesseract.pytesseract.tesseract_cmd = tcmd

  img = Image.open(path)
  # 支持中文简体 + 英文，如未安装中文包会在运行时抛错
  text = pytesseract.image_to_string(img, lang=lang)
  return {"text": text.strip(), "engine": "tesseract", "lang": lang}

def ocr_status() -> dict:
  tess = _find_tesseract()
  langs = []
  chi_sim_ok = False
  if tess:
    from pathlib import Path
    td = Path(tess).parent / "tessdata"
    if td.exists():
      for p in td.glob("*.traineddata"):
        langs.append(p.stem)
        if p.stem == "chi_sim":
          chi_sim_ok = True
  return {
    "tesseract_path": tess,
    "available": tess is not None,
    "languages": sorted(langs),
    "chi_sim_installed": chi_sim_ok,
    "install_hint": (
      "1) Download Tesseract Windows installer from "
      "https://github.com/UB-Mannheim/tesseract/wiki ; "
      "2) During install, check 'Additional language data -> Chinese (Simplified)'; "
      "3) Add the Tesseract install directory to PATH (or set TESSERACT_CMD env var); "
      "4) Restart the backend."
    ),
  }
