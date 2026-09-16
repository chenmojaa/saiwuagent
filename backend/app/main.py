import logging
import os
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.api.health import router as health_router
from app.api.chat import router as chat_router
from app.api.settings import router as settings_router
from app.api.notes import router as notes_router
from app.api.search import router as search_router
from app.api.sessions import router as sessions_router
from app.api.custom_models import router as custom_models_router
from app.api.feishu import router as feishu_router
from app.api.skills import router as skills_router
from app.api.auth import router as auth_router
from app.api.mcp import router as mcp_router
from app.api.memory import router as memory_router
from app.api.project_rules import router as project_rules_router
from app.api.hooks import router as hooks_router
from app.api.agents import router as agents_router
from app.api.background import router as background_router
from app.api.permissions_rules import router as permissions_rules_router

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_root = logging.getLogger()
_root.setLevel(settings.log_level.upper())
_stream = logging.StreamHandler()
_stream.setFormatter(_fmt)
_root.addHandler(_stream)
# Rotating file handler: 10MB x 5 backups (per OPTIMIZATION.md \u00a73)
_file = RotatingFileHandler(
  LOG_DIR / "hd.log",
  maxBytes=10 * 1024 * 1024,
  backupCount=5,
  encoding="utf-8",
)
_file.setFormatter(_fmt)
_root.addHandler(_file)
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.environ.get("HD_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
_ALLOWED_ORIGINS = [
  o.strip() for o in os.environ.get("HD_ALLOWED_ORIGINS", "").split(",") if o.strip()
] or [
  "http://127.0.0.1:5174", "http://localhost:5174",
  "https://11gv92qt74799.vicp.fun",
]

@asynccontextmanager
async def lifespan(app: FastAPI):
  """应用生命周期钩子，替代已弃用的 on_event(startup) 写法。

  启动阶段：打启动日志 -> 检测 Tesseract/OCR -> 起飞书后台同步循环。
  关闭阶段：取消飞书后台循环，避免任务悬挂。
  """
  logger.info("=" * 50)
  logger.info("HEAR Agent starting up (v0.6 + LangChain + LangGraph + multi-format)")
  logger.info(f"LLM:      {settings.llm_provider}/{settings.llm_model}")
  logger.info(f"Embedding:{settings.embedding_provider}/{settings.embedding_model}")
  logger.info(f"Storage:  SQLite={settings.sqlite_path}")
  logger.info(f"Server:   http://{settings.host}:{settings.port}")
  logger.info("Supported file types: pdf, docx, pptx, xlsx, csv, html, txt/md, images(OCR)")
  try:
    from app.tools.ocr import _find_tesseract
    tess = _find_tesseract()
    if tess:
      from pathlib import Path
      td = Path(tess).parent / "tessdata"
      langs = sorted([p.stem for p in td.glob("*.traineddata")]) if td.exists() else []
      logger.info(f"OCR:      tesseract={tess}, langs={langs}")
      if "chi_sim" not in langs:
        logger.info("OCR hint: Chinese OCR needs chi_sim.traineddata in tessdata/")
    else:
      logger.info("OCR:      tesseract NOT installed (image OCR disabled)")
  except Exception as e:
    logger.info(f"OCR check failed: {e}")

  # ---- Feishu background sync ----
  # The loop always starts and re-checks the runtime config each tick, so a user
  # who fills in the Feishu settings form after boot gets syncing without a
  # restart. When not configured/enabled the tick is a cheap no-op.
  _feishu_task = None
  if settings.feishu_sync_interval_min > 0:
    import asyncio
    from app.feishu_sync import sync_all
    from app.storage import feishu_config_store as _fcs
    interval_s = max(60, settings.feishu_sync_interval_min * 60)

    async def _feishu_loop():
      logger.info(f"Feishu background sync loop started, interval={interval_s}s")
      while True:
        try:
          if _fcs.is_enabled():
            results = await asyncio.to_thread(sync_all)
            for r in results:
              logger.info(
                f"Feishu sync [{r.space_name}]: synced={r.synced} skipped={r.skipped} failed={r.failed}"
              )
        except Exception as e:
          logger.warning(f"Feishu background sync failed: {type(e).__name__}: {e}")
        await asyncio.sleep(interval_s)

    _feishu_task = asyncio.create_task(_feishu_loop())
  else:
    logger.info("Feishu sync interval=0 (manual sync only)")

  logger.info("=" * 50)
  try:
    yield
  finally:
    if _feishu_task is not None and not _feishu_task.done():
      _feishu_task.cancel()


app = FastAPI(
  title="HEAR Agent",
  description="Personal knowledge base with multi-LLM + RAG + chat history",
  version="0.6.0",
  lifespan=lifespan,
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=_ALLOWED_ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


@app.middleware("http")
async def _limit_upload_size(request: Request, call_next):
  cl = request.headers.get("content-length")
  if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_BYTES:
    return JSONResponse(
      status_code=413,
      content={"detail": "payload too large (>%d bytes)" % MAX_UPLOAD_BYTES},
    )
  return await call_next(request)


@app.middleware("http")
async def _require_auth(request: Request, call_next):
  """Protect all /api routes except /api/auth/* and /api/health.

  Accepts `Authorization: Bearer <token>` or `X-Auth-Token: <token>`.
  Sets request.state.user_id on success so endpoints can identify the user.
  """
  path = request.url.path
  # Public auth surface: anything outside this set needs a valid token.
  _PUBLIC_AUTH_PATHS = {
    "/api/auth/register",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/change-password",
    "/api/health",
  }
  if path.startswith("/api") and path not in _PUBLIC_AUTH_PATHS:
    from app.api.auth import verify_token
    token = (request.headers.get("authorization") or "").strip()
    if token.lower().startswith("bearer "):
      token = token[7:].strip()
    if not token:
      token = (request.headers.get("x-auth-token") or "").strip()
    if not token:
      return JSONResponse(status_code=401, content={"detail": "未登录或会话已过期"})
    user_id = verify_token(token)
    if not user_id:
      return JSONResponse(status_code=401, content={"detail": "未登录或会话已过期"})
    request.state.user_id = user_id
  return await call_next(request)


@app.middleware("http")
async def _mirror_api_key(request: Request, call_next):
  """Persist the per-request API key server-side (best effort).

  The frontend sends the user's key on every request via X-API-Key. Background
  jobs (Feishu auto-sync re-vectorization) have no request context, so the first
  time we see a key we also store it in llm_config_store. This makes background
  embedding work regardless of which domain the user opened the app from.
  """
  key = (request.headers.get("x-api-key") or "").strip()
  base_url = (request.headers.get("x-base-url") or "").strip()
  if key:
    request.state.api_key_override = key
    try:
      from app.storage import llm_config_store as _lcs
      if not _lcs.get_api_key():
        _lcs.update_config({"api_key": key})
    except Exception:
      pass
  if base_url:
    request.state.base_url_override = base_url
  return await call_next(request)

app.include_router(health_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(notes_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(custom_models_router, prefix="/api")
app.include_router(feishu_router, prefix="/api")
app.include_router(skills_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(memory_router, prefix="/api")
app.include_router(project_rules_router, prefix="/api")
app.include_router(hooks_router, prefix="/api")
app.include_router(agents_router, prefix="/api")
app.include_router(background_router, prefix="/api")
app.include_router(permissions_rules_router, prefix="/api")

app.include_router(mcp_router, prefix='/api')

# ---- Serve frontend static files (for 花生壳 / production) ----
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
  from fastapi.staticfiles import StaticFiles
  from fastapi.responses import FileResponse

  # Serve root-level static files (favicon.png, logo.png, etc.)
  for _f in _FRONTEND_DIST.glob("*"):
    if _f.is_file():
      _name = _f.name
      _file = _f  # capture current file in closure

      @app.get(f"/{_name}", include_in_schema=False)
      async def _serve_static_file(_file=_file):
        return FileResponse(_file)

  # Serve assets directory
  app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="assets")

  def _index_response() -> FileResponse:
    """index.html 禁止缓存：文件名带 hash 的 assets 可长缓存，但 index.html 必须每次
    重新验证，否则浏览器会一直引用构建前的旧 JS/CSS 文件名。"""
    resp = FileResponse(_FRONTEND_DIST / "index.html")
    resp.headers["Cache-Control"] = "no-cache"
    return resp

  @app.get("/{_:path}", include_in_schema=False)
  async def _spa_fallback(_: str):
    """Serve index.html for SPA client-side routing. Non-API GET requests fall here."""
    return _index_response()

  @app.get("/", include_in_schema=False)
  async def _root():
    return _index_response()

  logger.info(f"Frontend static files mounted from {_FRONTEND_DIST}")
