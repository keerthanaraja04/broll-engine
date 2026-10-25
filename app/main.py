"""
FastAPI application factory.

Usage (dev):
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Usage (production, via Dockerfile CMD):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import CorrelationIdMiddleware, RequestLoggingMiddleware
from app.core.ratelimit import RateLimitMiddleware


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    configure_logging(level=s.LOG_LEVEL, fmt=s.LOG_FORMAT)
    logger = get_logger(__name__)
    logger.info(
        "app_startup",
        extra={"app": s.APP_NAME, "version": s.APP_VERSION, "env": s.APP_ENV},
    )

    # Create / verify DB tables (idempotent — safe to run on every start)
    try:
        from app.db.session import create_all_tables
        await create_all_tables()
        logger.info("db_ready")
    except Exception as exc:
        logger.warning("db_init_failed", extra={"error": str(exc)})

    # Ensure object-storage buckets exist (if storage endpoint is reachable)
    try:
        import socket
        endpoint = str(s.STORAGE_ENDPOINT)
        host, port = endpoint.split(":", 1) if ":" in endpoint else (endpoint, 9000)
        with socket.create_connection((host, int(port)), timeout=0.1):
            pass
        from app.adapters.storage import get_storage_client
        await get_storage_client().ensure_all_buckets()
        logger.info("storage_buckets_ready")
    except Exception as exc:
        logger.info("storage_bucket_init_skipped", extra={"reason": "storage_offline"})

    yield

    # Graceful engine disposal
    try:
        from app.db.session import dispose_engine
        await dispose_engine()
    except Exception:
        pass

    logger.info("app_shutdown")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    s = get_settings()

    application = FastAPI(
        title=s.APP_NAME,
        version=s.APP_VERSION,
        debug=s.APP_DEBUG,
        root_path=s.APP_ROOT_PATH,
        docs_url="/docs" if s.APP_ENV != "production" else None,
        redoc_url="/redoc" if s.APP_ENV != "production" else None,
        openapi_url="/openapi.json" if s.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware  (outermost = first added = last to run on response)
    # ------------------------------------------------------------------
    # 1. CORS — must be outermost so OPTIONS pre-flights are handled first
    application.add_middleware(
        CORSMiddleware,
        allow_origins=s.CORS_ALLOW_ORIGINS,
        allow_methods=s.CORS_ALLOW_METHODS,
        allow_headers=s.CORS_ALLOW_HEADERS,
        allow_credentials=s.CORS_ALLOW_CREDENTIALS,
    )
    # 2. Rate limiting — reject excess requests before any route logic
    application.add_middleware(RateLimitMiddleware)
    # 3. Request logging — log method/path/status/latency
    application.add_middleware(RequestLoggingMiddleware)
    # 4. Correlation-Id — inject / echo X-Correlation-Id
    application.add_middleware(CorrelationIdMiddleware)

    # ------------------------------------------------------------------
    # Error handlers
    # ------------------------------------------------------------------
    register_error_handlers(application)

    # ------------------------------------------------------------------
    # Routers
    # ------------------------------------------------------------------
    application.include_router(api_router)

    # ------------------------------------------------------------------
    # Frontend Web UI & Static Assets
    # ------------------------------------------------------------------
    import os
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    media_dir = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
    os.makedirs(media_dir, exist_ok=True)

    application.mount("/static", StaticFiles(directory=static_dir), name="static")
    application.mount("/media", StaticFiles(directory=media_dir), name="media")

    @application.get("/", include_in_schema=False)
    async def serve_ui():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return RedirectResponse(url="/docs")

    @application.get("/ui", include_in_schema=False)
    async def serve_ui_alias():
        index_path = os.path.join(static_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return RedirectResponse(url="/docs")

    return application


app = create_app()
