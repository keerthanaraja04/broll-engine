"""
Health and readiness endpoints.

GET /health  — lightweight liveness probe (no I/O, always fast)
GET /ready   — readiness probe: checks PostgreSQL, Redis, and MinIO/S3 storage
"""

from __future__ import annotations

import time
from typing import Any

import redis.asyncio as aioredis
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["observability"])

_START_TIME = time.time()


# ---------------------------------------------------------------------------
# /health  — liveness
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    summary="Liveness probe",
    response_description="Service is alive",
    status_code=status.HTTP_200_OK,
)
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": get_settings().APP_NAME,
        "version": get_settings().APP_VERSION,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


# ---------------------------------------------------------------------------
# /ready  — readiness
# ---------------------------------------------------------------------------
async def _check_postgres() -> dict[str, Any]:
    settings = get_settings()
    try:
        engine = create_async_engine(settings.database_url, pool_size=1, max_overflow=0)
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        await engine.dispose()
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("readiness_check_postgres_failed", extra={"error": str(exc)})
        return {"status": "error", "error": str(exc)}


async def _check_redis() -> dict[str, Any]:
    settings = get_settings()
    try:
        client = aioredis.from_url(settings.redis_url, socket_timeout=2.0)
        await client.ping()
        await client.aclose()
        return {"status": "ok"}
    except Exception as exc:
        logger.warning("readiness_check_redis_failed", extra={"error": str(exc)})
        return {"status": "error", "error": str(exc)}


async def _check_storage() -> dict[str, Any]:
    settings = get_settings()
    try:
        import aioboto3  # type: ignore[import]

        session = aioboto3.Session()
        async with session.client(
            "s3",
            endpoint_url=f"http{'s' if settings.STORAGE_USE_SSL else ''}://{settings.STORAGE_ENDPOINT}",
            aws_access_key_id=settings.STORAGE_ACCESS_KEY,
            aws_secret_access_key=settings.STORAGE_SECRET_KEY,
            region_name=settings.STORAGE_REGION,
        ) as s3:
            await s3.list_buckets()
        return {"status": "ok"}
    except ImportError:
        # aioboto3 optional — fall back to sync minio client check
        try:
            from minio import Minio  # type: ignore[import]

            client = Minio(
                settings.STORAGE_ENDPOINT,
                access_key=settings.STORAGE_ACCESS_KEY,
                secret_key=settings.STORAGE_SECRET_KEY,
                secure=settings.STORAGE_USE_SSL,
            )
            client.list_buckets()
            return {"status": "ok"}
        except Exception as exc:
            logger.warning("readiness_check_storage_failed", extra={"error": str(exc)})
            return {"status": "error", "error": str(exc)}
    except Exception as exc:
        logger.warning("readiness_check_storage_failed", extra={"error": str(exc)})
        return {"status": "error", "error": str(exc)}


@router.get(
    "/ready",
    summary="Readiness probe",
    response_description="Dependency health summary",
)
async def ready() -> JSONResponse:
    postgres = await _check_postgres()
    redis_result = await _check_redis()
    storage = await _check_storage()

    checks = {
        "postgres": postgres,
        "redis": redis_result,
        "storage": storage,
    }

    all_ok = all(v["status"] == "ok" for v in checks.values())
    http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=http_status,
        content={
            "status": "ok" if all_ok else "degraded",
            "checks": checks,
        },
    )
