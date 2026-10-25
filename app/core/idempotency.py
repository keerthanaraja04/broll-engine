"""
Idempotency-Key support backed by Redis.

Protocol
--------
* Every POST that carries an ``Idempotency-Key`` header is looked up in Redis.
* If the key is **new**, we store ``{hash: <body_hash>, job_id: <id>}`` and let
  the request proceed normally.
* If the key is **seen before** with the **same body hash**, we return the
  stored ``job_id`` immediately (replay).
* If the key is **seen before** with a **different body hash**, we raise
  ``IdempotencyKeyConflictError`` (409).

Storage
-------
Redis key: ``<IDEMPOTENCY_REDIS_PREFIX><idempotency_key>``
Value:      JSON ``{"hash": str, "job_id": str}``
TTL:        ``IDEMPOTENCY_TTL_SECONDS`` (default 24 h)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import redis.asyncio as aioredis

from app.core.config import get_settings
from app.core.errors import IdempotencyKeyConflictError
from app.core.logging import get_logger

logger = get_logger(__name__)

_redis_client: aioredis.Redis | None = None  # type: ignore[type-arg]


def _get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    global _redis_client
    if _redis_client is None:
        s = get_settings()
        _redis_client = aioredis.from_url(
            s.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_timeout=s.REDIS_SOCKET_TIMEOUT_SECONDS,
        )
    return _redis_client


@dataclass
class IdempotencyResult:
    """Returned by ``check_or_store``."""
    is_replay: bool
    job_id: str | None = None   # set when is_replay is True


async def check_or_store(
    idempotency_key: str,
    body_hash: str,
    job_id: str,
) -> IdempotencyResult:
    """
    Check whether ``idempotency_key`` has been seen before.

    * **New key** → store ``{hash, job_id}`` and return ``IdempotencyResult(is_replay=False)``.
    * **Replay (same hash)** → return ``IdempotencyResult(is_replay=True, job_id=<stored>)``.
    * **Conflict (different hash)** → raise ``IdempotencyKeyConflictError``.

    Args:
        idempotency_key: The raw value from the ``Idempotency-Key`` header.
        body_hash:       SHA-256 hex digest of the canonical request body.
        job_id:          The new job_id to store if the key is fresh.
    """
    s = get_settings()
    redis_key = f"{s.IDEMPOTENCY_REDIS_PREFIX}{idempotency_key}"
    redis = _get_redis()

    existing_raw: str | None = await redis.get(redis_key)

    if existing_raw is None:
        # Brand-new key — store it
        payload = json.dumps({"hash": body_hash, "job_id": job_id})
        await redis.setex(redis_key, s.IDEMPOTENCY_TTL_SECONDS, payload)
        logger.debug("idempotency_stored", extra={"key": idempotency_key, "job_id": job_id})
        return IdempotencyResult(is_replay=False)

    existing = json.loads(existing_raw)

    if existing["hash"] == body_hash:
        logger.info(
            "idempotency_replay",
            extra={"key": idempotency_key, "original_job_id": existing["job_id"]},
        )
        return IdempotencyResult(is_replay=True, job_id=existing["job_id"])

    # Same key, different body — conflict
    logger.warning(
        "idempotency_conflict",
        extra={"key": idempotency_key, "existing_hash": existing["hash"], "new_hash": body_hash},
    )
    raise IdempotencyKeyConflictError(
        "Idempotency-Key was already used with a different request body. "
        "Use a new key for a different request.",
        details={"idempotency_key": idempotency_key},
    )


async def invalidate(idempotency_key: str) -> None:
    """Remove an idempotency entry (e.g. when job creation failed after storage)."""
    s = get_settings()
    redis_key = f"{s.IDEMPOTENCY_REDIS_PREFIX}{idempotency_key}"
    await _get_redis().delete(redis_key)
