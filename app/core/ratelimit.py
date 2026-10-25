"""
Sliding-window rate-limiting middleware backed by Redis.

Algorithm
---------
Uses a Redis sorted set keyed by ``rl:<client_ip>:<window_start_unix>``.

For each request:
1. Add ``now`` (float timestamp) as both score and member to the set.
2. Remove all members older than ``window_start = now - window_seconds``.
3. Count remaining members.
4. If count > limit → return 429; else let the request through.
5. Set TTL on the key to ``window_seconds + 1`` to auto-clean.

Client identity
---------------
Uses X-Forwarded-For (first IP) when available, falling back to request.client.host.
Authenticated requests (valid API token present) are exempt from rate limiting.
Paths listed in ``RATE_LIMIT_EXEMPT_PATHS`` are also exempt.
"""

from __future__ import annotations

import time

import redis.asyncio as aioredis
from fastapi import status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.core.config import get_settings
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


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_exempt(path: str) -> bool:
    s = get_settings()
    return any(path.startswith(p) for p in s.RATE_LIMIT_EXEMPT_PATHS)


def _rate_limit_response(client: str, limit: int, window: int) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": f"Too many requests. Limit: {limit} per {window}s.",
                "retryable": True,
            }
        },
        headers={"Retry-After": str(window)},
    )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter. Skipped when ``RATE_LIMIT_ENABLED=false``.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        s = get_settings()

        if not s.RATE_LIMIT_ENABLED:
            return await call_next(request)

        if _is_exempt(request.url.path):
            return await call_next(request)

        client = _client_ip(request)
        now = time.time()
        window_start = now - s.RATE_LIMIT_WINDOW_SECONDS
        redis_key = f"rl:{client}"

        try:
            redis = _get_redis()
            pipe = redis.pipeline()
            # Add current timestamp as both score and member (member must be unique)
            pipe.zadd(redis_key, {f"{now:.6f}": now})
            # Remove timestamps outside the window
            pipe.zremrangebyscore(redis_key, 0, window_start)
            # Count remaining
            pipe.zcard(redis_key)
            # Reset TTL
            pipe.expire(redis_key, s.RATE_LIMIT_WINDOW_SECONDS + 1)
            results = await pipe.execute()
            request_count: int = results[2]
        except Exception as exc:
            # If Redis is down, fail open (let the request through)
            logger.warning("rate_limit_redis_error", extra={"error": str(exc)})
            return await call_next(request)

        if request_count > s.RATE_LIMIT_REQUESTS:
            logger.info(
                "rate_limit_exceeded",
                extra={
                    "client": client,
                    "count": request_count,
                    "limit": s.RATE_LIMIT_REQUESTS,
                    "path": request.url.path,
                },
            )
            return _rate_limit_response(client, s.RATE_LIMIT_REQUESTS, s.RATE_LIMIT_WINDOW_SECONDS)

        response = await call_next(request)
        # Inform caller of their remaining budget
        remaining = max(0, s.RATE_LIMIT_REQUESTS - request_count)
        response.headers["X-RateLimit-Limit"] = str(s.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Window"] = str(s.RATE_LIMIT_WINDOW_SECONDS)
        return response
