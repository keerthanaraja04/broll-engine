"""
Shared retry + 429 handling utilities for all real adapters.

Usage::

    from app.adapters._retry import with_retry, parse_retry_after

    result = await with_retry(
        fn=lambda: _call_api(...),
        max_retries=3,
        adapter_name="stt",
        retryable_exceptions=(httpx.TimeoutException, httpx.ConnectError),
    )
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Type, TypeVar

import httpx

from app.core.errors import GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def parse_retry_after(response: httpx.Response, default: float = 60.0) -> float:
    """Return the Retry-After value in seconds from an HTTP response."""
    raw = response.headers.get("retry-after", "")
    try:
        return max(1.0, float(raw))
    except (ValueError, TypeError):
        return default


async def with_retry(
    fn: Callable[[], Coroutine[Any, Any, T]],
    max_retries: int,
    adapter_name: str,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[Type[Exception], ...] = (
        httpx.TimeoutException,
        httpx.ConnectError,
        GroqServiceError,
    ),
) -> T:
    """
    Retry *fn* up to *max_retries* times with exponential backoff.

    RateLimitError is retried after waiting the Retry-After period.
    Other retryable exceptions get exponential backoff.
    Non-retryable exceptions are re-raised immediately.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return await fn()
        except RateLimitError as exc:
            retry_after = (exc.details or {}).get("retry_after_seconds", 60.0)
            logger.warning(
                "adapter_rate_limited",
                extra={"adapter": adapter_name, "attempt": attempt,
                       "retry_after": retry_after},
            )
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(float(retry_after))
            continue
        except retryable_exceptions as exc:  # type: ignore[misc]
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "adapter_retry",
                extra={"adapter": adapter_name, "attempt": attempt,
                       "max_retries": max_retries, "error": str(exc),
                       "delay": delay},
            )
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(delay)
            continue

    raise last_exc or RuntimeError(
        f"{adapter_name}: all {max_retries} retries exhausted"
    )
