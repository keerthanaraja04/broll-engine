"""
Async token-bucket rate limiter for adapter calls.

Algorithm
---------
A token bucket holds up to ``capacity`` tokens.  Tokens refill at
``refill_rate`` tokens per second.  Each adapter call consumes one token.
If the bucket is empty, ``acquire()`` waits until a token is available.

Each adapter gets its own named bucket so their limits are independent.
Bucket state is kept in-process (not Redis) because:
  - Adapter calls from a single process share the same bucket naturally.
  - We avoid the latency/complexity of a distributed bucket.
  - Multiple worker processes each respect their own per-process limit,
    which naturally distributes load across processes.

Usage
-----
    limiter = TokenBucket("stt", capacity=20.0, refill_rate=0.5)
    await limiter.acquire()          # blocks until a token is available
    result = await _call_api(...)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from app.core.logging import get_logger

logger = get_logger(__name__)

# Global registry: name → TokenBucket
_buckets: dict[str, "TokenBucket"] = {}


class TokenBucket:
    """
    Async token-bucket rate limiter.

    Parameters
    ----------
    name:
        Identifier used for logging and the global registry.
    capacity:
        Maximum number of tokens (burst size).
    refill_rate:
        Tokens added per second (sustain rate).
    """

    def __init__(self, name: str, capacity: float, refill_rate: float) -> None:
        self._name = name
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity          # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill = now

    async def acquire(self, tokens: float = 1.0) -> None:
        """
        Consume *tokens* from the bucket.  Waits asynchronously if empty.

        Args:
            tokens: Number of tokens to consume (default 1).
        """
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    logger.debug(
                        "rate_limiter_acquired",
                        extra={"bucket": self._name, "remaining": round(self._tokens, 2)},
                    )
                    return
                # Calculate wait time until enough tokens are available
                deficit = tokens - self._tokens
                wait = deficit / self._refill_rate

            logger.debug(
                "rate_limiter_waiting",
                extra={"bucket": self._name, "wait_seconds": round(wait, 3)},
            )
            await asyncio.sleep(wait)

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Non-blocking attempt to consume tokens.

        Returns ``True`` if acquired, ``False`` if the bucket doesn't have enough.
        """
        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        """Current token level (approximate — not thread-safe without lock)."""
        return self._tokens

    def reset(self) -> None:
        """Refill the bucket to capacity. Useful in tests."""
        self._tokens = self._capacity
        self._last_refill = time.monotonic()


def get_bucket(name: str, capacity: float, refill_rate: float) -> TokenBucket:
    """
    Return the named bucket, creating it on first call.

    Parameters must match on every call for the same *name* — only the first
    call's parameters take effect (subsequent calls reuse the existing bucket).
    """
    if name not in _buckets:
        _buckets[name] = TokenBucket(name, capacity, refill_rate)
    return _buckets[name]


def reset_all_buckets() -> None:
    """Reset every registered bucket to full capacity.  Use in tests only."""
    for bucket in _buckets.values():
        bucket.reset()


def clear_all_buckets() -> None:
    """Remove all registered buckets.  Use in tests only."""
    _buckets.clear()
