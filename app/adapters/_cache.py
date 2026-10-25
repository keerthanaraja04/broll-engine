"""
Redis-backed response cache for AI adapter calls.

Key format
----------
    <ADAPTER_CACHE_REDIS_PREFIX><adapter_name>:<sha256_of_inputs>

e.g.  adapter:stt:a3f2b1...

The cache stores JSON-serialisable Python dicts.  Callers are responsible for
serialising / deserialising their own dataclasses (see each adapter for the
``_to_cache`` / ``_from_cache`` helpers).

Usage
-----
    from app.adapters._cache import AdapterCache

    cache = AdapterCache("stt", ttl=86400)

    hit = await cache.get(audio_bytes + language.encode())
    if hit is not None:
        return _from_cache(hit)

    result = await _call_real_api(...)
    await cache.set(audio_bytes + language.encode(), _to_cache(result))
    return result
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

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


def _cache_key(prefix: str, adapter_name: str, raw: bytes) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    return f"{prefix}{adapter_name}:{digest}"


class AdapterCache:
    """
    Namespace-aware Redis cache for one adapter type.

    Parameters
    ----------
    adapter_name:
        Short identifier used in the Redis key, e.g. ``"stt"``, ``"llm"``.
    ttl:
        Time-to-live in seconds.  ``0`` = no expiry.
    enabled:
        When ``False``, ``get`` always returns ``None`` and ``set`` is a no-op.
        Wired to ``ADAPTER_CACHE_ENABLED`` and the per-adapter ``*_CACHE_ENABLED``.
    """

    def __init__(self, adapter_name: str, ttl: int = 3600, enabled: bool = True) -> None:
        self._adapter = adapter_name
        self._ttl = ttl
        self._enabled = enabled
        s = get_settings()
        self._prefix = s.ADAPTER_CACHE_REDIS_PREFIX

    def _key(self, raw: bytes) -> str:
        return _cache_key(self._prefix, self._adapter, raw)

    async def get(self, raw: bytes) -> dict[str, Any] | None:
        """Return cached value or ``None`` on miss / error / disabled."""
        if not self._enabled:
            return None
        s = get_settings()
        if not s.ADAPTER_CACHE_ENABLED:
            return None
        try:
            value = await _get_redis().get(self._key(raw))
            if value is None:
                return None
            logger.debug("adapter_cache_hit", extra={"adapter": self._adapter})
            return json.loads(value)  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("adapter_cache_get_error", extra={"adapter": self._adapter, "error": str(exc)})
            return None

    async def set(self, raw: bytes, value: dict[str, Any]) -> None:
        """Store *value* under the SHA-256 of *raw*.  Errors are swallowed."""
        if not self._enabled:
            return
        s = get_settings()
        if not s.ADAPTER_CACHE_ENABLED:
            return
        try:
            serialised = json.dumps(value, default=str)
            redis = _get_redis()
            if self._ttl > 0:
                await redis.setex(self._key(raw), self._ttl, serialised)
            else:
                await redis.set(self._key(raw), serialised)
            logger.debug("adapter_cache_set", extra={"adapter": self._adapter, "ttl": self._ttl})
        except Exception as exc:
            logger.warning("adapter_cache_set_error", extra={"adapter": self._adapter, "error": str(exc)})

    async def invalidate(self, raw: bytes) -> None:
        """Delete a cache entry.  Errors are swallowed."""
        try:
            await _get_redis().delete(self._key(raw))
        except Exception:
            pass


def make_cache_key_bytes(*parts: bytes | str) -> bytes:
    """
    Convenience helper: concatenate an ordered set of parts into a single
    bytes object suitable for use as the ``raw`` argument to ``get``/``set``.

    Strings are UTF-8 encoded; bytes are used as-is.  A ``|`` separator is
    inserted between parts so that ``("ab", "cd")`` and ``("a", "bcd")``
    produce different keys.
    """
    sep = b"\x00|\x00"
    return sep.join(p.encode() if isinstance(p, str) else p for p in parts)
