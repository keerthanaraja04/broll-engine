"""
ClipStore — Redis-backed catalogue of every clip that has been downloaded and
stored in MinIO.

Purpose
-------
When Pixabay is unreachable (tier-1 and tier-2 both fail), the FootageRouter
falls back to tier-3: a previously-downloaded clip whose tags overlap with the
current keyword.  The ClipStore makes that lookup fast (O(1) tag → clip_id
index in Redis) without querying MinIO.

Data layout in Redis
--------------------
CLIP_STORE_REDIS_PREFIX + clip_id   →  JSON blob (Candidate dict)
CLIP_STORE_INDEX_KEY                →  Redis Set of all clip_ids
``tagidx:<tag>``                    →  Redis Set of clip_ids having that tag

All keys carry TTL = CLIP_STORE_TTL_SECONDS (default 30 days).

The store is capped at CLIP_STORE_MAX_ENTRIES; when the cap is hit the oldest
(first-registered) entry is evicted using the index set as a FIFO queue.
"""

from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis

from app.adapters.footage_types import Candidate, FootageKind
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


def _tag_index_key(tag: str) -> str:
    s = get_settings()
    return f"{s.CLIP_STORE_REDIS_PREFIX}tagidx:{tag.strip().lower()}"


class ClipStore:
    """
    Persistent registry of downloaded clips, queryable by tags.

    All methods are async and swallow Redis errors (the store is advisory —
    a miss just causes the router to try the next tier).
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def register(self, candidate: Candidate) -> None:
        """
        Persist *candidate* metadata in the store.

        If the store is at capacity, the oldest clip is evicted first.
        """
        s = get_settings()
        redis = _get_redis()
        ttl = s.CLIP_STORE_TTL_SECONDS

        try:
            # Evict oldest if at cap
            count = await redis.scard(s.CLIP_STORE_INDEX_KEY)
            if count >= s.CLIP_STORE_MAX_ENTRIES:
                await self._evict_oldest(redis, s)

            clip_key = f"{s.CLIP_STORE_REDIS_PREFIX}{candidate.id}"
            blob = json.dumps(candidate.to_dict(), default=str)

            pipe = redis.pipeline()
            pipe.setex(clip_key, ttl, blob)
            pipe.sadd(s.CLIP_STORE_INDEX_KEY, candidate.id)
            pipe.expire(s.CLIP_STORE_INDEX_KEY, ttl)

            # Tag index
            for tag in candidate.tag_set():
                tag_key = _tag_index_key(tag)
                pipe.sadd(tag_key, candidate.id)
                pipe.expire(tag_key, ttl)

            await pipe.execute()
            logger.debug(
                "clip_store_registered",
                extra={"clip_id": candidate.id, "kind": candidate.kind.value, "tags": list(candidate.tag_set())},
            )
        except Exception as exc:
            logger.warning("clip_store_register_error", extra={"error": str(exc), "clip_id": candidate.id})

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def find_by_tags(
        self,
        tags: list[str],
        kind: FootageKind | None = None,
        limit: int = 5,
    ) -> list[Candidate]:
        """
        Return up to *limit* clips whose tags intersect with *tags*.

        Candidates are scored by the number of matching tags (most matches first).
        If *kind* is given, only clips of that kind are returned.
        Returns [] on any error.
        """
        if not tags:
            return []
        s = get_settings()
        redis = _get_redis()

        try:
            # Collect clip IDs matching any tag
            scores: dict[str, int] = {}
            for tag in tags:
                tag_key = _tag_index_key(tag.strip().lower())
                members = await redis.smembers(tag_key)
                for clip_id in members:
                    scores[clip_id] = scores.get(clip_id, 0) + 1

            if not scores:
                return []

            # Sort by match count descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            candidates: list[Candidate] = []

            for clip_id, _ in ranked:
                if len(candidates) >= limit:
                    break
                clip_key = f"{s.CLIP_STORE_REDIS_PREFIX}{clip_id}"
                blob = await redis.get(clip_key)
                if blob is None:
                    continue
                try:
                    c = Candidate.from_dict(json.loads(blob))
                    if kind is not None and c.kind != kind:
                        continue
                    candidates.append(c)
                except Exception:
                    continue

            logger.debug(
                "clip_store_find_by_tags",
                extra={"tags": tags, "matched": len(candidates)},
            )
            return candidates
        except Exception as exc:
            logger.warning("clip_store_find_error", extra={"error": str(exc)})
            return []

    async def get(self, clip_id: str) -> Candidate | None:
        """Fetch a single clip by ID. Returns None on miss or error."""
        s = get_settings()
        try:
            blob = await _get_redis().get(f"{s.CLIP_STORE_REDIS_PREFIX}{clip_id}")
            if blob is None:
                return None
            return Candidate.from_dict(json.loads(blob))
        except Exception as exc:
            logger.warning("clip_store_get_error", extra={"clip_id": clip_id, "error": str(exc)})
            return None

    async def count(self) -> int:
        """Return the number of registered clips (0 on error)."""
        try:
            s = get_settings()
            return await _get_redis().scard(s.CLIP_STORE_INDEX_KEY)  # type: ignore[return-value]
        except Exception:
            return 0

    async def all_ids(self) -> list[str]:
        """Return all clip IDs currently in the store."""
        try:
            s = get_settings()
            members = await _get_redis().smembers(s.CLIP_STORE_INDEX_KEY)
            return list(members)  # type: ignore[arg-type]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def remove(self, clip_id: str) -> None:
        """Remove a clip from the store (best-effort)."""
        s = get_settings()
        redis = _get_redis()
        try:
            clip_key = f"{s.CLIP_STORE_REDIS_PREFIX}{clip_id}"
            blob = await redis.get(clip_key)
            pipe = redis.pipeline()
            pipe.delete(clip_key)
            pipe.srem(s.CLIP_STORE_INDEX_KEY, clip_id)
            if blob:
                try:
                    c = Candidate.from_dict(json.loads(blob))
                    for tag in c.tag_set():
                        pipe.srem(_tag_index_key(tag), clip_id)
                except Exception:
                    pass
            await pipe.execute()
        except Exception as exc:
            logger.warning("clip_store_remove_error", extra={"clip_id": clip_id, "error": str(exc)})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _evict_oldest(self, redis: aioredis.Redis, s: Any) -> None:  # type: ignore[type-arg]
        """Pop one member from the index and delete it."""
        try:
            evicted = await redis.spop(s.CLIP_STORE_INDEX_KEY)
            if evicted:
                await redis.delete(f"{s.CLIP_STORE_REDIS_PREFIX}{evicted}")
                logger.info("clip_store_evicted", extra={"clip_id": evicted})
        except Exception as exc:
            logger.warning("clip_store_evict_error", extra={"error": str(exc)})


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_store: ClipStore | None = None


def get_clip_store() -> ClipStore:
    global _store
    if _store is None:
        _store = ClipStore()
    return _store


def reset_clip_store() -> None:
    global _store
    _store = None
