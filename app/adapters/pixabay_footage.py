"""
PixabayFootageAdapter — normalised footage search over Pixabay Video + Image APIs.

Features
--------
* Searches both /api/videos/ and /api/ (images) and returns normalised Candidate objects.
* Picks the best rendition for the configured target resolution.
* Token-bucket rate limiter: ~100 requests / 60 s (configurable via env).
* Redis search-result cache with 24 h TTL (keyed by SHA-256 of query + params).
* Retry with exponential backoff on network errors.
* Explicit handling of:
    - 429 Too Many Requests  → RateLimitError (respects Retry-After header)
    - 401 / 400 auth error   → PixabayAuthError (non-retryable)
    - 5xx / timeout          → PixabayServiceError (retryable)
    - Empty results          → returns []  (caller decides what to do)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.adapters._cache import AdapterCache, make_cache_key_bytes
from app.adapters._ratelimiter import get_bucket
from app.adapters.footage_types import Candidate, FootageKind, RenditionInfo
from app.core.config import get_settings
from app.core.errors import PixabayAuthError, PixabayServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Video quality preference order (highest first)
_VIDEO_QUALITY_ORDER = ["large", "medium", "small", "tiny"]

# Pixabay error bodies that indicate an invalid API key
_AUTH_ERROR_SUBSTRINGS = (
    "invalid api key",
    "api key invalid",
    "no api_key",
    "error: api_key",
    "api_key invalid",
    "invalid api_key",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_retry_after(response: httpx.Response) -> float:
    """Return Retry-After seconds from response header, defaulting to 60."""
    raw = response.headers.get("retry-after", "")
    try:
        return max(1.0, float(raw))
    except (ValueError, TypeError):
        return 60.0


def _is_auth_error(response: httpx.Response) -> bool:
    body = response.text.lower()
    return response.status_code in (400, 401, 403) and any(
        s in body for s in _AUTH_ERROR_SUBSTRINGS
    )


def _build_video_renditions(videos_dict: dict[str, Any]) -> list[RenditionInfo]:
    renditions: list[RenditionInfo] = []
    for label in _VIDEO_QUALITY_ORDER:
        v = videos_dict.get(label)
        if not v or not v.get("url"):
            continue
        renditions.append(
            RenditionInfo(
                url=v["url"],
                width=v.get("width", 0),
                height=v.get("height", 0),
                size_bytes=v.get("size", 0),
                label=label,
                content_type="video/mp4",
            )
        )
    return renditions


def _build_image_renditions(hit: dict[str, Any]) -> list[RenditionInfo]:
    renditions: list[RenditionInfo] = []
    # Full-size (may not always be present / accessible)
    if hit.get("imageURL"):
        renditions.append(
            RenditionInfo(
                url=hit["imageURL"],
                width=hit.get("imageWidth", 0),
                height=hit.get("imageHeight", 0),
                size_bytes=hit.get("imageSize", 0),
                label="full",
                content_type="image/jpeg",
            )
        )
    # Web format (always present, usually 1280 px wide)
    if hit.get("webformatURL"):
        renditions.append(
            RenditionInfo(
                url=hit["webformatURL"],
                width=hit.get("webformatWidth", 1280),
                height=hit.get("webformatHeight", 720),
                label="webformat",
                content_type="image/jpeg",
            )
        )
    return renditions


def _video_hit_to_candidate(hit: dict[str, Any], s: Any) -> Candidate | None:
    renditions = _build_video_renditions(hit.get("videos", {}))
    if not renditions:
        return None

    # Pick best URL for the target resolution
    best = None
    for r in renditions:
        if r.meets_minimum(s.PIXABAY_FOOTAGE_TARGET_WIDTH, s.PIXABAY_FOOTAGE_TARGET_HEIGHT):
            best = r
            break
    if best is None:
        for r in renditions:
            if r.meets_minimum(s.PIXABAY_FOOTAGE_FALLBACK_WIDTH, s.PIXABAY_FOOTAGE_FALLBACK_HEIGHT):
                best = r
                break
    if best is None:
        best = renditions[0]

    tags = [t.strip() for t in hit.get("tags", "").split(",") if t.strip()]
    user = hit.get("user", "")
    pid = hit.get("id", 0)
    page_url = hit.get("pageURL", "")

    return Candidate(
        id=f"pixabay:{pid}",
        source="pixabay",
        kind=FootageKind.VIDEO,
        url=best.url,
        thumb_url=hit.get("picture_id", ""),
        width=best.width,
        height=best.height,
        duration=float(hit.get("duration", 0)),
        license="Pixabay Content License",
        commercial_use=True,
        attribution=f"Video by {user} on Pixabay" if user else "Pixabay",
        page_url=page_url,
        tags=tags,
        is_ai_generated=False,
        renditions=renditions,
    )


def _image_hit_to_candidate(hit: dict[str, Any], s: Any) -> Candidate | None:
    renditions = _build_image_renditions(hit)
    if not renditions:
        return None

    best = renditions[0]  # full > webformat already sorted by _build_image_renditions
    tags = [t.strip() for t in hit.get("tags", "").split(",") if t.strip()]
    user = hit.get("user", "")
    pid = hit.get("id", 0)

    return Candidate(
        id=f"pixabay_img:{pid}",
        source="pixabay",
        kind=FootageKind.IMAGE,
        url=best.url,
        thumb_url=hit.get("previewURL", ""),
        width=best.width,
        height=best.height,
        duration=0.0,
        license="Pixabay Content License",
        commercial_use=True,
        attribution=f"Image by {user} on Pixabay" if user else "Pixabay",
        page_url=hit.get("pageURL", ""),
        tags=tags,
        is_ai_generated=False,
        renditions=renditions,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class PixabayFootageAdapter:
    """
    Normalised footage search over the Pixabay Video + Image APIs.

    Usage::

        adapter = PixabayFootageAdapter()
        videos = await adapter.search_videos("city skyline", per_page=10)
        images = await adapter.search_images("mountain", per_page=5)
    """

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._video_base = str(s.PIXABAY_VIDEO_BASE_URL).rstrip("/") + "/"
        self._image_base = str(s.PIXABAY_BASE_URL).rstrip("/") + "/"
        self._timeout = httpx.Timeout(s.PIXABAY_FOOTAGE_TIMEOUT_SECONDS)
        self._cache = AdapterCache(
            "footage_search",
            ttl=s.PIXABAY_FOOTAGE_SEARCH_CACHE_TTL,
            enabled=s.PIXABAY_FOOTAGE_SEARCH_CACHE_ENABLED,
        )
        self._bucket = get_bucket(
            "pixabay_footage",
            capacity=s.PIXABAY_FOOTAGE_RATE_LIMIT_TOKENS,
            refill_rate=s.PIXABAY_FOOTAGE_RATE_LIMIT_REFILL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search_videos(
        self,
        query: str,
        per_page: int | None = None,
        page: int = 1,
        video_type: str | None = None,
        min_width: int | None = None,
        min_height: int | None = None,
    ) -> list[Candidate]:
        """
        Search Pixabay for videos and return normalised Candidates.

        Returns an empty list when Pixabay returns no hits (not an error).
        Raises PixabayAuthError, PixabayServiceError, or RateLimitError on API failure.
        """
        s = self._s
        params: dict[str, Any] = {
            "key": s.PIXABAY_API_KEY,
            "q": query,
            "per_page": per_page or s.PIXABAY_RESULTS_PER_PAGE,
            "page": page,
            "video_type": video_type or s.PIXABAY_VIDEO_TYPE,
            "min_width": min_width or s.PIXABAY_FOOTAGE_FALLBACK_WIDTH,
            "min_height": min_height or s.PIXABAY_FOOTAGE_FALLBACK_HEIGHT,
        }
        cache_key = make_cache_key_bytes("video", query, json.dumps(params, sort_keys=True))
        cached = await self._cache.get(cache_key)
        if cached is not None:
            candidates = [Candidate.from_dict(d) for d in cached.get("hits", [])]
            logger.debug("pixabay_video_cache_hit", extra={"query": query, "count": len(candidates)})
            return candidates

        data = await self._get(self._video_base, params)
        hits = data.get("hits", [])
        candidates = [c for h in hits if (c := _video_hit_to_candidate(h, s)) is not None]
        logger.info(
            "pixabay_video_search",
            extra={"query": query, "total_hits": data.get("totalHits", 0), "candidates": len(candidates)},
        )
        await self._cache.set(cache_key, {"hits": [c.to_dict() for c in candidates]})
        return candidates

    async def search_images(
        self,
        query: str,
        per_page: int | None = None,
        page: int = 1,
        image_type: str = "photo",
        min_width: int | None = None,
        min_height: int | None = None,
    ) -> list[Candidate]:
        """
        Search Pixabay images and return normalised IMAGE Candidates.
        """
        s = self._s
        params: dict[str, Any] = {
            "key": s.PIXABAY_API_KEY,
            "q": query,
            "per_page": per_page or s.PIXABAY_RESULTS_PER_PAGE,
            "page": page,
            "image_type": image_type,
            "min_width": min_width or s.PIXABAY_FOOTAGE_FALLBACK_WIDTH,
            "min_height": min_height or s.PIXABAY_FOOTAGE_FALLBACK_HEIGHT,
        }
        cache_key = make_cache_key_bytes("image", query, json.dumps(params, sort_keys=True))
        cached = await self._cache.get(cache_key)
        if cached is not None:
            candidates = [Candidate.from_dict(d) for d in cached.get("hits", [])]
            logger.debug("pixabay_image_cache_hit", extra={"query": query, "count": len(candidates)})
            return candidates

        data = await self._get(self._image_base, params)
        hits = data.get("hits", [])
        candidates = [c for h in hits if (c := _image_hit_to_candidate(h, s)) is not None]
        logger.info(
            "pixabay_image_search",
            extra={"query": query, "total_hits": data.get("totalHits", 0), "candidates": len(candidates)},
        )
        await self._cache.set(cache_key, {"hits": [c.to_dict() for c in candidates]})
        return candidates

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Fetch *url* with rate-limiting and retry logic."""
        s = self._s
        last_exc: Exception | None = None

        for attempt in range(1, s.PIXABAY_FOOTAGE_MAX_RETRIES + 1):
            await self._bucket.acquire()

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, params=params)
            except httpx.TimeoutException as exc:
                logger.warning(
                    "pixabay_timeout",
                    extra={"url": url, "attempt": attempt, "error": str(exc)},
                )
                last_exc = PixabayServiceError(
                    f"Pixabay request timed out (attempt {attempt})",
                    details={"url": url},
                )
                await asyncio.sleep(2 ** attempt)
                continue
            except httpx.ConnectError as exc:
                logger.warning(
                    "pixabay_connect_error",
                    extra={"url": url, "attempt": attempt, "error": str(exc)},
                )
                last_exc = PixabayServiceError(
                    f"Could not connect to Pixabay (attempt {attempt})",
                    details={"url": url},
                )
                await asyncio.sleep(2 ** attempt)
                continue

            # 429 — respect Retry-After
            if response.status_code == 429:
                wait = _parse_retry_after(response)
                logger.warning(
                    "pixabay_rate_limited",
                    extra={"retry_after": wait, "attempt": attempt},
                )
                if attempt < s.PIXABAY_FOOTAGE_MAX_RETRIES:
                    await asyncio.sleep(wait)
                    continue
                raise RateLimitError(
                    "Pixabay rate limit exceeded",
                    details={"retry_after_seconds": wait},
                )

            # Auth errors — do not retry
            if _is_auth_error(response):
                raise PixabayAuthError(
                    "Pixabay API key is invalid or missing",
                    details={"status_code": response.status_code, "body": response.text[:300]},
                )

            # Other non-2xx
            if not response.is_success:
                last_exc = PixabayServiceError(
                    f"Pixabay returned HTTP {response.status_code}",
                    details={"body": response.text[:300]},
                )
                logger.warning(
                    "pixabay_http_error",
                    extra={"status": response.status_code, "attempt": attempt},
                )
                await asyncio.sleep(2 ** attempt)
                continue

            try:
                return response.json()  # type: ignore[no-any-return]
            except Exception as exc:
                last_exc = PixabayServiceError("Pixabay response is not valid JSON")
                await asyncio.sleep(2 ** attempt)
                continue

        raise last_exc or PixabayServiceError("Pixabay request failed after all retries")


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_adapter: PixabayFootageAdapter | None = None


def get_pixabay_footage_adapter() -> PixabayFootageAdapter:
    global _adapter
    if _adapter is None:
        _adapter = PixabayFootageAdapter()
    return _adapter


def reset_pixabay_footage_adapter() -> None:
    """Force re-initialisation (useful in tests that change settings)."""
    global _adapter
    _adapter = None
