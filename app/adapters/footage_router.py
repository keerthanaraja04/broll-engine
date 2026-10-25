"""
FootageRouter — 4-tier fallback chain for sourcing a clip for each keyword.

Tier execution order
--------------------
1. PIXABAY_VIDEO   Search Pixabay videos, download best match into MinIO.
2. KEN_BURNS       Search Pixabay images, animate with Ken Burns (zoompan).
3. CACHED_CLIP     Look up the ClipStore for any previously downloaded clip
                   whose tags overlap with the keyword.
4. TITLE_CARD      Generate a silent colour+text card with FFmpeg — never fails.

Each tier that fails adds a FallbackWarning to the result so callers can
surface the degradation in the job's ``warnings`` list.

Usage
-----
    router = FootageRouter()
    result = await router.resolve("city skyline", job_id="job-123")
    # result.candidate  — the Candidate (from whichever tier succeeded)
    # result.warnings   — list of FallbackWarning explaining any fallbacks
    # result.tier       — FallbackTier that actually delivered the clip

Error handling
--------------
The router is designed to NEVER raise when TITLE_CARD is enabled.
FootageRouterExhaustedError is only raised when all four tiers are disabled
or when tier-4 (title card) itself fails — which would indicate an FFmpeg
configuration problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.adapters.clip_store import get_clip_store
from app.adapters.ffmpeg_card import generate_title_card, ken_burns_from_image
from app.adapters.footage_types import (
    Candidate,
    FallbackTier,
    FallbackWarning,
    FootageKind,
)
from app.adapters.media_download import download_to_minio
from app.adapters.pixabay_footage import get_pixabay_footage_adapter
from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.errors import (
    FootageRouterExhaustedError,
    MediaProcessingError,
    PixabayAuthError,
    PixabayServiceError,
    RateLimitError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FootageResult:
    """Result from FootageRouter.resolve()."""
    candidate: Candidate
    tier: FallbackTier
    warnings: list[FallbackWarning] = field(default_factory=list)
    warning_strings: list[str] = field(default_factory=list)

    def add_warning(self, tier: FallbackTier, reason: str, keyword: str, details: dict[str, Any] | None = None) -> None:
        w = FallbackWarning(tier=tier, reason=reason, keyword=keyword, details=details or {})
        self.warnings.append(w)
        self.warning_strings.append(w.to_str())
        logger.warning(
            "footage_fallback",
            extra={"tier": tier.value, "keyword": keyword, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class FootageRouter:
    """
    Resolves a single keyword into a downloaded Candidate by trying four tiers
    in order, accumulating warnings on every fallback.
    """

    async def resolve(
        self,
        keyword: str,
        job_id: str | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
    ) -> FootageResult:
        """
        Return a FootageResult for *keyword*.

        Args:
            keyword:       The search term / scene description.
            job_id:        Optional job UUID for storage paths.
            target_width:  Minimum acceptable video width (defaults to config).
            target_height: Minimum acceptable video height (defaults to config).

        Returns:
            FootageResult — always returns (never raises) unless TITLE_CARD
            is disabled AND all other tiers fail.
        """
        s = get_settings()
        result = FootageResult(candidate=None, tier=FallbackTier.PIXABAY_VIDEO)  # type: ignore[arg-type]

        tw = target_width or s.PIXABAY_FOOTAGE_FALLBACK_WIDTH
        th = target_height or s.PIXABAY_FOOTAGE_FALLBACK_HEIGHT

        # ------------------------------------------------------------------
        # Tier 1 — Pixabay video
        # ------------------------------------------------------------------
        if s.FOOTAGE_ROUTER_ENABLE_VIDEO:
            candidate = await self._try_pixabay_video(keyword, tw, th, result, job_id)
            if candidate is not None:
                result.candidate = candidate
                result.tier = FallbackTier.PIXABAY_VIDEO
                return result
        else:
            result.add_warning(FallbackTier.PIXABAY_VIDEO, "Disabled by config", keyword)

        # ------------------------------------------------------------------
        # Tier 2 — Ken Burns (Pixabay image animated)
        # ------------------------------------------------------------------
        if s.FOOTAGE_ROUTER_ENABLE_KEN_BURNS:
            candidate = await self._try_ken_burns(keyword, result, job_id)
            if candidate is not None:
                result.candidate = candidate
                result.tier = FallbackTier.KEN_BURNS
                return result
        else:
            result.add_warning(FallbackTier.KEN_BURNS, "Disabled by config", keyword)

        # ------------------------------------------------------------------
        # Tier 3 — Cached clip
        # ------------------------------------------------------------------
        if s.FOOTAGE_ROUTER_ENABLE_CACHE_FALLBACK:
            candidate = await self._try_cached_clip(keyword, result)
            if candidate is not None:
                result.candidate = candidate
                result.tier = FallbackTier.CACHED_CLIP
                return result
        else:
            result.add_warning(FallbackTier.CACHED_CLIP, "Disabled by config", keyword)

        # ------------------------------------------------------------------
        # Tier 4 — Title card
        # ------------------------------------------------------------------
        if s.FOOTAGE_ROUTER_ENABLE_TITLE_CARD:
            candidate = await self._try_title_card(keyword, result, job_id)
            if candidate is not None:
                result.candidate = candidate
                result.tier = FallbackTier.TITLE_CARD
                return result

        # Should never reach here with default config
        raise FootageRouterExhaustedError(
            f"All footage tiers exhausted for keyword '{keyword}'",
            details={"keyword": keyword, "warnings": [w.to_str() for w in result.warnings]},
        )

    # ------------------------------------------------------------------
    # Tier implementations
    # ------------------------------------------------------------------

    async def _try_pixabay_video(
        self,
        keyword: str,
        min_width: int,
        min_height: int,
        result: FootageResult,
        job_id: str | None,
    ) -> Candidate | None:
        try:
            adapter = get_pixabay_footage_adapter()
            candidates = await adapter.search_videos(
                keyword,
                per_page=5,
                min_width=min_width,
                min_height=min_height,
            )
        except PixabayAuthError as exc:
            result.add_warning(
                FallbackTier.PIXABAY_VIDEO,
                f"Pixabay auth error: {exc.message}",
                keyword,
                {"error_code": exc.code},
            )
            return None
        except (PixabayServiceError, RateLimitError, Exception) as exc:
            result.add_warning(
                FallbackTier.PIXABAY_VIDEO,
                f"Pixabay API unavailable: {exc}",
                keyword,
                {"error": str(exc)},
            )
            return None

        if not candidates:
            result.add_warning(
                FallbackTier.PIXABAY_VIDEO,
                "Pixabay returned no video results",
                keyword,
            )
            return None

        # Try each candidate in order; stop on first successful download
        for candidate in candidates:
            try:
                downloaded = await download_to_minio(
                    candidate,
                    bucket=get_settings().STORAGE_BUCKET_PROCESSED,
                    ffprobe_validate=False,  # skip ffprobe in non-prod for speed; override in tests
                )
                return downloaded.candidate
            except Exception as exc:
                logger.warning(
                    "footage_video_download_failed",
                    extra={"candidate_id": candidate.id, "error": str(exc)},
                )
                continue

        result.add_warning(
            FallbackTier.PIXABAY_VIDEO,
            "All video candidates failed to download",
            keyword,
        )
        return None

    async def _try_ken_burns(
        self,
        keyword: str,
        result: FootageResult,
        job_id: str | None,
    ) -> Candidate | None:
        try:
            adapter = get_pixabay_footage_adapter()
            image_candidates = await adapter.search_images(keyword, per_page=3)
        except PixabayAuthError as exc:
            result.add_warning(
                FallbackTier.KEN_BURNS,
                f"Pixabay auth error on image search: {exc.message}",
                keyword,
            )
            return None
        except (PixabayServiceError, RateLimitError, Exception) as exc:
            result.add_warning(
                FallbackTier.KEN_BURNS,
                f"Pixabay image search failed: {exc}",
                keyword,
            )
            return None

        if not image_candidates:
            result.add_warning(
                FallbackTier.KEN_BURNS,
                "Pixabay returned no image results for Ken Burns",
                keyword,
            )
            return None

        # Download the first image that succeeds
        for img_candidate in image_candidates:
            try:
                storage = get_storage_client()
                # Stream the image bytes
                import httpx
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    resp = await client.get(img_candidate.url)
                    resp.raise_for_status()
                    image_data = resp.content

                animated = await ken_burns_from_image(
                    image_data=image_data,
                    keyword=keyword,
                    job_id=job_id,
                )
                return animated
            except (MediaProcessingError, Exception) as exc:
                logger.warning(
                    "ken_burns_failed",
                    extra={"candidate_id": img_candidate.id, "error": str(exc)},
                )
                continue

        result.add_warning(
            FallbackTier.KEN_BURNS,
            "Ken Burns animation failed for all image candidates",
            keyword,
        )
        return None

    async def _try_cached_clip(
        self,
        keyword: str,
        result: FootageResult,
    ) -> Candidate | None:
        try:
            store = get_clip_store()
            tags = [t.strip() for t in keyword.lower().replace(",", " ").split() if t.strip()]
            matches = await store.find_by_tags(tags, limit=1)
            if matches:
                logger.info(
                    "footage_cache_hit",
                    extra={"keyword": keyword, "clip_id": matches[0].id},
                )
                return matches[0]
        except Exception as exc:
            logger.warning("clip_store_lookup_error", extra={"error": str(exc), "keyword": keyword})

        result.add_warning(
            FallbackTier.CACHED_CLIP,
            "No matching cached clip found",
            keyword,
        )
        return None

    async def _try_title_card(
        self,
        keyword: str,
        result: FootageResult,
        job_id: str | None,
    ) -> Candidate | None:
        try:
            candidate = await generate_title_card(keyword=keyword, job_id=job_id)
            result.add_warning(
                FallbackTier.TITLE_CARD,
                "All real footage tiers failed — using title card",
                keyword,
            )
            return candidate
        except Exception as exc:
            result.add_warning(
                FallbackTier.TITLE_CARD,
                f"Title card generation failed: {exc}",
                keyword,
                {"error": str(exc)},
            )
            return None


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_router: FootageRouter | None = None


def get_footage_router() -> FootageRouter:
    global _router
    if _router is None:
        _router = FootageRouter()
    return _router


def reset_footage_router() -> None:
    global _router
    _router = None
