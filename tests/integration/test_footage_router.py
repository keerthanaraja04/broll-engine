"""
Integration tests for FootageRouter — 4-tier fallback chain.

Tests all 4 tiers, the "Pixabay off mid-job" scenario, and warning
accumulation. Uses the footage conftest fixtures (mock adapters, fake
Redis, mock storage).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")

from tests.fixtures.conftest_footage import (
    make_video_candidate, make_image_candidate, make_card_candidate,
)


class TestTier1PixabayVideo:
    @pytest.mark.asyncio
    async def test_tier1_succeeds(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        router = FootageRouter()
        result = await router.resolve("city skyline", job_id="job-1")
        assert result.tier == FallbackTier.PIXABAY_VIDEO
        assert result.candidate is not None
        assert result.candidate.id == "pixabay:1001"
        assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_tier1_empty_results_falls_through(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any,
        mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        router = FootageRouter()
        # Pixabay returns empty for both video and image searches
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos", AsyncMock(return_value=[])), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images", AsyncMock(return_value=[])):
            result = await router.resolve("nothing here")
        assert result.tier == FallbackTier.TITLE_CARD
        assert len(result.warnings) >= 2
        assert any("no video results" in w.reason for w in result.warnings)
        assert any("no image results" in w.reason for w in result.warnings)


class TestTier2KenBurns:
    @pytest.mark.asyncio
    async def test_tier1_fails_tier2_succeeds(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        router = FootageRouter()
        # Tier 1: video search returns [] → falls to tier 2
        # Tier 2: image search returns a candidate → Ken Burns
        kb_candidate = make_image_candidate()
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos", AsyncMock(return_value=[])), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images",
                   AsyncMock(return_value=[kb_candidate])), \
             patch("app.adapters.footage_router.ken_burns_from_image",
                   AsyncMock(return_value=make_card_candidate("mountain"))), \
             patch("httpx.AsyncClient") as mock_http:
            mock_client = MagicMock()
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.content = b"fake-image"
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_http.return_value = mock_client
            result = await router.resolve("mountain", job_id="job-2")
        assert result.tier == FallbackTier.KEN_BURNS
        assert len(result.warnings) >= 1
        assert any("no video results" in w.reason for w in result.warnings)


class TestTier3CachedClip:
    @pytest.mark.asyncio
    async def test_tier3_succeeds_when_pixabay_down(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_with_match: Any, mock_storage: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        from app.core.errors import PixabayServiceError
        router = FootageRouter()
        # Both Pixabay searches fail → tier 3 cached clip found
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos",
                   AsyncMock(side_effect=PixabayServiceError("API down"))), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images",
                   AsyncMock(side_effect=PixabayServiceError("API down"))):
            result = await router.resolve("city skyline", job_id="job-3")
        assert result.tier == FallbackTier.CACHED_CLIP
        assert result.candidate.id == "pixabay:cached-99"
        assert len(result.warnings) >= 2
        assert any("API unavailable" in w.reason for w in result.warnings)
        assert any("Pixabay image search failed" in w.reason
                   for w in result.warnings)


class TestTier4TitleCard:
    @pytest.mark.asyncio
    async def test_all_tiers_fail_falls_to_title_card(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any,
        mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        router = FootageRouter()
        # All searches return empty, cache empty, title card generates
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos", AsyncMock(return_value=[])), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images", AsyncMock(return_value=[])):
            result = await router.resolve("missing keyword")
        assert result.tier == FallbackTier.TITLE_CARD
        assert result.candidate.source == "ffmpeg_card"
        assert len(result.warnings) >= 3
        assert any("no video results" in w.reason for w in result.warnings)
        assert any("no image results" in w.reason for w in result.warnings)
        assert any("No matching cached clip" in w.reason
                   for w in result.warnings)
        assert any("title card" in w.reason.lower()
                   for w in result.warnings)

    @pytest.mark.asyncio
    async def test_title_card_is_last_resort(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any,
        mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        router = FootageRouter()
        from app.core.errors import PixabayAuthError
        # Pixabay auth error → non-retryable, both tiers skip
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos",
                   AsyncMock(side_effect=PixabayAuthError("bad key"))), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images",
                   AsyncMock(side_effect=PixabayAuthError("bad key"))):
            result = await router.resolve("test keyword")
        assert result.tier == FallbackTier.TITLE_CARD


class TestPixabayOffMidJob:
    """
    Simulate Pixabay going down between keywords during a multi-keyword job.
    First keyword succeeds via Pixabay; second keyword finds Pixabay down
    and falls through to cached clip or title card — the job still finishes.
    """

    @pytest.mark.asyncio
    async def test_first_succeeds_second_pixabay_down(
        self, mock_download_to_minio: Any, mock_clip_store_empty: Any,
        mock_storage: Any, mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        from app.core.errors import PixabayServiceError
        router = FootageRouter()

        call_count = 0

        async def _flaky_search(*args: Any, **kwargs: Any) -> list[Any]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return [make_video_candidate(cid="pixabay:ok-1")]
            raise PixabayServiceError("API went down mid-job")

        async def _flaky_images(*args: Any, **kwargs: Any) -> list[Any]:
            raise PixabayServiceError("API down")

        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos", side_effect=_flaky_search), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images", side_effect=_flaky_images):
            # First keyword: Pixabay video succeeds
            r1 = await router.resolve("city", job_id="job-mid")
            assert r1.tier == FallbackTier.PIXABAY_VIDEO
            assert len(r1.warnings) == 0

            # Second keyword: Pixabay is down → falls to title card
            r2 = await router.resolve("forest", job_id="job-mid")
            assert r2.tier == FallbackTier.TITLE_CARD
            assert len(r2.warnings) >= 2

    @pytest.mark.asyncio
    async def test_job_always_completes(
        self, mock_download_to_minio: Any, mock_clip_store_empty: Any,
        mock_storage: Any, mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.adapters.footage_types import FallbackTier
        from app.core.errors import PixabayServiceError
        router = FootageRouter()
        # Pixabay fully down for all keywords
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos",
                   AsyncMock(side_effect=PixabayServiceError("down"))), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images",
                   AsyncMock(side_effect=PixabayServiceError("down"))):
            keywords = ["city", "nature", "abstract"]
            results = [await router.resolve(kw) for kw in keywords]
        # Every keyword must produce a result — none should raise
        for r in results:
            assert r.candidate is not None
            assert r.tier == FallbackTier.TITLE_CARD
            assert len(r.warning_strings) > 0


class TestWarningAccumulation:
    @pytest.mark.asyncio
    async def test_warnings_contain_tier_and_reason(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any,
        mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter, FootageResult
        router = FootageRouter()
        with patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_videos", AsyncMock(return_value=[])), \
             patch("app.adapters.pixabay_footage.PixabayFootageAdapter"
                   ".search_images", AsyncMock(return_value=[])):
            result = await router.resolve("test")
        for w in result.warnings:
            assert w.tier is not None
            assert w.reason
            assert w.keyword == "test"
        assert len(result.warning_strings) == len(result.warnings)
        # Warning strings should contain tier numbers
        for s in result.warning_strings:
            assert "Tier" in s

    @pytest.mark.asyncio
    async def test_no_warnings_on_success(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        router = FootageRouter()
        result = await router.resolve("city skyline", job_id="job-clean")
        assert len(result.warnings) == 0
        assert len(result.warning_strings) == 0


class TestDisabledTiers:
    @pytest.mark.asyncio
    async def test_all_disabled_raises_exhausted(
        self, mock_pixabay_videos: Any, mock_download_to_minio: Any,
        mock_clip_store_empty: Any, mock_storage: Any,
        mock_ffmpeg_card_ok: Any
    ) -> None:
        from app.adapters.footage_router import FootageRouter
        from app.core.errors import FootageRouterExhaustedError
        router = FootageRouter()
        with patch.dict(os.environ, {
            "FOOTAGE_ROUTER_ENABLE_VIDEO": "false",
            "FOOTAGE_ROUTER_ENABLE_KEN_BURNS": "false",
            "FOOTAGE_ROUTER_ENABLE_CACHE_FALLBACK": "false",
            "FOOTAGE_ROUTER_ENABLE_TITLE_CARD": "false",
        }):
            from app.core.config import get_settings
            get_settings.cache_clear()
            with pytest.raises(FootageRouterExhaustedError):
                await router.resolve("test")
            get_settings.cache_clear()
