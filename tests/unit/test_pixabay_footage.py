"""
Unit tests for PixabayFootageAdapter.

Tests: candidate normalisation, search flow, cache hit/miss, 429 handling,
timeout, empty results, invalid API key. All network calls mocked.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


def _video_hit(pid: int = 1001, tags: str = "city, skyline") -> dict[str, Any]:
    return {
        "id": pid, "pageURL": f"https://pixabay.com/videos/{pid}/",
        "tags": tags, "duration": 10, "views": 1000, "downloads": 500,
        "likes": 100, "user": "testuser", "user_id": 42,
        "picture_id": f"pic-{pid}",
        "videos": {
            "large": {"url": f"https://cdn.pixabay.com/{pid}-large.mp4",
                       "width": 1920, "height": 1080, "size": 10_000_000},
            "medium": {"url": f"https://cdn.pixabay.com/{pid}-med.mp4",
                        "width": 1280, "height": 720, "size": 5_000_000},
        },
    }


def _image_hit(pid: int = 2001) -> dict[str, Any]:
    return {
        "id": pid, "pageURL": f"https://pixabay.com/photos/{pid}/",
        "tags": "mountain, nature",
        "previewURL": f"https://cdn.pixabay.com/{pid}-prev.jpg",
        "webformatURL": f"https://cdn.pixabay.com/{pid}-web.jpg",
        "webformatWidth": 1280, "webformatHeight": 720,
        "imageURL": f"https://cdn.pixabay.com/{pid}-full.jpg",
        "imageWidth": 2400, "imageHeight": 1600, "imageSize": 2_000_000,
        "user": "testuser", "views": 500, "downloads": 200, "likes": 50,
    }


class TestCandidateNormalisation:
    def test_video_hit_to_candidate(self) -> None:
        from app.adapters.pixabay_footage import _video_hit_to_candidate
        from app.adapters.footage_types import FootageKind
        from app.core.config import get_settings
        s = get_settings()
        c = _video_hit_to_candidate(_video_hit(), s)
        assert c is not None
        assert c.id == "pixabay:1001"
        assert c.source == "pixabay"
        assert c.kind == FootageKind.VIDEO
        assert c.license == "Pixabay Content License"
        assert c.commercial_use is True
        assert c.is_ai_generated is False
        assert "city" in c.tags
        assert "skyline" in c.tags
        assert "testuser" in c.attribution
        assert len(c.renditions) == 2

    def test_image_hit_to_candidate(self) -> None:
        from app.adapters.pixabay_footage import _image_hit_to_candidate
        from app.adapters.footage_types import FootageKind
        from app.core.config import get_settings
        c = _image_hit_to_candidate(_image_hit(), get_settings())
        assert c is not None
        assert c.kind == FootageKind.IMAGE
        assert c.duration == 0.0
        assert c.width == 2400

    def test_video_hit_no_renditions_returns_none(self) -> None:
        from app.adapters.pixabay_footage import _video_hit_to_candidate
        from app.core.config import get_settings
        hit = _video_hit()
        hit["videos"] = {}
        assert _video_hit_to_candidate(hit, get_settings()) is None

    def test_best_rendition_picks_target(self) -> None:
        from app.adapters.pixabay_footage import _video_hit_to_candidate
        from app.core.config import get_settings
        c = _video_hit_to_candidate(_video_hit(), get_settings())
        assert c is not None
        assert c.width == 1920
        assert c.height == 1080


class TestSearchVideos:
    @pytest.mark.asyncio
    async def test_returns_candidates(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            return_value={"totalHits": 1, "hits": [_video_hit()]}
        )):
            results = await adapter.search_videos("city skyline")
        assert len(results) == 1
        assert results[0].id == "pixabay:1001"

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty_list(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            return_value={"totalHits": 0, "hits": []}
        )):
            assert await adapter.search_videos("xyz") == []

    @pytest.mark.asyncio
    async def test_search_images_returns_candidates(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            return_value={"totalHits": 1, "hits": [_image_hit()]}
        )):
            results = await adapter.search_images("mountain")
        assert len(results) == 1
        assert results[0].kind.value == "image"


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_429_raises_rate_limit(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        from app.core.errors import RateLimitError
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            side_effect=RateLimitError("rate limited",
                details={"retry_after_seconds": 1})
        )):
            with pytest.raises(RateLimitError):
                await adapter.search_videos("test")

    @pytest.mark.asyncio
    async def test_timeout_raises_service_error(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        from app.core.errors import PixabayServiceError
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            side_effect=PixabayServiceError("timeout")
        )):
            with pytest.raises(PixabayServiceError):
                await adapter.search_videos("test")

    @pytest.mark.asyncio
    async def test_auth_error_raises_pixabay_auth(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        from app.core.errors import PixabayAuthError
        adapter = PixabayFootageAdapter()
        with patch.object(adapter, "_get", AsyncMock(
            side_effect=PixabayAuthError("invalid key")
        )):
            with pytest.raises(PixabayAuthError):
                await adapter.search_videos("test")

    @pytest.mark.asyncio
    async def test_auth_error_is_non_retryable(self) -> None:
        from app.core.errors import PixabayAuthError
        err = PixabayAuthError("bad key")
        assert err.retryable is False
        assert err.code == "PIXABAY_AUTH_ERROR"

    def test_is_auth_error_detects_invalid_key(self) -> None:
        from app.adapters.pixabay_footage import _is_auth_error
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "Error: API key invalid"
        assert _is_auth_error(resp) is True

    def test_is_auth_error_negative(self) -> None:
        from app.adapters.pixabay_footage import _is_auth_error
        resp = MagicMock()
        resp.status_code = 500
        resp.text = "Internal server error"
        assert _is_auth_error(resp) is False

    def test_parse_retry_after_defaults_60(self) -> None:
        from app.adapters.pixabay_footage import _parse_retry_after
        resp = MagicMock()
        resp.headers = {}
        assert _parse_retry_after(resp) == 60.0

    def test_parse_retry_after_reads_header(self) -> None:
        from app.adapters.pixabay_footage import _parse_retry_after
        resp = MagicMock()
        resp.headers = {"retry-after": "5"}
        assert _parse_retry_after(resp) == 5.0


class TestSearchCache:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, fake_redis) -> None:  # type: ignore[no-untyped-def]
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        from app.adapters.footage_types import Candidate, FootageKind, RenditionInfo, FallbackTier
        import json
        adapter = PixabayFootageAdapter()
        # Pre-populate cache
        cached_cand = Candidate(
            id="pixabay:999", source="pixabay", kind=FootageKind.VIDEO,
            url="https://cached.mp4", thumb_url="", width=1920, height=1080,
            duration=10.0, tags=["cached"],
            renditions=[RenditionInfo(url="https://cached.mp4",
                width=1920, height=1080, label="large")],
        )
        cache_key_bytes = adapter._cache._key(
            __import__("app.adapters._cache", fromlist=["make_cache_key_bytes"])
            .make_cache_key_bytes("video", "city",
                json.dumps({"key": "test"}, sort_keys=True))
        ) if False else b""  # We'll use mock instead

        # Simpler: mock _get to verify it's NOT called on cache hit
        call_count = 0
        original_get = adapter._cache.get

        async def _fake_get(raw: bytes) -> dict[str, Any] | None:
            return {"hits": [cached_cand.to_dict()]}

        adapter._cache.get = _fake_get  # type: ignore[assignment]
        adapter._cache.set = AsyncMock()  # type: ignore[assignment]

        with patch.object(adapter, "_get", AsyncMock()) as mock_get:
            results = await adapter.search_videos("city")
            assert len(results) == 1
            assert results[0].id == "pixabay:999"
            mock_get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self) -> None:
        from app.adapters.pixabay_footage import PixabayFootageAdapter
        adapter = PixabayFootageAdapter()
        adapter._cache.get = AsyncMock(return_value=None)  # type: ignore[assignment]
        adapter._cache.set = AsyncMock()  # type: ignore[assignment]
        with patch.object(adapter, "_get", AsyncMock(
            return_value={"totalHits": 1, "hits": [_video_hit()]}
        )):
            results = await adapter.search_videos("city")
        assert len(results) == 1
        adapter._cache.set.assert_called_once()


class TestFactory:
    def test_factory_returns_adapter(self) -> None:
        from app.adapters.pixabay_footage import (
            get_pixabay_footage_adapter,
            reset_pixabay_footage_adapter,
            PixabayFootageAdapter,
        )
        reset_pixabay_footage_adapter()
        adapter = get_pixabay_footage_adapter()
        assert isinstance(adapter, PixabayFootageAdapter)
        reset_pixabay_footage_adapter()

    def test_factory_returns_singleton(self) -> None:
        from app.adapters.pixabay_footage import (
            get_pixabay_footage_adapter, reset_pixabay_footage_adapter)
        reset_pixabay_footage_adapter()
        a1 = get_pixabay_footage_adapter()
        a2 = get_pixabay_footage_adapter()
        assert a1 is a2
        reset_pixabay_footage_adapter()
