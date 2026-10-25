"""
Shared pytest fixtures for footage adapter tests.

Import in test files with:
    from tests.fixtures.conftest_footage import *

Or add to conftest.py:
    pytest_plugins = ["tests.fixtures.conftest_footage"]

All fixtures use no real network, no real Redis, no real MinIO, no ffprobe.
Everything is deterministic and works in CI with no API keys.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Bootstrap env before any app import
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")


# ---------------------------------------------------------------------------
# Canonical test fixtures (Candidate objects, renditions, etc.)
# ---------------------------------------------------------------------------

def make_rendition(
    url: str = "https://cdn.pixabay.com/clip.mp4",
    width: int = 1920,
    height: int = 1080,
    label: str = "large",
    content_type: str = "video/mp4",
    size_bytes: int = 5_000_000,
) -> "RenditionInfo":  # type: ignore[name-defined]
    from app.adapters.footage_types import RenditionInfo
    return RenditionInfo(
        url=url, width=width, height=height,
        label=label, content_type=content_type, size_bytes=size_bytes,
    )


def make_video_candidate(
    cid: str = "pixabay:1001",
    url: str = "https://cdn.pixabay.com/clip.mp4",
    tags: list[str] | None = None,
    duration: float = 10.0,
    width: int = 1920,
    height: int = 1080,
) -> "Candidate":  # type: ignore[name-defined]
    from app.adapters.footage_types import Candidate, FallbackTier, FootageKind
    return Candidate(
        id=cid,
        source="pixabay",
        kind=FootageKind.VIDEO,
        url=url,
        thumb_url="https://cdn.pixabay.com/thumb.jpg",
        width=width,
        height=height,
        duration=duration,
        license="Pixabay Content License",
        commercial_use=True,
        attribution="Video by testuser on Pixabay",
        page_url="https://pixabay.com/videos/test-1001/",
        tags=tags or ["city", "skyline", "urban"],
        is_ai_generated=False,
        renditions=[
            make_rendition(url=url, width=1920, height=1080, label="large"),
            make_rendition(url=url.replace(".mp4", "_med.mp4"), width=1280, height=720, label="medium"),
        ],
        fallback_tier=FallbackTier.PIXABAY_VIDEO,
    )


def make_image_candidate(
    cid: str = "pixabay_img:2001",
    url: str = "https://cdn.pixabay.com/image.jpg",
    tags: list[str] | None = None,
) -> "Candidate":  # type: ignore[name-defined]
    from app.adapters.footage_types import Candidate, FallbackTier, FootageKind, RenditionInfo
    return Candidate(
        id=cid,
        source="pixabay",
        kind=FootageKind.IMAGE,
        url=url,
        thumb_url="https://cdn.pixabay.com/preview.jpg",
        width=1920,
        height=1280,
        duration=0.0,
        license="Pixabay Content License",
        commercial_use=True,
        attribution="Image by testuser on Pixabay",
        page_url="https://pixabay.com/photos/test-2001/",
        tags=tags or ["mountain", "nature", "landscape"],
        is_ai_generated=False,
        renditions=[
            RenditionInfo(
                url=url, width=1920, height=1280,
                label="full", content_type="image/jpeg",
            )
        ],
        fallback_tier=FallbackTier.KEN_BURNS,
    )


def make_card_candidate(keyword: str = "city skyline") -> "Candidate":  # type: ignore[name-defined]
    from app.adapters.footage_types import Candidate, FallbackTier, FootageKind, RenditionInfo
    h = hashlib.sha256(keyword.encode()).hexdigest()[:16]
    obj = f"footage/cards/shared/{h}.mp4"
    return Candidate(
        id=f"card:{h}",
        source="ffmpeg_card",
        kind=FootageKind.CARD,
        url="",
        thumb_url="",
        width=1920,
        height=1080,
        duration=5.0,
        license="Generated",
        commercial_use=True,
        attribution="",
        page_url="",
        tags=[keyword.lower()],
        is_ai_generated=False,
        renditions=[RenditionInfo(url=obj, width=1920, height=1080, size_bytes=100_000, label="card", content_type="video/mp4")],
        storage_object=obj,
        storage_bucket="broll-processed",
        checksum_sha256=hashlib.sha256(b"fake_card").hexdigest(),
        file_size_bytes=100_000,
        fallback_tier=FallbackTier.TITLE_CARD,
    )


# ---------------------------------------------------------------------------
# Mock minimal MP4 bytes (not a real video — just enough for byte-level tests)
# ---------------------------------------------------------------------------

FAKE_MP4_BYTES: bytes = (
    b"\x00\x00\x00\x20ftypisom"   # ftyp box header
    + b"\x00" * 24
    + b"\x00\x00\x00\x08free"     # free box
)

FAKE_JPEG_BYTES: bytes = (
    b"\xff\xd8\xff\xe0"            # JPEG SOI + APP0
    + b"\x00\x10JFIF\x00"
    + b"\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    + b"\xff\xd9"                  # EOI
)


# ---------------------------------------------------------------------------
# Fake in-memory Redis store
# ---------------------------------------------------------------------------

class FakeRedis:
    """
    Synchronous in-memory dict that exposes the async Redis interface
    used by AdapterCache, ClipStore, and PixabayFootageAdapter.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._sets: dict[str, set[str]] = {}

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def set(self, key: str, value: str) -> None:
        self._store[key] = value

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self._store[key] = value

    async def delete(self, *keys: str) -> None:
        for k in keys:
            self._store.pop(k, None)
            self._sets.pop(k, None)

    async def sadd(self, key: str, *members: str) -> int:
        s = self._sets.setdefault(key, set())
        before = len(s)
        s.update(members)
        return len(s) - before

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def smembers(self, key: str) -> set[str]:
        return set(self._sets.get(key, set()))

    async def srem(self, key: str, *members: str) -> int:
        s = self._sets.get(key, set())
        before = len(s)
        s.difference_update(members)
        return before - len(s)

    async def spop(self, key: str) -> str | None:
        s = self._sets.get(key, set())
        if s:
            member = next(iter(s))
            s.discard(member)
            return member
        return None

    async def expire(self, key: str, ttl: int) -> None:
        pass  # no-op — no TTL in the fake

    def pipeline(self) -> "_FakePipeline":
        return _FakePipeline(self)

    def clear(self) -> None:
        self._store.clear()
        self._sets.clear()


class _FakePipeline:
    """Queues commands and executes them synchronously on execute()."""

    def __init__(self, redis: FakeRedis) -> None:
        self._r = redis
        self._cmds: list[tuple[str, tuple[Any, ...]]] = []

    def setex(self, key: str, ttl: int, value: str) -> "_FakePipeline":
        self._cmds.append(("setex", (key, ttl, value)))
        return self

    def sadd(self, key: str, *members: str) -> "_FakePipeline":
        self._cmds.append(("sadd", (key, *members)))
        return self

    def srem(self, key: str, *members: str) -> "_FakePipeline":
        self._cmds.append(("srem", (key, *members)))
        return self

    def delete(self, *keys: str) -> "_FakePipeline":
        self._cmds.append(("delete", keys))
        return self

    def expire(self, key: str, ttl: int) -> "_FakePipeline":
        return self  # no-op

    async def execute(self) -> list[Any]:
        results = []
        for cmd, args in self._cmds:
            fn = getattr(self._r, cmd)
            import inspect
            if inspect.iscoroutinefunction(fn):
                results.append(await fn(*args))
            else:
                results.append(fn(*args))
        return results


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_redis() -> FakeRedis:
    """A fresh in-memory Redis fake for each test."""
    return FakeRedis()


@pytest.fixture()
def patched_redis(fake_redis: FakeRedis) -> Iterator[FakeRedis]:
    """Patch all Redis clients in footage modules to use the fake."""
    with patch("app.adapters._cache._get_redis", return_value=fake_redis), \
         patch("app.adapters.clip_store._get_redis", return_value=fake_redis), \
         patch("app.adapters.pixabay_footage.AdapterCache", autospec=False):
        yield fake_redis


@pytest.fixture()
def mock_storage() -> Iterator[MagicMock]:
    """Mock storage client — no real MinIO needed."""
    storage = MagicMock()
    storage.ensure_bucket = AsyncMock()
    storage.upload_bytes = AsyncMock(return_value="footage/test/clip.mp4")
    storage.upload_file = AsyncMock(return_value="footage/test/clip.mp4")
    storage.download_bytes = AsyncMock(return_value=FAKE_MP4_BYTES)
    storage.presigned_get_url = AsyncMock(return_value="https://minio.test/clip.mp4")

    with patch("app.adapters.media_download.get_storage_client", return_value=storage), \
         patch("app.adapters.ffmpeg_card.get_storage_client", return_value=storage), \
         patch("app.adapters.footage_router.get_storage_client", return_value=storage):
        yield storage


@pytest.fixture()
def mock_ffprobe_ok() -> Iterator[None]:
    """Patch _ffprobe_info to return a valid 1920x1080 video probe."""
    probe = {
        "streams": [{
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "duration": "10.0",
        }],
        "format": {"duration": "10.0"},
    }
    with patch("app.adapters.media_download._ffprobe_info", return_value=probe):
        yield


@pytest.fixture()
def mock_ffprobe_fail() -> Iterator[None]:
    """Patch _ffprobe_info to raise FfprobeValidationError."""
    from app.core.errors import FfprobeValidationError
    with patch(
        "app.adapters.media_download._ffprobe_info",
        side_effect=FfprobeValidationError("no video stream"),
    ):
        yield


@pytest.fixture()
def mock_ffmpeg_card_ok() -> Iterator[MagicMock]:
    """Mock generate_title_card to return a deterministic card Candidate."""
    async def _fake_card(keyword: str, job_id: str | None = None, bucket: str | None = None):
        return make_card_candidate(keyword)

    with patch("app.adapters.footage_router.generate_title_card", side_effect=_fake_card) as m:
        yield m


@pytest.fixture()
def mock_pixabay_videos(request) -> Iterator[MagicMock]:  # type: ignore[no-untyped-def]
    """
    Patch PixabayFootageAdapter.search_videos.

    Parameterise via:
        @pytest.mark.parametrize("mock_pixabay_videos", [raises_timeout], indirect=True)
    Default: returns one video candidate.
    """
    candidates = getattr(request, "param", None)
    if candidates is None:
        candidates = [make_video_candidate()]

    async def _search_videos(*args: Any, **kwargs: Any) -> list[Any]:
        if isinstance(candidates, Exception):
            raise candidates
        if callable(candidates) and not isinstance(candidates, list):
            return candidates()
        return candidates  # type: ignore[return-value]

    with patch(
        "app.adapters.pixabay_footage.PixabayFootageAdapter.search_videos",
        side_effect=_search_videos,
    ) as m:
        yield m


@pytest.fixture()
def mock_pixabay_images(request) -> Iterator[MagicMock]:  # type: ignore[no-untyped-def]
    """Patch PixabayFootageAdapter.search_images. Default: returns one image candidate."""
    candidates = getattr(request, "param", None)
    if candidates is None:
        candidates = [make_image_candidate()]

    async def _search_images(*args: Any, **kwargs: Any) -> list[Any]:
        if isinstance(candidates, Exception):
            raise candidates
        return candidates  # type: ignore[return-value]

    with patch(
        "app.adapters.pixabay_footage.PixabayFootageAdapter.search_images",
        side_effect=_search_images,
    ) as m:
        yield m


@pytest.fixture()
def mock_download_to_minio() -> Iterator[MagicMock]:
    """Patch download_to_minio to return a DownloadedClip without any I/O."""
    from app.adapters.footage_types import DownloadedClip

    async def _fake_download(candidate: Any, bucket: str | None = None, ffprobe_validate: bool = True) -> DownloadedClip:
        candidate.storage_object = f"footage/pixabay/{candidate.id.replace(':', '_')}.mp4"
        candidate.storage_bucket = bucket or "broll-processed"
        candidate.checksum_sha256 = hashlib.sha256(candidate.url.encode()).hexdigest()
        candidate.file_size_bytes = 5_000_000
        return DownloadedClip(
            candidate=candidate,
            object_key=candidate.storage_object,
            bucket=candidate.storage_bucket,
            checksum_sha256=candidate.checksum_sha256,
            file_size_bytes=5_000_000,
            validated_width=candidate.width,
            validated_height=candidate.height,
            validated_duration=candidate.duration,
        )

    with patch("app.adapters.footage_router.download_to_minio", side_effect=_fake_download) as m:
        yield m


@pytest.fixture()
def mock_clip_store_empty() -> Iterator[MagicMock]:
    """Patch ClipStore.find_by_tags to return empty (no cached clips)."""
    store = MagicMock()
    store.find_by_tags = AsyncMock(return_value=[])
    store.register = AsyncMock()
    store.count = AsyncMock(return_value=0)
    with patch("app.adapters.footage_router.get_clip_store", return_value=store), \
         patch("app.adapters.clip_store.get_clip_store", return_value=store):
        yield store


@pytest.fixture()
def mock_clip_store_with_match() -> Iterator[MagicMock]:
    """Patch ClipStore.find_by_tags to return one matching cached clip."""
    store = MagicMock()
    cached = make_video_candidate(cid="pixabay:cached-99", tags=["city", "skyline"])
    cached.storage_object = "footage/pixabay/pixabay_cached-99.mp4"
    cached.storage_bucket = "broll-processed"
    store.find_by_tags = AsyncMock(return_value=[cached])
    store.register = AsyncMock()
    store.count = AsyncMock(return_value=1)
    with patch("app.adapters.footage_router.get_clip_store", return_value=store), \
         patch("app.adapters.clip_store.get_clip_store", return_value=store):
        yield store
