"""
Unit tests for Candidate, RenditionInfo, and footage_types helpers.
No network, no Redis, no MinIO.
"""

from __future__ import annotations

import hashlib
import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")

from app.adapters.footage_types import (
    Candidate,
    DownloadedClip,
    FallbackTier,
    FallbackWarning,
    FootageKind,
    RenditionInfo,
)


# ---------------------------------------------------------------------------
# RenditionInfo
# ---------------------------------------------------------------------------

class TestRenditionInfo:
    def test_pixels_product(self) -> None:
        r = RenditionInfo(url="x", width=1920, height=1080)
        assert r.pixels == 1920 * 1080

    def test_meets_minimum_exact(self) -> None:
        r = RenditionInfo(url="x", width=1280, height=720)
        assert r.meets_minimum(1280, 720) is True

    def test_meets_minimum_larger(self) -> None:
        r = RenditionInfo(url="x", width=1920, height=1080)
        assert r.meets_minimum(1280, 720) is True

    def test_meets_minimum_fails_width(self) -> None:
        r = RenditionInfo(url="x", width=640, height=1080)
        assert r.meets_minimum(1280, 720) is False

    def test_meets_minimum_fails_height(self) -> None:
        r = RenditionInfo(url="x", width=1920, height=360)
        assert r.meets_minimum(1280, 720) is False


# ---------------------------------------------------------------------------
# Candidate.best_rendition()
# ---------------------------------------------------------------------------

class TestCandidateBestRendition:
    def _make(self, renditions: list[RenditionInfo]) -> Candidate:
        return Candidate(
            id="test:1", source="pixabay", kind=FootageKind.VIDEO,
            url="https://example.com/clip.mp4", thumb_url="", width=1920, height=1080,
            duration=10.0, renditions=renditions,
        )

    def test_returns_none_when_no_renditions(self) -> None:
        c = self._make([])
        assert c.best_rendition() is None

    def test_returns_target_resolution_first(self) -> None:
        r1080 = RenditionInfo(url="1080.mp4", width=1920, height=1080, label="large")
        r720 = RenditionInfo(url="720.mp4", width=1280, height=720, label="medium")
        r480 = RenditionInfo(url="480.mp4", width=854, height=480, label="small")
        c = self._make([r480, r720, r1080])  # deliberately un-sorted
        best = c.best_rendition(target_width=1920, target_height=1080)
        assert best is not None
        assert best.url == "1080.mp4"

    def test_falls_back_to_fallback_resolution(self) -> None:
        r720 = RenditionInfo(url="720.mp4", width=1280, height=720, label="medium")
        r480 = RenditionInfo(url="480.mp4", width=854, height=480, label="small")
        c = self._make([r720, r480])
        best = c.best_rendition(target_width=1920, target_height=1080, fallback_width=1280, fallback_height=720)
        assert best is not None
        assert best.url == "720.mp4"

    def test_falls_back_to_any_rendition(self) -> None:
        r480 = RenditionInfo(url="480.mp4", width=854, height=480, label="small")
        c = self._make([r480])
        best = c.best_rendition(target_width=1920, target_height=1080, fallback_width=1280, fallback_height=720)
        assert best is not None
        assert best.url == "480.mp4"

    def test_highest_pixel_count_wins_among_equals(self) -> None:
        r1 = RenditionInfo(url="a.mp4", width=2560, height=1440, label="xl")
        r2 = RenditionInfo(url="b.mp4", width=1920, height=1080, label="large")
        c = self._make([r2, r1])
        best = c.best_rendition(target_width=1920, target_height=1080)
        assert best is not None
        assert best.url == "a.mp4"


# ---------------------------------------------------------------------------
# Candidate.tag_set()
# ---------------------------------------------------------------------------

class TestCandidateTagSet:
    def test_tags_lowercased(self) -> None:
        c = Candidate(
            id="t:1", source="p", kind=FootageKind.VIDEO,
            url="x", thumb_url="", width=100, height=100, duration=1.0,
            tags=["City", "SKYLINE", "urban life"],
        )
        ts = c.tag_set()
        assert "city" in ts
        assert "skyline" in ts
        assert "urban life" in ts

    def test_empty_tags_give_empty_frozenset(self) -> None:
        c = Candidate(
            id="t:2", source="p", kind=FootageKind.VIDEO,
            url="x", thumb_url="", width=100, height=100, duration=1.0,
        )
        assert c.tag_set() == frozenset()

    def test_whitespace_tags_stripped(self) -> None:
        c = Candidate(
            id="t:3", source="p", kind=FootageKind.VIDEO,
            url="x", thumb_url="", width=100, height=100, duration=1.0,
            tags=["  city  ", "   "],
        )
        ts = c.tag_set()
        assert "city" in ts
        assert "" not in ts


# ---------------------------------------------------------------------------
# Candidate serialisation round-trip
# ---------------------------------------------------------------------------

class TestCandidateSerialisation:
    def _make_full(self) -> Candidate:
        return Candidate(
            id="pixabay:9999",
            source="pixabay",
            kind=FootageKind.VIDEO,
            url="https://cdn.example.com/video.mp4",
            thumb_url="https://cdn.example.com/thumb.jpg",
            width=1920,
            height=1080,
            duration=15.5,
            license="Pixabay Content License",
            commercial_use=True,
            attribution="Video by user on Pixabay",
            page_url="https://pixabay.com/videos/9999/",
            tags=["city", "skyline"],
            is_ai_generated=False,
            renditions=[
                RenditionInfo(url="https://cdn.example.com/large.mp4", width=1920, height=1080,
                              size_bytes=10_000_000, label="large", content_type="video/mp4"),
                RenditionInfo(url="https://cdn.example.com/medium.mp4", width=1280, height=720,
                              size_bytes=5_000_000, label="medium", content_type="video/mp4"),
            ],
            storage_object="footage/pixabay/pixabay_9999.mp4",
            storage_bucket="broll-processed",
            checksum_sha256="a" * 64,
            file_size_bytes=10_000_000,
            fallback_tier=FallbackTier.PIXABAY_VIDEO,
        )

    def test_to_dict_from_dict_roundtrip(self) -> None:
        original = self._make_full()
        d = original.to_dict()
        restored = Candidate.from_dict(d)

        assert restored.id == original.id
        assert restored.kind == original.kind
        assert restored.width == original.width
        assert restored.duration == original.duration
        assert restored.tags == original.tags
        assert len(restored.renditions) == len(original.renditions)
        assert restored.renditions[0].url == original.renditions[0].url
        assert restored.checksum_sha256 == original.checksum_sha256
        assert restored.fallback_tier == original.fallback_tier

    def test_image_candidate_roundtrip(self) -> None:
        c = Candidate(
            id="pixabay_img:5555", source="pixabay", kind=FootageKind.IMAGE,
            url="https://cdn.pixabay.com/img.jpg", thumb_url="", width=2400, height=1600,
            duration=0.0,
        )
        restored = Candidate.from_dict(c.to_dict())
        assert restored.kind == FootageKind.IMAGE
        assert restored.duration == 0.0

    def test_card_candidate_roundtrip(self) -> None:
        c = Candidate(
            id="card:abc123", source="ffmpeg_card", kind=FootageKind.CARD,
            url="", thumb_url="", width=1920, height=1080, duration=5.0,
            license="Generated", fallback_tier=FallbackTier.TITLE_CARD,
        )
        restored = Candidate.from_dict(c.to_dict())
        assert restored.kind == FootageKind.CARD
        assert restored.fallback_tier == FallbackTier.TITLE_CARD


# ---------------------------------------------------------------------------
# FallbackWarning
# ---------------------------------------------------------------------------

class TestFallbackWarning:
    def test_to_str_includes_tier_and_reason(self) -> None:
        w = FallbackWarning(
            tier=FallbackTier.KEN_BURNS,
            reason="Pixabay returned no results",
            keyword="city skyline",
        )
        s = w.to_str()
        assert "Tier 2" in s
        assert "city skyline" in s
        assert "Pixabay returned no results" in s

    def test_details_optional(self) -> None:
        w = FallbackWarning(tier=FallbackTier.TITLE_CARD, reason="r", keyword="k")
        assert w.details == {}


# ---------------------------------------------------------------------------
# FootageKind
# ---------------------------------------------------------------------------

class TestFootageKind:
    def test_video_value(self) -> None:
        assert FootageKind.VIDEO.value == "video"

    def test_image_value(self) -> None:
        assert FootageKind.IMAGE.value == "image"

    def test_card_value(self) -> None:
        assert FootageKind.CARD.value == "card"

    def test_from_string(self) -> None:
        assert FootageKind("video") == FootageKind.VIDEO
        assert FootageKind("image") == FootageKind.IMAGE
