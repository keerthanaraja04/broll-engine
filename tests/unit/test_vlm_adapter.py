"""
Unit tests for VLMAdapter — scoring, batching, mock determinism.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("VLM_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")

from tests.fixtures import CLIP_CITY, CLIP_NATURE, CLIP_ABSTRACT, image_bytes


def _load_images() -> list[bytes]:
    return [image_bytes(CLIP_CITY), image_bytes(CLIP_NATURE),
            image_bytes(CLIP_ABSTRACT)]


class TestMockVLMAdapter:
    @pytest.mark.asyncio
    async def test_one_score_per_image(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        scores = await MockVLMAdapter().score(_load_images(), "city")
        assert len(scores) == 3

    @pytest.mark.asyncio
    async def test_score_in_range_1_to_5(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        for sc in await MockVLMAdapter().score(_load_images(), "nature"):
            assert 1 <= sc.score <= 5

    @pytest.mark.asyncio
    async def test_image_index_correct(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        scores = await MockVLMAdapter().score(_load_images(), "test")
        for i, sc in enumerate(scores):
            assert sc.image_index == i

    @pytest.mark.asyncio
    async def test_deterministic_same_image(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        img = image_bytes(CLIP_CITY)
        s1 = (await MockVLMAdapter().score([img], "text"))[0]
        s2 = (await MockVLMAdapter().score([img], "text"))[0]
        assert s1.score == s2.score

    @pytest.mark.asyncio
    async def test_empty_images_returns_empty(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        assert await MockVLMAdapter().score([], "text") == []

    @pytest.mark.asyncio
    async def test_bbox_normalised(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        sc = (await MockVLMAdapter().score(
            [image_bytes(CLIP_CITY)], "city"))[0]
        bbox = sc.subject_bbox
        assert 0.0 <= bbox.x1 <= 1.0
        assert 0.0 <= bbox.y1 <= 1.0
        assert 0.0 <= bbox.x2 <= 1.0
        assert 0.0 <= bbox.y2 <= 1.0

    @pytest.mark.asyncio
    async def test_reason_is_string(self) -> None:
        from app.adapters.vlm_adapter import MockVLMAdapter
        sc = (await MockVLMAdapter().score(
            [image_bytes(CLIP_NATURE)], "forest"))[0]
        assert isinstance(sc.reason, str)
        assert len(sc.reason) > 0


class TestVLMFactory:
    def test_mock_selector(self) -> None:
        from app.adapters.vlm_adapter import (
            MockVLMAdapter, get_vlm_adapter, reset_vlm_adapter)
        reset_vlm_adapter()
        assert isinstance(get_vlm_adapter(), MockVLMAdapter)
        reset_vlm_adapter()

    def test_singleton(self) -> None:
        from app.adapters.vlm_adapter import get_vlm_adapter, reset_vlm_adapter
        reset_vlm_adapter()
        assert get_vlm_adapter() is get_vlm_adapter()
        reset_vlm_adapter()


class TestScoreResponseParser:
    def test_parse_valid(self) -> None:
        from app.adapters.vlm_adapter import _parse_score_response
        body = {"choices": [{"message": {"content":
            '{"score": 4, "reason": "Good", '
            '"subject_bbox": {"x1":0.1,"y1":0.1,"x2":0.8,"y2":0.8}, '
            '"flags": ["text_overlay"]}'}}]}
        sc = _parse_score_response(body, index=2)
        assert sc.score == 4
        assert sc.reason == "Good"
        assert "text_overlay" in sc.flags
        assert sc.image_index == 2

    def test_score_clamped_1_to_5(self) -> None:
        from app.adapters.vlm_adapter import _parse_score_response
        body = {"choices": [{"message": {"content":
            '{"score": 99, "reason": "x"}'}}]}
        assert _parse_score_response(body, 0).score == 5

        body2 = {"choices": [{"message": {"content":
            '{"score": -1, "reason": "x"}'}}]}
        assert _parse_score_response(body2, 0).score == 1

    def test_embedded_json_in_text(self) -> None:
        from app.adapters.vlm_adapter import _parse_score_response
        body = {"choices": [{"message": {"content":
            'Answer: {"score": 3, "reason": "ok"} end'}}]}
        assert _parse_score_response(body, 0).score == 3


class TestVLMBatching:
    @pytest.mark.asyncio
    async def test_each_image_scored_independently(self) -> None:
        from app.adapters.vlm_adapter import GroqVLMAdapter
        from app.adapters.base import BoundingBox, VLMScore
        call_count = 0

        async def _fake(self_ref: Any, img: bytes, text: str,
                        idx: int) -> VLMScore:
            nonlocal call_count
            call_count += 1
            return VLMScore(score=3, reason="ok",
                            subject_bbox=BoundingBox(0, 0, 1, 1),
                            flags=[], image_index=idx)

        with patch.object(GroqVLMAdapter, "_score_one", _fake):
            scores = await GroqVLMAdapter().score(_load_images(), "test")
            assert call_count == 3
            assert len(scores) == 3
