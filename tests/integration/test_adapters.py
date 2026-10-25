"""
Integration tests for external adapters (Groq, Pixabay).

These tests use httpx mocking so they run without live external services.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest


class TestGroqClientIntegration:
    @pytest.mark.asyncio
    async def test_chat_completion_returns_keywords(self) -> None:
        from app.adapters.groq_client import GroqClient

        mock_response = {
            "id": "chatcmpl-mock",
            "choices": [{"message": {"content": '["city skyline", "tech office"]'}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 50},
        }

        client = GroqClient()
        with patch.object(client, "_post", new=AsyncMock(return_value=mock_response)):
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "extract keywords"}]
            )

        assert result["choices"][0]["message"]["content"] == '["city skyline", "tech office"]'

    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self) -> None:
        from app.adapters.groq_client import GroqClient

        mock_response = {
            "text": "Hello world",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}],
        }

        client = GroqClient()
        with patch.object(client, "_multipart_post", new=AsyncMock(return_value=mock_response)):
            result = await client.transcribe(b"fake-audio")

        assert result["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_groq_rate_limit_raises(self) -> None:
        import httpx
        from app.adapters.groq_client import GroqClient
        from app.core.errors import RateLimitError

        client = GroqClient()
        mock_resp = httpx.Response(429, text="rate limited")

        with patch.object(client, "_post", side_effect=RateLimitError("rate limit")):
            with pytest.raises(RateLimitError):
                await client.chat_completion(messages=[])


class TestPixabayClientIntegration:
    @pytest.mark.asyncio
    async def test_search_videos_returns_list(self) -> None:
        from app.adapters.pixabay_client import PixabayClient

        mock_data = {
            "total": 100,
            "totalHits": 50,
            "hits": [
                {
                    "id": 1001,
                    "pageURL": "https://pixabay.com/videos/1001/",
                    "tags": "city, people",
                    "duration": 12,
                    "views": 1000,
                    "downloads": 500,
                    "likes": 100,
                    "user": "testuser",
                    "user_id": 42,
                    "picture_id": "1001",
                    "videos": {
                        "large": {"url": "https://cdn.pixabay.com/1001-large.mp4", "width": 1920, "height": 1080},
                        "medium": {"url": "https://cdn.pixabay.com/1001-medium.mp4", "width": 1280, "height": 720},
                    },
                }
            ],
        }

        client = PixabayClient()
        with patch.object(client, "_get", new=AsyncMock(return_value=mock_data)):
            videos = await client.search_videos("city skyline")

        assert len(videos) == 1
        assert videos[0].id == 1001
        assert videos[0].duration == 12
        assert videos[0].best_url(min_width=1280) is not None

    @pytest.mark.asyncio
    async def test_best_url_respects_min_width(self) -> None:
        from app.adapters.pixabay_client import PixabayVideo

        v = PixabayVideo(
            id=1,
            page_url="",
            tags="",
            duration=5,
            views=0,
            downloads=0,
            likes=0,
            videos={
                "large": {"url": "large.mp4", "width": 1920, "height": 1080},
                "small": {"url": "small.mp4", "width": 640, "height": 360},
            },
        )
        assert v.best_url(min_width=1280) == "large.mp4"
        assert v.best_url(min_width=2000) is not None  # falls back
