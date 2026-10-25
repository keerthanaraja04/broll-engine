"""
Shared pytest fixtures available to all test layers (unit, integration, e2e).
"""

from __future__ import annotations

import os
from typing import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# ---------------------------------------------------------------------------
# Force test environment before any app import reads settings
# ---------------------------------------------------------------------------
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-pixabay-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("STORAGE_ENDPOINT", "localhost:9000")
os.environ["DATABASE_URL"] = ""


# ---------------------------------------------------------------------------
# Settings override (clears the lru_cache so each test sees fresh settings)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# FastAPI test client (sync)
# ---------------------------------------------------------------------------
@pytest.fixture()
def client() -> Iterator[TestClient]:
    from app.main import create_app
    application = create_app()
    with TestClient(application, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# FastAPI async client
# ---------------------------------------------------------------------------
@pytest_asyncio.fixture()
async def async_client() -> AsyncIterator[AsyncClient]:
    from app.main import create_app
    application = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Mock Groq client
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_groq_client() -> Iterator[MagicMock]:
    with patch("app.adapters.groq_client.GroqClient") as mock_cls:
        instance = MagicMock()
        instance.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": '["city skyline", "people walking"]'}}]
        })
        instance.transcribe = AsyncMock(return_value={
            "text": "Mock transcription text",
            "segments": [{"start": 0.0, "end": 5.0, "text": "Mock transcription text"}],
        })
        mock_cls.return_value = instance
        yield instance


# ---------------------------------------------------------------------------
# Mock Pixabay client
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_pixabay_client() -> Iterator[MagicMock]:
    from app.adapters.pixabay_client import PixabayVideo
    with patch("app.adapters.pixabay_client.PixabayClient") as mock_cls:
        instance = MagicMock()
        instance.search_videos = AsyncMock(return_value=[
            PixabayVideo(
                id=1001,
                page_url="https://pixabay.com/videos/mock-1001/",
                tags="city, skyline",
                duration=10,
                views=5000,
                downloads=1000,
                likes=200,
                videos={
                    "large": {"url": "https://cdn.pixabay.com/mock-1001-large.mp4", "width": 1920, "height": 1080},
                    "medium": {"url": "https://cdn.pixabay.com/mock-1001-medium.mp4", "width": 1280, "height": 720},
                },
            )
        ])
        mock_cls.return_value = instance
        yield instance


# ---------------------------------------------------------------------------
# Mock storage client
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_storage_client() -> Iterator[MagicMock]:
    with patch("app.adapters.storage.StorageClient") as mock_cls:
        instance = MagicMock()
        instance.download_bytes = AsyncMock(return_value=b"fake-audio-bytes")
        instance.upload_bytes = AsyncMock(return_value="jobs/test-job/output.mp4")
        instance.upload_file = AsyncMock(return_value="jobs/test-job/output.mp4")
        instance.presigned_get_url = AsyncMock(return_value="https://storage.example.com/jobs/test-job/output.mp4")
        instance.ensure_bucket = AsyncMock()
        instance.ensure_all_buckets = AsyncMock()
        mock_cls.return_value = instance
        yield instance

# ---------------------------------------------------------------------------
# Load footage-specific fixtures (FakeRedis, mock adapters, etc.)
# ---------------------------------------------------------------------------
pytest_plugins = ["tests.fixtures.conftest_footage"]
