"""
End-to-end tests that run against the live docker-compose stack.

Prerequisites:
    docker compose up -d
    pytest tests/e2e -m e2e

The API_BASE_URL env var can override the default http://localhost:8000.
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
MOCK_BASE = os.getenv("MOCK_PROVIDERS_URL", "http://localhost:9999")

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def http() -> httpx.Client:
    with httpx.Client(base_url=API_BASE, timeout=15.0) as client:
        yield client


# ---------------------------------------------------------------------------
# Stack smoke tests
# ---------------------------------------------------------------------------
class TestLiveStack:
    def test_api_health(self, http: httpx.Client) -> None:
        r = http.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_api_ready(self, http: httpx.Client) -> None:
        r = http.get("/ready")
        body = r.json()
        # All deps should be healthy when compose stack is up
        assert body["status"] == "ok"
        assert body["checks"]["postgres"]["status"] == "ok"
        assert body["checks"]["redis"]["status"] == "ok"
        assert body["checks"]["storage"]["status"] == "ok"

    def test_correlation_id_round_trip(self, http: httpx.Client) -> None:
        custom_id = "e2e-test-correlation-id"
        r = http.get("/health", headers={"X-Correlation-Id": custom_id})
        assert r.headers.get("x-correlation-id") == custom_id

    def test_404_error_shape(self, http: httpx.Client) -> None:
        r = http.get("/no-such-route")
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "retryable" in error
        # UPPER_SNAKE_CASE check
        assert error["code"] == error["code"].upper()
        assert "_" in error["code"] or error["code"].isalpha()

    def test_mock_providers_health(self) -> None:
        r = httpx.get(f"{MOCK_BASE}/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_mock_groq_chat(self) -> None:
        r = httpx.post(
            f"{MOCK_BASE}/groq/v1/chat/completions",
            json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert "choices" in body
        content = body["choices"][0]["message"]["content"]
        import json
        keywords = json.loads(content)
        assert isinstance(keywords, list)

    def test_mock_pixabay_videos(self) -> None:
        r = httpx.get(
            f"{MOCK_BASE}/pixabay/videos/",
            params={"key": "mock-key", "q": "city skyline", "per_page": 5},
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert "hits" in body
        assert len(body["hits"]) == 5
        hit = body["hits"][0]
        assert "id" in hit
        assert "videos" in hit
        assert "large" in hit["videos"]

    def test_mock_groq_transcription(self) -> None:
        r = httpx.post(
            f"{MOCK_BASE}/groq/v1/audio/transcriptions",
            data={"model": "whisper-large-v3", "response_format": "verbose_json"},
            files={"file": ("audio.mp3", b"fake-audio", "audio/mpeg")},
            timeout=5,
        )
        assert r.status_code == 200
        body = r.json()
        assert "text" in body
        assert "segments" in body
        assert len(body["segments"]) > 0
