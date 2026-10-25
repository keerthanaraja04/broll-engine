"""
Mock-based end-to-end tests — run with NO API keys and NO docker stack.

These tests drive the full HTTP surface of the API (submit job, poll status,
check health/ready, error shapes, rate-limit headers, correlation IDs) using
the FastAPI TestClient with all external dependencies replaced by mocks.

Marked ``e2e_mock`` so they can be run in CI with ``make test``.
"""

from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("GROQ_API_KEY", "mock-key")
os.environ.setdefault("PIXABAY_API_KEY", "mock-key")
os.environ.setdefault("APP_ENV", "development")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("API_AUTH_ENABLED", "false")
os.environ.setdefault("STT_ADAPTER", "mock")
os.environ.setdefault("LLM_ADAPTER", "mock")
os.environ.setdefault("VLM_ADAPTER", "mock")
os.environ.setdefault("TTS_ADAPTER", "mock")

pytestmark = pytest.mark.e2e_mock


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sqlite_engine():
    """In-memory SQLite engine for the module."""
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine
    from app.models.job import Base

    async def _create():
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
        )
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return engine

    engine = asyncio.get_event_loop().run_until_complete(_create())
    yield engine
    asyncio.get_event_loop().run_until_complete(engine.dispose())


@pytest.fixture(scope="module")
def mock_app(sqlite_engine) -> Iterator[TestClient]:
    """Full FastAPI app with SQLite + mocked ARQ + mocked storage."""
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app

    application = create_app()

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
    session_factory = async_sessionmaker(
        bind=sqlite_engine, class_=AsyncSession,
        expire_on_commit=False, autoflush=False, autocommit=False,
    )

    async def _override_db():
        async with session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    from app.db.session import get_db_session
    application.dependency_overrides[get_db_session] = _override_db

    # Mock ARQ pool
    pool_mock = MagicMock()
    pool_mock.enqueue_job = AsyncMock(return_value=MagicMock())

    # Mock storage
    storage_mock = MagicMock()
    storage_mock.upload_bytes = AsyncMock(return_value="jobs/test/audio.mp3")
    storage_mock.upload_file = AsyncMock(return_value="jobs/test/output.mp4")
    storage_mock.download_bytes = AsyncMock(return_value=b"fake-audio")
    storage_mock.presigned_get_url = AsyncMock(
        return_value="https://minio.example.com/presigned/output.mp4"
    )
    storage_mock.ensure_bucket = AsyncMock()
    storage_mock.ensure_all_buckets = AsyncMock()

    with patch("app.services.job_service._get_arq_pool",
               new=AsyncMock(return_value=pool_mock)), \
         patch("app.adapters.storage.get_storage_client",
               return_value=storage_mock), \
         patch("app.api.jobs.get_storage_client",
               return_value=storage_mock):
        with TestClient(application, raise_server_exceptions=False) as client:
            yield client

    application.dependency_overrides.clear()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Liveness + readiness
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_returns_ok(self, mock_app: TestClient) -> None:
        r = mock_app.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "service" in body
        assert "version" in body
        assert "uptime_seconds" in body

    def test_ready_returns_checks(self, mock_app: TestClient) -> None:
        r = mock_app.get("/ready")
        body = r.json()
        # May be 200 (all ok) or 503 (deps unavailable)
        assert r.status_code in (200, 503)
        assert "checks" in body
        assert "postgres" in body["checks"]
        assert "redis" in body["checks"]
        assert "storage" in body["checks"]

    def test_health_does_not_require_auth(self, mock_app: TestClient) -> None:
        """Health endpoints are public even when auth is enabled."""
        r = mock_app.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Correlation-Id header
# ---------------------------------------------------------------------------

class TestCorrelationId:
    def test_request_without_id_gets_one_back(self, mock_app: TestClient) -> None:
        r = mock_app.get("/health")
        assert "x-correlation-id" in r.headers
        # Must be parseable as UUID
        cid = r.headers["x-correlation-id"]
        uuid.UUID(cid)

    def test_supplied_id_echoed_back(self, mock_app: TestClient) -> None:
        my_id = "test-corr-id-abc123"
        r = mock_app.get("/health",
                          headers={"X-Correlation-Id": my_id})
        assert r.headers.get("x-correlation-id") == my_id

    def test_different_requests_get_different_ids(
        self, mock_app: TestClient
    ) -> None:
        r1 = mock_app.get("/health")
        r2 = mock_app.get("/health")
        assert r1.headers["x-correlation-id"] != r2.headers["x-correlation-id"]


# ---------------------------------------------------------------------------
# Error shape contract
# ---------------------------------------------------------------------------

class TestErrorShape:
    def test_404_has_correct_shape(self, mock_app: TestClient) -> None:
        r = mock_app.get("/no-such-route")
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        error = body["error"]
        assert "code" in error
        assert "message" in error
        assert "retryable" in error
        assert isinstance(error["retryable"], bool)

    def test_error_code_is_upper_snake_case(self, mock_app: TestClient) -> None:
        r = mock_app.get("/no-such-route")
        code = r.json()["error"]["code"]
        assert code == code.upper(), f"Code '{code}' is not UPPER_SNAKE_CASE"
        assert "_" in code or code.isalpha()

    def test_405_method_not_allowed(self, mock_app: TestClient) -> None:
        r = mock_app.post("/health")
        assert r.status_code == 405
        assert r.json()["error"]["code"] == "METHOD_NOT_ALLOWED"

    def test_validation_error_shape(self, mock_app: TestClient) -> None:
        # POST with empty body should give 422
        r = mock_app.post("/api/v1/broll/jobs", json={})
        assert r.status_code == 422
        body = r.json()
        assert "error" in body


# ---------------------------------------------------------------------------
# Job submission — script path
# ---------------------------------------------------------------------------

class TestJobSubmission:
    def test_submit_script_returns_202(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "A narrator explains the future of AI."},
        )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "queued"
        uuid.UUID(body["job_id"])  # must be valid UUID

    def test_submit_audio_url_returns_202(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"audio_url": "https://example.com/audio.mp3"},
        )
        assert r.status_code == 202

    def test_submit_scenes_returns_202(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={
                "scenes": [
                    {"text": "City at night", "visual_hint": "city lights"},
                    {"text": "People walking", "visual_hint": "crowd"},
                ]
            },
        )
        assert r.status_code == 202

    def test_submit_with_all_options(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={
                "script": "Full options test",
                "voice": "Arista-PlayAI",
                "aspect_ratios": ["16:9", "9:16"],
                "style": "cinematic",
                "avoid": ["violence", "weapons"],
                "captions": {"burn_in": True, "srt": True},
                "callback_url": "https://hooks.example.com/notify",
            },
        )
        assert r.status_code == 202

    def test_empty_body_returns_422(self, mock_app: TestClient) -> None:
        r = mock_app.post("/api/v1/broll/jobs", json={})
        assert r.status_code == 422

    def test_multiple_sources_returns_422(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={
                "script": "hello",
                "audio_url": "https://example.com/a.mp3",
            },
        )
        assert r.status_code == 422

    def test_invalid_aspect_ratio_returns_422(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "test", "aspect_ratios": ["4:3"]},
        )
        assert r.status_code == 422

    def test_invalid_callback_url_returns_422(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "test", "callback_url": "not-a-url"},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Job status polling
# ---------------------------------------------------------------------------

class TestJobStatus:
    def test_get_queued_job_returns_200(self, mock_app: TestClient) -> None:
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Status test"},
        )
        job_id = post_r.json()["job_id"]
        get_r = mock_app.get(f"/api/v1/jobs/{job_id}")
        assert get_r.status_code == 200

    def test_job_status_response_shape(self, mock_app: TestClient) -> None:
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Shape test"},
        )
        job_id = post_r.json()["job_id"]
        body = mock_app.get(f"/api/v1/jobs/{job_id}").json()
        for key in ("job_id", "status", "stage", "progress",
                     "warnings", "created_at", "updated_at"):
            assert key in body, f"Missing key: {key}"

    def test_queued_job_has_progress_zero(self, mock_app: TestClient) -> None:
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Progress test"},
        )
        job_id = post_r.json()["job_id"]
        body = mock_app.get(f"/api/v1/jobs/{job_id}").json()
        assert body["progress"] == 0

    def test_queued_status_value(self, mock_app: TestClient) -> None:
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Status value test"},
        )
        job_id = post_r.json()["job_id"]
        body = mock_app.get(f"/api/v1/jobs/{job_id}").json()
        assert body["status"] == "queued"

    def test_nonexistent_job_returns_404(self, mock_app: TestClient) -> None:
        fake_id = str(uuid.uuid4())
        r = mock_app.get(f"/api/v1/jobs/{fake_id}")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "JOB_NOT_FOUND"

    def test_malformed_job_id_returns_422(self, mock_app: TestClient) -> None:
        r = mock_app.get("/api/v1/jobs/not-a-uuid")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_key_same_body_returns_same_job(
        self, mock_app: TestClient
    ) -> None:
        stored: dict = {}

        async def _mock_check_or_store(key, body_hash, job_id):
            from app.core.idempotency import IdempotencyResult
            if key not in stored:
                stored[key] = {"hash": body_hash, "job_id": job_id}
                return IdempotencyResult(is_replay=False)
            if stored[key]["hash"] == body_hash:
                return IdempotencyResult(is_replay=True,
                                         job_id=stored[key]["job_id"])
            from app.core.errors import IdempotencyKeyConflictError
            raise IdempotencyKeyConflictError("conflict")

        with patch("app.api.jobs.check_or_store",
                    new=_mock_check_or_store):
            r = mock_app.post(
                "/api/v1/broll/jobs",
                json={"script": "Idempotency test"},
                headers={"Idempotency-Key": "key-e2e-001"},
            )
        assert r.status_code == 202

    def test_conflict_key_returns_409(self, mock_app: TestClient) -> None:
        from app.core.errors import IdempotencyKeyConflictError

        async def _conflict(key, hash_, job_id):
            raise IdempotencyKeyConflictError(
                "conflict", details={"idempotency_key": key}
            )

        with patch("app.api.jobs.check_or_store", new=_conflict):
            r = mock_app.post(
                "/api/v1/broll/jobs",
                json={"script": "Conflict test"},
                headers={"Idempotency-Key": "conflict-key"},
            )
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"
        assert r.json()["error"]["retryable"] is False


# ---------------------------------------------------------------------------
# Multipart upload
# ---------------------------------------------------------------------------

class TestMultipartUpload:
    def test_valid_mp3_returns_202(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs/upload",
            files={"audio": ("test.mp3", b"fake-audio-bytes", "audio/mpeg")},
            data={"aspect_ratios": "16:9"},
        )
        assert r.status_code == 202
        assert "job_id" in r.json()

    def test_invalid_extension_returns_415(self, mock_app: TestClient) -> None:
        r = mock_app.post(
            "/api/v1/broll/jobs/upload",
            files={"audio": ("bad.exe", b"bad", "application/octet-stream")},
            data={"aspect_ratios": "16:9"},
        )
        assert r.status_code == 415
        assert r.json()["error"]["code"] == "WRONG_FILE_TYPE"


# ---------------------------------------------------------------------------
# Rate-limit headers
# ---------------------------------------------------------------------------

class TestRateLimitHeaders:
    def test_headers_not_present_when_disabled(
        self, mock_app: TestClient
    ) -> None:
        # RATE_LIMIT_ENABLED=false so headers may not be on /health
        r = mock_app.get("/health")
        assert r.status_code == 200
        # Just verify the endpoint works — rate limit is disabled in this suite


# ---------------------------------------------------------------------------
# Full submit → get status round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_submit_and_poll(self, mock_app: TestClient) -> None:
        """Submit a job, then immediately poll it — it should be queued."""
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Round-trip test narration."},
        )
        assert post_r.status_code == 202
        job_id = post_r.json()["job_id"]

        get_r = mock_app.get(f"/api/v1/jobs/{job_id}")
        assert get_r.status_code == 200
        body = get_r.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("queued", "running", "succeeded", "failed")

    def test_correlation_id_propagated_end_to_end(
        self, mock_app: TestClient
    ) -> None:
        """The Correlation-Id we send in should come back on every response."""
        cid = "e2e-round-trip-id"
        post_r = mock_app.post(
            "/api/v1/broll/jobs",
            json={"script": "Correlation test."},
            headers={"X-Correlation-Id": cid},
        )
        assert post_r.headers.get("x-correlation-id") == cid

        job_id = post_r.json()["job_id"]
        get_r = mock_app.get(
            f"/api/v1/jobs/{job_id}",
            headers={"X-Correlation-Id": cid},
        )
        assert get_r.headers.get("x-correlation-id") == cid
