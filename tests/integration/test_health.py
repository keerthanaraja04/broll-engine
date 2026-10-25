"""
Integration tests for /health and /ready endpoints.

/health is purely in-process (no external deps) so it runs in standard CI.
/ready calls real external services — it is gated behind a pytest mark and
only runs when the full docker-compose stack is up.
"""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    def test_health_status_ok(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_response_shape(self, client) -> None:  # type: ignore[no-untyped-def]
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "service" in body
        assert "version" in body
        assert "uptime_seconds" in body
        assert isinstance(body["uptime_seconds"], float)

    def test_health_has_correlation_id_header(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/health")
        assert "x-correlation-id" in r.headers

    @pytest.mark.asyncio
    async def test_health_async(self, async_client) -> None:  # type: ignore[no-untyped-def]
        r = await async_client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


@pytest.mark.integration
class TestReadyEndpoint:
    """
    These tests require live postgres, redis, and minio.
    Run with:  pytest -m integration
    """

    def test_ready_returns_checks_dict(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/ready")
        # May be 200 (all ok) or 503 (deps unavailable in unit test env)
        assert r.status_code in (200, 503)
        body = r.json()
        assert "checks" in body
        assert "postgres" in body["checks"]
        assert "redis" in body["checks"]
        assert "storage" in body["checks"]

    def test_ready_check_structure(self, client) -> None:  # type: ignore[no-untyped-def]
        body = client.get("/ready").json()
        for dep in ("postgres", "redis", "storage"):
            check = body["checks"][dep]
            assert "status" in check
            assert check["status"] in ("ok", "error")
