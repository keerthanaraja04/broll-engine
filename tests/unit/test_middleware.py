"""
Unit tests for HTTP middleware.
"""

from __future__ import annotations

import uuid


class TestCorrelationIdMiddleware:
    def test_generates_correlation_id_when_absent(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/health")
        assert "x-correlation-id" in r.headers
        # Should be a valid UUID4
        corr_id = r.headers["x-correlation-id"]
        parsed = uuid.UUID(corr_id)
        assert str(parsed) == corr_id

    def test_echoes_provided_correlation_id(self, client) -> None:  # type: ignore[no-untyped-def]
        my_id = "my-custom-correlation-id-12345"
        r = client.get("/health", headers={"X-Correlation-Id": my_id})
        assert r.headers.get("x-correlation-id") == my_id

    def test_different_requests_get_different_ids(self, client) -> None:  # type: ignore[no-untyped-def]
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["x-correlation-id"] != r2.headers["x-correlation-id"]
