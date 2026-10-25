"""
Unit tests for the error taxonomy and response format.
"""

from __future__ import annotations

import pytest
from fastapi import status

from app.core.errors import (
    BrollError,
    GroqServiceError,
    JobNotFoundError,
    NotFoundError,
    PipelineError,
    PixabayServiceError,
    RateLimitError,
    ValidationError,
)


class TestErrorClasses:
    def test_base_error_defaults(self) -> None:
        err = BrollError("something broke")
        assert err.code == "INTERNAL_ERROR"
        assert err.http_status == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert err.retryable is False
        assert str(err) == "something broke"

    def test_not_found_error(self) -> None:
        err = NotFoundError("resource missing")
        assert err.code == "NOT_FOUND"
        assert err.http_status == 404
        assert err.retryable is False

    def test_rate_limit_is_retryable(self) -> None:
        err = RateLimitError()
        assert err.retryable is True
        assert err.http_status == 429

    def test_groq_service_error_is_retryable(self) -> None:
        err = GroqServiceError("groq down")
        assert err.retryable is True
        assert err.code == "GROQ_SERVICE_ERROR"

    def test_pixabay_service_error(self) -> None:
        err = PixabayServiceError("pixabay down", details={"body": "503"})
        assert err.code == "PIXABAY_SERVICE_ERROR"
        assert err.details == {"body": "503"}

    def test_job_not_found_inherits_not_found(self) -> None:
        err = JobNotFoundError("job abc not found")
        assert err.code == "JOB_NOT_FOUND"
        assert err.http_status == 404
        assert isinstance(err, NotFoundError)

    def test_pipeline_error_not_retryable(self) -> None:
        err = PipelineError("step failed")
        assert err.retryable is False

    def test_validation_error_status(self) -> None:
        err = ValidationError("bad input")
        assert err.http_status == 422


class TestErrorHandlers:
    """Test that FastAPI returns the correct JSON shape for each error type."""

    def test_health_returns_200(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_404_returns_error_shape(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.get("/nonexistent-route")
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert body["error"]["code"] == "NOT_FOUND"
        assert "retryable" in body["error"]
        assert isinstance(body["error"]["retryable"], bool)

    def test_method_not_allowed(self, client) -> None:  # type: ignore[no-untyped-def]
        r = client.post("/health")  # only GET is defined
        assert r.status_code == 405
        body = r.json()
        assert body["error"]["code"] == "METHOD_NOT_ALLOWED"
