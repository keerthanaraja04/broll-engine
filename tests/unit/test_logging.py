"""
Unit tests for structured logging context vars and formatter.
"""

from __future__ import annotations

import logging

from app.core.logging import (
    get_correlation_id,
    get_job_id,
    get_logger,
    set_correlation_id,
    set_job_id,
)


class TestContextVars:
    def test_job_id_default_is_none(self) -> None:
        set_job_id(None)
        assert get_job_id() is None

    def test_set_and_get_job_id(self) -> None:
        set_job_id("job-abc-123")
        assert get_job_id() == "job-abc-123"
        set_job_id(None)  # cleanup

    def test_set_and_get_correlation_id(self) -> None:
        set_correlation_id("corr-xyz-456")
        assert get_correlation_id() == "corr-xyz-456"
        set_correlation_id(None)  # cleanup

    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
