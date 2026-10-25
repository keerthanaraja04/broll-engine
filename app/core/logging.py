"""
Structured JSON logging setup.

Every log record carries:
  - timestamp  (ISO-8601)
  - level
  - logger     (name)
  - message
  - job_id     (from context var, populated by middleware / workers)
  - correlation_id  (from context var, populated by CorrelationIdMiddleware)
  - plus any extra kwargs passed to the logger call
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

from pythonjsonlogger import jsonlogger

# ---------------------------------------------------------------------------
# Context variables — set by middleware and worker tasks
# ---------------------------------------------------------------------------
_job_id_ctx: ContextVar[str | None] = ContextVar("job_id", default=None)
_correlation_id_ctx: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_job_id(job_id: str | None) -> None:
    _job_id_ctx.set(job_id)


def set_correlation_id(correlation_id: str | None) -> None:
    _correlation_id_ctx.set(correlation_id)


def get_job_id() -> str | None:
    return _job_id_ctx.get()


def get_correlation_id() -> str | None:
    return _correlation_id_ctx.get()


# ---------------------------------------------------------------------------
# Custom JSON formatter — injects context vars into every record
# ---------------------------------------------------------------------------
class BrollJsonFormatter(jsonlogger.JsonFormatter):
    """Extends python-json-logger with context-var injection."""

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)

        # Rename fields to match the desired schema
        log_record["timestamp"] = log_record.pop("asctime", None) or log_record.get("timestamp")
        log_record["level"] = log_record.pop("levelname", record.levelname)
        log_record["logger"] = record.name

        # Inject context
        job_id = get_job_id()
        correlation_id = get_correlation_id()
        if job_id:
            log_record["job_id"] = job_id
        if correlation_id:
            log_record["correlation_id"] = correlation_id

        # Drop keys we don't want at the top level
        for key in ("color_message",):
            log_record.pop(key, None)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Call once at application startup.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt:   "json" for structured output, "text" for human-readable dev output.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    # Remove any existing handlers (e.g., uvicorn defaults)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if fmt == "json":
        formatter = BrollJsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S.%fZ",
            rename_fields={"levelname": "level", "name": "logger"},
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "botocore", "boto3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — use in every module instead of logging.getLogger."""
    return logging.getLogger(name)
