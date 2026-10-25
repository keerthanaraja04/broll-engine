"""
Application-wide error taxonomy and FastAPI exception handlers.

Every HTTP error response is serialised as:

    {
      "error": {
        "code":      "UPPER_SNAKE_CASE",
        "message":   "Human-readable description",
        "retryable": true | false,
        "details":   { ... }   # optional, arbitrary context
      }
    }
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


# ---------------------------------------------------------------------------
# Error payload models
# ---------------------------------------------------------------------------
class ErrorBody(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Base application exception
# ---------------------------------------------------------------------------
class BrollError(Exception):
    """Base for all domain errors raised inside the application."""

    code: str = "INTERNAL_ERROR"
    http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


# ---------------------------------------------------------------------------
# Concrete error classes  (add more as the domain grows)
# ---------------------------------------------------------------------------
class InternalError(BrollError):
    code = "INTERNAL_ERROR"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


class NotFoundError(BrollError):
    code = "NOT_FOUND"
    http_status = status.HTTP_404_NOT_FOUND
    retryable = False


class ValidationError(BrollError):
    code = "VALIDATION_ERROR"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class ConflictError(BrollError):
    code = "CONFLICT"
    http_status = status.HTTP_409_CONFLICT
    retryable = False


class UnauthorizedError(BrollError):
    code = "UNAUTHORIZED"
    http_status = status.HTTP_401_UNAUTHORIZED
    retryable = False


class ForbiddenError(BrollError):
    code = "FORBIDDEN"
    http_status = status.HTTP_403_FORBIDDEN
    retryable = False


class RateLimitError(BrollError):
    code = "RATE_LIMIT_EXCEEDED"
    http_status = status.HTTP_429_TOO_MANY_REQUESTS
    retryable = True


class ExternalServiceError(BrollError):
    code = "EXTERNAL_SERVICE_ERROR"
    http_status = status.HTTP_502_BAD_GATEWAY
    retryable = True


class GroqServiceError(ExternalServiceError):
    code = "GROQ_SERVICE_ERROR"


class PixabayServiceError(ExternalServiceError):
    code = "PIXABAY_SERVICE_ERROR"


class StorageError(BrollError):
    code = "STORAGE_ERROR"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = True


class PipelineError(BrollError):
    code = "PIPELINE_ERROR"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


class JobNotFoundError(NotFoundError):
    code = "JOB_NOT_FOUND"


class MediaProcessingError(BrollError):
    code = "MEDIA_PROCESSING_ERROR"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


class ServiceUnavailableError(BrollError):
    code = "SERVICE_UNAVAILABLE"
    http_status = status.HTTP_503_SERVICE_UNAVAILABLE
    retryable = True


# ---------------------------------------------------------------------------
# Jobs-specific errors
# ---------------------------------------------------------------------------
class EmptyScriptError(BrollError):
    """Script payload was provided but contains no usable content."""
    code = "EMPTY_SCRIPT"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class WrongFileTypeError(BrollError):
    """Uploaded file has an extension / MIME type not accepted."""
    code = "WRONG_FILE_TYPE"
    http_status = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    retryable = False


class BadAspectRatioError(BrollError):
    """Requested aspect ratio is not in the allowed set (16:9, 9:16, 1:1)."""
    code = "BAD_ASPECT_RATIO"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class AudioTooLongError(BrollError):
    """Audio file exceeds the maximum allowed duration (10 minutes)."""
    code = "AUDIO_TOO_LONG"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class IdempotencyKeyConflictError(BrollError):
    """
    Same Idempotency-Key was submitted with a *different* request body.
    The original response is not replayed; the caller must use a new key.
    """
    code = "IDEMPOTENCY_KEY_CONFLICT"
    http_status = status.HTTP_409_CONFLICT
    retryable = False


# ---------------------------------------------------------------------------
# Footage / download errors
# ---------------------------------------------------------------------------

class FootageUnavailableError(BrollError):
    """No footage could be found or generated for a keyword after all fallbacks."""
    code = "FOOTAGE_UNAVAILABLE"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class CorruptedDownloadError(BrollError):
    """Downloaded file failed checksum or ffprobe validation."""
    code = "CORRUPTED_DOWNLOAD"
    http_status = status.HTTP_502_BAD_GATEWAY
    retryable = True


class DownloadSizeLimitError(BrollError):
    """Remote file exceeds the configured maximum download size."""
    code = "DOWNLOAD_SIZE_LIMIT"
    http_status = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    retryable = False


class FfprobeValidationError(BrollError):
    """ffprobe reported the downloaded file has no valid video stream."""
    code = "FFPROBE_VALIDATION_ERROR"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = True


class PixabayAuthError(BrollError):
    """Pixabay API key is missing or invalid (HTTP 400 with auth error body)."""
    code = "PIXABAY_AUTH_ERROR"
    http_status = status.HTTP_401_UNAUTHORIZED
    retryable = False


class FootageRouterExhaustedError(BrollError):
    """All 4 tiers of the FootageRouter failed for a keyword."""
    code = "FOOTAGE_ROUTER_EXHAUSTED"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


# ---------------------------------------------------------------------------
# Input stage errors
# ---------------------------------------------------------------------------

class AudioSilentError(BrollError):
    """
    Audio file contains only silence or near-silence throughout.
    The silencedetect filter found no audible content.
    """
    code = "AUDIO_SILENT"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class AudioNoSpeechError(BrollError):
    """
    STT transcription returned a result but with extremely low confidence,
    or the transcript is effectively empty (no recognisable words).
    """
    code = "AUDIO_NO_SPEECH"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


# ---------------------------------------------------------------------------
# Rendering errors
# ---------------------------------------------------------------------------

class RenderError(BrollError):
    """Base for all rendering-stage errors."""
    code = "RENDER_ERROR"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


class PostRenderCheckError(BrollError):
    """A post-render quality check failed (black frames, freeze, drift)."""
    code = "POST_RENDER_CHECK_FAILED"
    http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    retryable = False


# ---------------------------------------------------------------------------
# Licence / safety gate errors
# ---------------------------------------------------------------------------

class LicenceGateError(BrollError):
    """A chosen clip lacks licence or attribution, or is flagged watermarked."""
    code = "LICENCE_GATE_FAILED"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


class SafetyViolationError(BrollError):
    """A safety rule was violated (named person face search, graphic content, etc.)."""
    code = "SAFETY_VIOLATION"
    http_status = status.HTTP_422_UNPROCESSABLE_ENTITY
    retryable = False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _error_response(
    code: str,
    message: str,
    retryable: bool,
    http_status: int,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    from fastapi.encoders import jsonable_encoder

    safe_details = jsonable_encoder(details) if details is not None else None
    payload = ErrorResponse(
        error=ErrorBody(code=code, message=message, retryable=retryable, details=safe_details)
    )
    return JSONResponse(status_code=http_status, content=jsonable_encoder(payload.model_dump(exclude_none=True)))


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
async def broll_error_handler(request: Request, exc: BrollError) -> JSONResponse:
    return _error_response(
        code=exc.code,
        message=exc.message,
        retryable=exc.retryable,
        http_status=exc.http_status,
        details=exc.details,
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    # Map common HTTP status codes to UPPER_SNAKE_CASE codes
    _code_map: dict[int, tuple[str, bool]] = {
        400: ("BAD_REQUEST", False),
        401: ("UNAUTHORIZED", False),
        403: ("FORBIDDEN", False),
        404: ("NOT_FOUND", False),
        405: ("METHOD_NOT_ALLOWED", False),
        409: ("CONFLICT", False),
        422: ("UNPROCESSABLE_ENTITY", False),
        429: ("RATE_LIMIT_EXCEEDED", True),
        500: ("INTERNAL_ERROR", False),
        502: ("BAD_GATEWAY", True),
        503: ("SERVICE_UNAVAILABLE", True),
        504: ("GATEWAY_TIMEOUT", True),
    }
    code, retryable = _code_map.get(exc.status_code, ("HTTP_ERROR", False))
    return _error_response(
        code=code,
        message=str(exc.detail),
        retryable=retryable,
        http_status=exc.status_code,
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    from fastapi.encoders import jsonable_encoder
    details: dict[str, Any] = {"errors": jsonable_encoder(exc.errors())}
    return _error_response(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        retryable=False,
        http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
        details=details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    import logging
    logging.getLogger("app.core.errors").exception("unhandled_exception: %s", exc)
    return _error_response(
        code="INTERNAL_ERROR",
        message="An unexpected internal error occurred",
        retryable=False,
        http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------
def register_error_handlers(app: FastAPI) -> None:
    """Attach all error handlers to the FastAPI application."""
    app.add_exception_handler(BrollError, broll_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)  # type: ignore[arg-type]
