"""
Pydantic request and response schemas for the jobs API.

POST /api/v1/broll/jobs  →  JobCreateRequest  / JobCreateResponse
GET  /api/v1/jobs/{id}   →  JobStatusResponse
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field, model_validator

from app.core.config import get_settings
from app.core.errors import (
    BadAspectRatioError,
    EmptyScriptError,
    WrongFileTypeError,
)
from app.models.job import JobStage, JobStatus

# ---------------------------------------------------------------------------
# Nested schemas
# ---------------------------------------------------------------------------

class CaptionOptions(BaseModel):
    """Controls subtitle/caption output."""
    burn_in: bool = Field(False, description="Burn subtitles directly into the video frames")
    srt: bool = Field(False, description="Also generate a .srt subtitle file")


class SceneInput(BaseModel):
    """A single scene when sending structured scenes instead of a transcript."""
    text: str = Field(..., min_length=1, description="Scene narration / subtitle text")
    duration_hint_seconds: float | None = Field(
        None, ge=0.5, le=120.0, description="Approximate duration for this scene"
    )
    duration: float | None = Field(
        None, ge=0.5, le=120.0, description="Alias for duration_hint_seconds"
    )
    keywords: list[str] = Field(default_factory=list, description="Search keywords for this scene")
    visual_hint: str | None = Field(None, description="Visual description or hint for b-roll search")


# ---------------------------------------------------------------------------
# Job create request  (JSON body)
# ---------------------------------------------------------------------------

class JobCreateRequest(BaseModel):
    """
    JSON body for POST /api/v1/broll/jobs.

    Exactly ONE of audio_url, script, or scenes must be supplied.
    (Multipart audio upload is handled separately in the route layer.)
    """

    # Input — exactly one required
    audio_url: AnyHttpUrl | None = Field(
        None, description="Publicly accessible URL of an audio file"
    )
    script: str | None = Field(
        None, min_length=1, description="Plain-text script to generate b-roll for"
    )
    scenes: list[SceneInput] | None = Field(
        None, min_length=1, description="Structured list of scenes"
    )

    # Presentation
    voice: str | None = Field(None, max_length=128, description="Voice/speaker identifier")
    aspect_ratios: list[Literal["16:9", "9:16", "1:1"]] = Field(
        default=["16:9"],
        min_length=1,
        description="One or more output aspect ratios",
    )
    style: str | None = Field(None, max_length=128, description="Visual style hint for video search")
    avoid: list[str] = Field(
        default_factory=list,
        description="Terms to exclude from video search results",
    )
    captions: CaptionOptions = Field(default_factory=CaptionOptions)

    # Delivery
    callback_url: AnyHttpUrl | None = Field(
        None,
        description="Webhook URL called when the job finishes (POST with job payload)",
    )

    @model_validator(mode="after")
    def _validate_input_source(self) -> "JobCreateRequest":
        sources = [
            v for v in (self.audio_url, self.script, self.scenes) if v is not None
        ]
        if len(sources) == 0:
            raise ValueError(
                "Provide exactly one of: audio_url, script, or scenes"
            )
        if len(sources) > 1:
            raise ValueError(
                "Only one of audio_url, script, or scenes may be provided"
            )

        # Empty script guard
        if self.script is not None and not self.script.strip():
            raise EmptyScriptError("Script must not be blank or whitespace-only")

        # Empty scenes guard
        if self.scenes is not None:
            texts = [s.text.strip() for s in self.scenes]
            if not any(texts):
                raise EmptyScriptError("At least one scene must have non-empty text")

        return self

    def canonical_hash(self) -> str:
        """
        Deterministic SHA-256 of the request body (used for idempotency conflict detection).
        """
        serialised = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            default=str,
        ).encode()
        return hashlib.sha256(serialised).hexdigest()


# ---------------------------------------------------------------------------
# Job create response  (202 Accepted)
# ---------------------------------------------------------------------------

class JobCreateResponse(BaseModel):
    job_id: UUID
    status: str = JobStatus.QUEUED.value
    message: str = "Job accepted and queued for processing"


# ---------------------------------------------------------------------------
# Job status response  (200 OK)
# ---------------------------------------------------------------------------

class JobResultLinks(BaseModel):
    """Presigned download URLs and scene clip details produced after a successful job."""
    video_url: str | None = None
    srt_url: str | None = None
    thumbnail_url: str | None = None
    scenes: list[dict[str, Any]] | None = None
    average_relevance_score: float | None = None


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: JobStatus
    stage: JobStage
    progress: int = Field(ge=0, le=100, description="Completion percentage 0–100")
    warnings: list[str] = Field(default_factory=list)
    result: JobResultLinks | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str  # ISO-8601
    updated_at: str  # ISO-8601
    started_at: str | None = None
    finished_at: str | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Audio upload validation helpers (called from the route, not as a schema)
# ---------------------------------------------------------------------------

def validate_audio_filename(filename: str) -> str:
    """
    Raise WrongFileTypeError if the extension is not in the allowed list.
    Returns the lower-cased extension.
    """
    s = get_settings()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in s.AUDIO_ALLOWED_EXTENSIONS:
        raise WrongFileTypeError(
            f"File type '.{ext}' is not accepted. "
            f"Allowed: {', '.join(s.AUDIO_ALLOWED_EXTENSIONS)}",
            details={"extension": ext, "allowed": s.AUDIO_ALLOWED_EXTENSIONS},
        )
    return ext


def validate_aspect_ratios(ratios: list[str]) -> None:
    """Raise BadAspectRatioError for any ratio not in the allowed set."""
    s = get_settings()
    bad = [r for r in ratios if r not in s.VALID_ASPECT_RATIOS]
    if bad:
        raise BadAspectRatioError(
            f"Invalid aspect ratio(s): {bad}. Allowed: {s.VALID_ASPECT_RATIOS}",
            details={"invalid": bad, "allowed": s.VALID_ASPECT_RATIOS},
        )
