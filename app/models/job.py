"""
SQLAlchemy ORM model for a broll generation job.

Table: jobs
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class JobStage(str, enum.Enum):
    PENDING = "pending"
    UPLOADING = "uploading"
    TRANSCRIBING = "transcribing"
    EXTRACTING_KEYWORDS = "extracting_keywords"
    SEARCHING_VIDEOS = "searching_videos"
    FILTERING_CLIPS = "filtering_clips"
    ASSEMBLING_MEDIA = "assembling_media"
    UPLOADING_RESULT = "uploading_result"
    DONE = "done"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Job model
# ---------------------------------------------------------------------------
class Job(Base):
    __tablename__ = "jobs"

    # Primary key — caller-supplied or server-generated UUID
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # ------------------------------------------------------------------
    # Request payload (what the caller sent)
    # ------------------------------------------------------------------
    # Source type: "audio_url" | "audio_upload" | "script" | "scenes"
    input_type: Mapped[str] = mapped_column(String(32), nullable=False)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    # object key in the raw-audio storage bucket (for uploads)
    audio_object: Mapped[str | None] = mapped_column(Text, nullable=True)
    script: Mapped[str | None] = mapped_column(Text, nullable=True)
    # JSON: list of scene dicts
    scenes: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    voice: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # JSON: list of aspect-ratio strings e.g. ["16:9", "9:16"]
    aspect_ratios: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    style: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # JSON: list of terms to avoid
    avoid: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    # JSON: {"burn_in": bool, "srt": bool}
    captions: Mapped[dict[str, bool] | None] = mapped_column(JSON, nullable=True)
    callback_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------
    # Status tracking
    # ------------------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=JobStatus.QUEUED.value,
        index=True,
    )
    stage: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default=JobStage.PENDING.value,
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # JSON: list of warning strings accumulated during processing
    warnings: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    # JSON: {"video_url": str, "srt_url": str|null, ...}
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Duration of the final output in seconds
    output_duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True
    )
    # SHA-256 hash of the canonical request body (used to detect body changes)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # ------------------------------------------------------------------
    # Auth / ownership
    # ------------------------------------------------------------------
    # Identifies the API token owner; null for unauthenticated requests
    owner_token: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_jobs_idempotency_key"),
    )

    def __repr__(self) -> str:
        return f"<Job id={self.id} status={self.status} stage={self.stage}>"
