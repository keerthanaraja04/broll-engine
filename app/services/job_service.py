"""
Job service — all database operations and ARQ enqueueing for jobs.

Functions
---------
create_job()            — insert a new Job row + enqueue ARQ task
get_job()               — fetch Job by UUID
update_job_status()     — set status + optional stage/progress
update_job_progress()   — set progress + stage only (cheap polling update)
mark_job_succeeded()    — write result links, set status=SUCCEEDED
mark_job_failed()       — write error info, set status=FAILED
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from arq.connections import ArqRedis, RedisSettings, create_pool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.job import Job, JobStage, JobStatus

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ARQ pool (lazily initialised)
# ---------------------------------------------------------------------------
_arq_pool: ArqRedis | None = None


async def _get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        s = get_settings()
        _arq_pool = await create_pool(
            RedisSettings(
                host=s.REDIS_HOST,
                port=s.REDIS_PORT,
                database=s.REDIS_DB,
                password=s.REDIS_PASSWORD or None,
            )
        )
    return _arq_pool


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------
async def create_job(
    db: AsyncSession,
    *,
    input_type: str,
    audio_url: str | None = None,
    audio_object: str | None = None,
    script: str | None = None,
    scenes: list[dict[str, Any]] | None = None,
    voice: str | None = None,
    aspect_ratios: list[str] | None = None,
    style: str | None = None,
    avoid: list[str] | None = None,
    captions: dict[str, bool] | None = None,
    callback_url: str | None = None,
    idempotency_key: str | None = None,
    request_hash: str | None = None,
    owner_token: str | None = None,
    job_id: uuid.UUID | None = None,
) -> Job:
    """
    Persist a new Job row and enqueue an ARQ task.

    Returns the newly created Job ORM object (already committed).
    """
    s = get_settings()
    job = Job(
        id=job_id or uuid.uuid4(),
        input_type=input_type,
        audio_url=audio_url,
        audio_object=audio_object,
        script=script,
        scenes=scenes,
        voice=voice,
        aspect_ratios=aspect_ratios or ["16:9"],
        style=style,
        avoid=avoid or [],
        captions=captions or {"burn_in": False, "srt": False},
        callback_url=callback_url,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        owner_token=owner_token,
        status=JobStatus.QUEUED.value,
        stage=JobStage.PENDING.value,
        progress=0,
    )

    db.add(job)
    await db.flush()  # write to DB within the current transaction (commit happens in dep)

    # Build the payload passed to the worker
    payload: dict[str, Any] = {
        "input_type": input_type,
        "audio_url": audio_url,
        "audio_object": audio_object,
        "script": script,
        "scenes": scenes,
        "voice": voice,
        "aspect_ratios": aspect_ratios,
        "style": style,
        "avoid": avoid,
        "captions": captions,
        "callback_url": callback_url,
        "audio_bucket": s.STORAGE_BUCKET_RAW,
    }

    # Enqueue the background task
    try:
        import asyncio
        pool = await asyncio.wait_for(_get_arq_pool(), timeout=1.0)
        await asyncio.wait_for(
            pool.enqueue_job(
                "generate_broll",
                str(job.id),
                payload,
                _job_id=str(job.id),
                _queue_name=s.WORKER_QUEUE_NAME,
            ),
            timeout=1.0,
        )
        logger.info("job_enqueued", extra={"job_id": str(job.id), "input_type": input_type})
    except Exception as exc:
        logger.warning(
            "arq_pool_unavailable_processing_in_background",
            extra={"job_id": str(job.id), "error": str(exc)},
        )
        import asyncio
        from app.worker.tasks import generate_broll
        asyncio.create_task(generate_broll(None, str(job.id), payload))

    return job


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------
async def get_job(db: AsyncSession, job_id: uuid.UUID) -> Job | None:
    result = await db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def get_job_by_idempotency_key(db: AsyncSession, key: str) -> Job | None:
    result = await db.execute(select(Job).where(Job.idempotency_key == key))
    return result.scalar_one_or_none()


async def list_jobs(db: AsyncSession, limit: int = 20, offset: int = 0) -> list[Job]:
    """Fetch recent jobs ordered by created_at descending."""
    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).offset(offset).limit(limit)
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Update helpers (called by the worker)
# ---------------------------------------------------------------------------
async def update_job_status(
    db: AsyncSession,
    job_id: uuid.UUID,
    status: JobStatus,
    stage: JobStage | None = None,
    progress: int | None = None,
) -> Job | None:
    job = await get_job(db, job_id)
    if job is None:
        logger.warning("update_job_status_not_found", extra={"job_id": str(job_id)})
        return None

    job.status = status.value
    if stage is not None:
        job.stage = stage.value
    if progress is not None:
        job.progress = max(0, min(100, progress))

    if status == JobStatus.RUNNING and job.started_at is None:
        job.started_at = datetime.now(timezone.utc)
    if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
        job.finished_at = datetime.now(timezone.utc)
        job.progress = 100 if status == JobStatus.SUCCEEDED else job.progress

    await db.flush()
    return job


async def update_job_progress(
    db: AsyncSession,
    job_id: uuid.UUID,
    progress: int,
    stage: JobStage,
) -> None:
    job = await get_job(db, job_id)
    if job is None:
        return
    job.progress = max(0, min(100, progress))
    job.stage = stage.value
    await db.flush()


async def mark_job_succeeded(
    db: AsyncSession,
    job_id: uuid.UUID,
    result: dict[str, Any],
    warnings: list[str] | None = None,
    output_duration_seconds: float | None = None,
) -> Job | None:
    job = await get_job(db, job_id)
    if job is None:
        return None

    job.status = JobStatus.SUCCEEDED.value
    job.stage = JobStage.DONE.value
    job.progress = 100
    job.result = result
    job.warnings = warnings or []
    job.output_duration_seconds = output_duration_seconds
    job.finished_at = datetime.now(timezone.utc)
    if job.started_at is None:
        job.started_at = job.finished_at

    await db.flush()
    logger.info("job_succeeded", extra={"job_id": str(job_id)})
    return job


async def mark_job_failed(
    db: AsyncSession,
    job_id: uuid.UUID,
    error_code: str,
    error_message: str,
    warnings: list[str] | None = None,
) -> Job | None:
    job = await get_job(db, job_id)
    if job is None:
        return None

    job.status = JobStatus.FAILED.value
    job.stage = JobStage.ERROR.value
    job.error_code = error_code
    job.error_message = error_message
    job.warnings = warnings or []
    job.finished_at = datetime.now(timezone.utc)
    if job.started_at is None:
        job.started_at = job.finished_at

    await db.flush()
    logger.warning("job_failed", extra={"job_id": str(job_id), "error_code": error_code})
    return job
