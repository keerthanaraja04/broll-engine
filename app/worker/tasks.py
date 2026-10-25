"""
ARQ worker task definitions.

generate_broll()
  — Full 6-step pipeline with DB progress updates at each step.
  — Fires the callback URL (if configured) on success or failure.
  — Updates job status in Postgres throughout execution.
"""

from __future__ import annotations

import uuid
from typing import Any

from app.core.logging import get_logger, set_job_id
from app.db.session import get_db
from app.models.job import JobStage, JobStatus
from app.pipeline.base import Pipeline, PipelineContext
from app.pipeline.steps import (
    ClipFilterStep,
    KeywordExtractionStep,
    MediaAssemblyStep,
    TranscribeStep,
    UploadResultStep,
    VideoSearchStep,
)
from app.services import job_service
from app.services.callback import fire_callback

logger = get_logger(__name__)

# Map pipeline step names → (stage, progress_percent)
_STEP_PROGRESS: dict[str, tuple[JobStage, int]] = {
    "TranscribeStep":        (JobStage.TRANSCRIBING,        15),
    "KeywordExtractionStep": (JobStage.EXTRACTING_KEYWORDS, 30),
    "VideoSearchStep":       (JobStage.SEARCHING_VIDEOS,    50),
    "ClipFilterStep":        (JobStage.FILTERING_CLIPS,     65),
    "MediaAssemblyStep":     (JobStage.ASSEMBLING_MEDIA,    85),
    "UploadResultStep":      (JobStage.UPLOADING_RESULT,    95),
}


async def _update_progress(job_id: str, stage: JobStage, progress: int) -> None:
    """Fire-and-forget DB progress update — errors are logged but not raised."""
    try:
        async with get_db() as db:
            await job_service.update_job_progress(db, uuid.UUID(job_id), progress, stage)
    except Exception as exc:
        logger.warning(
            "progress_update_failed",
            extra={"job_id": job_id, "stage": stage.value, "error": str(exc)},
        )


async def generate_broll(ctx: Any, job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    Full broll generation pipeline (ARQ task).

    Args:
        ctx:     ARQ context dict (redis connection, job metadata).
        job_id:  UUID string — also the DB primary key.
        payload: Job parameters from the API (audio_object, script, etc.).

    Returns:
        dict summarising the completed job.
    """
    set_job_id(job_id)
    logger.info("task_started", extra={"job_id": job_id})

    # Mark running in DB
    try:
        async with get_db() as db:
            await job_service.update_job_status(
                db, uuid.UUID(job_id), JobStatus.RUNNING, JobStage.PENDING, 0
            )
    except Exception as exc:
        logger.warning("task_status_update_failed", extra={"job_id": job_id, "error": str(exc)})

    # ------------------------------------------------------------------
    # Build pipeline with progress-reporting hooks
    # ------------------------------------------------------------------
    warnings: list[str] = []

    class ProgressPipeline(Pipeline):
        async def run(self, pipe_ctx: PipelineContext) -> PipelineContext:
            from app.pipeline.base import Step
            import time

            set_job_id(pipe_ctx.job_id)
            for step in self._steps:
                stage_info = _STEP_PROGRESS.get(step.name)
                if stage_info:
                    stage, pct = stage_info
                    await _update_progress(pipe_ctx.job_id, stage, pct)

                t0 = time.perf_counter()
                logger.info("pipeline_step_start", extra={"step": step.name})
                try:
                    pipe_ctx = await step.execute(pipe_ctx)
                except Exception as exc:
                    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                    logger.error(
                        "pipeline_step_failed",
                        extra={"step": step.name, "duration_ms": duration_ms, "error": str(exc)},
                    )
                    raise
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.info("pipeline_step_done", extra={"step": step.name, "duration_ms": duration_ms})

            return pipe_ctx

    pipeline = ProgressPipeline(
        steps=[
            TranscribeStep(),
            KeywordExtractionStep(),
            VideoSearchStep(),
            ClipFilterStep(),
            MediaAssemblyStep(),
            UploadResultStep(),
        ]
    )

    pipe_ctx = PipelineContext(job_id=job_id, payload=payload)

    try:
        final_ctx = await pipeline.run(pipe_ctx)
    except Exception as exc:
        error_code = getattr(exc, "code", "PIPELINE_ERROR")
        error_msg = str(exc)
        logger.error("task_failed", extra={"job_id": job_id, "error": error_msg})

        # Persist failure to DB
        try:
            async with get_db() as db:
                job = await job_service.mark_job_failed(
                    db, uuid.UUID(job_id), error_code, error_msg, warnings
                )
                callback_url = job.callback_url if job else payload.get("callback_url")
        except Exception as db_exc:
            logger.error("task_db_fail_update_error", extra={"error": str(db_exc)})
            callback_url = payload.get("callback_url")

        if callback_url:
            fire_callback(
                callback_url,
                job_id,
                "failed",
                error={"code": error_code, "message": error_msg},
            )

        return {"job_id": job_id, "status": "failed", "error": error_msg}

    # ------------------------------------------------------------------
    # Success
    # ------------------------------------------------------------------
    scene_details = final_ctx.get("scene_details") or []
    avg_relevance = (
        round(sum(float(s.get("relevance_score", 4.5)) for s in scene_details) / len(scene_details), 2)
        if scene_details
        else 4.5
    )
    result = {
        "video_url": final_ctx.get("output_url"),
        "video_object": final_ctx.get("output_object"),
        "srt_url": final_ctx.get("srt_url"),
        "selected_clips": len(final_ctx.get("selected_clips", [])),
        "keywords": final_ctx.get("keywords", []),
        "scenes": scene_details,
        "average_relevance_score": avg_relevance,
    }

    try:
        async with get_db() as db:
            job = await job_service.mark_job_succeeded(
                db, uuid.UUID(job_id), result, warnings=warnings
            )
            callback_url = job.callback_url if job else payload.get("callback_url")
    except Exception as exc:
        logger.error("task_db_success_update_error", extra={"job_id": job_id, "error": str(exc)})
        callback_url = payload.get("callback_url")

    if callback_url:
        fire_callback(callback_url, job_id, "succeeded", result=result)

    logger.info("task_completed", extra={"job_id": job_id, "clips": result["selected_clips"]})
    return {"job_id": job_id, "status": "succeeded", **result}
