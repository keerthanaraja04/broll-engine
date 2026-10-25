"""
Jobs API routes.

POST /api/v1/broll/jobs
    Accept JSON body or multipart/form-data (audio file upload).
    Validates input, handles idempotency, enqueues the pipeline.
    Returns 202 Accepted.

GET /api/v1/jobs/{job_id}
    Returns current status, progress 0-100, stage, warnings, and (once
    the job succeeds) presigned download links.
"""

from __future__ import annotations

import os
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.adapters.storage import get_storage_client
from app.api.deps import DbSession, OptionalToken, RequiredToken, get_job_or_404
from app.api.schemas.jobs import (
    JobCreateRequest,
    JobCreateResponse,
    JobResultLinks,
    JobStatusResponse,
    validate_audio_filename,
    validate_aspect_ratios,
)
from app.core.config import get_settings
from app.core.errors import (
    AudioTooLongError,
    ValidationError,
    WrongFileTypeError,
)
from app.core.idempotency import check_or_store
from app.core.logging import get_logger
from app.models.job import Job, JobStatus
from app.services.job_service import (
    create_job,
    get_job,
    get_job_by_idempotency_key,
    list_jobs,
)

logger = get_logger(__name__)
router = APIRouter()

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB hard cap before even reading


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_to_status_response(job: Job) -> JobStatusResponse:
    """Convert a Job ORM object to the API response schema."""
    result_links: JobResultLinks | None = None
    if job.result:
        result_links = JobResultLinks(
            video_url=job.result.get("video_url"),
            srt_url=job.result.get("srt_url"),
            thumbnail_url=job.result.get("thumbnail_url"),
            scenes=job.result.get("scenes"),
            average_relevance_score=job.result.get("average_relevance_score"),
        )

    return JobStatusResponse(
        job_id=job.id,
        status=JobStatus(job.status),
        stage=job.stage,  # type: ignore[arg-type]
        progress=job.progress,
        warnings=job.warnings or [],
        result=result_links,
        error_code=job.error_code,
        error_message=job.error_message,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        started_at=job.started_at.isoformat() if job.started_at else None,
        finished_at=job.finished_at.isoformat() if job.finished_at else None,
    )


async def _handle_idempotency(
    idempotency_key: str | None,
    body_hash: str,
    db: DbSession,
) -> JobCreateResponse | None:
    """
    Returns a JobCreateResponse if this is a valid replay, else None.
    Raises IdempotencyKeyConflictError on body mismatch.
    """
    if not idempotency_key:
        return None

    new_job_id = str(uuid.uuid4())
    result = await check_or_store(idempotency_key, body_hash, new_job_id)

    if result.is_replay and result.job_id:
        # Fetch the original job to confirm it still exists
        existing = await get_job_by_idempotency_key(db, idempotency_key)
        if existing:
            logger.info("idempotency_replay_returned", extra={"job_id": result.job_id})
            return JobCreateResponse(
                job_id=existing.id,
                status=existing.status,
                message="Replayed from idempotency cache",
            )

    return None  # Fresh key — caller should proceed with new_job_id


# ---------------------------------------------------------------------------
# POST /api/v1/broll/jobs  — JSON body
# ---------------------------------------------------------------------------

@router.post(
    "/broll/jobs",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreateResponse,
    summary="Submit a new b-roll generation job (JSON)",
    responses={
        202: {"description": "Job accepted"},
        409: {"description": "Idempotency key conflict"},
        415: {"description": "Unsupported audio file type"},
        422: {"description": "Validation error (bad ratio, empty script, audio too long)"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def submit_job_json(
    request: Request,
    body: JobCreateRequest,
    db: DbSession,
    token: OptionalToken,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    s = get_settings()

    # Validate aspect ratios (Pydantic Literal handles it, but belt-and-suspenders)
    validate_aspect_ratios([str(r) for r in body.aspect_ratios])

    body_hash = body.canonical_hash()

    # Idempotency check
    replay = await _handle_idempotency(idempotency_key, body_hash, db)
    if replay:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=replay.model_dump(mode="json"),
        )

    # Determine input type
    if body.audio_url:
        input_type = "audio_url"
    elif body.script:
        input_type = "script"
    else:
        input_type = "scenes"

    job = await create_job(
        db,
        input_type=input_type,
        audio_url=str(body.audio_url) if body.audio_url else None,
        script=body.script,
        scenes=[s.model_dump() for s in body.scenes] if body.scenes else None,
        voice=body.voice,
        aspect_ratios=[str(r) for r in body.aspect_ratios],
        style=body.style,
        avoid=body.avoid,
        captions=body.captions.model_dump(),
        callback_url=str(body.callback_url) if body.callback_url else None,
        idempotency_key=idempotency_key,
        request_hash=body_hash,
        owner_token=token,
    )

    logger.info(
        "job_created",
        extra={"job_id": str(job.id), "input_type": input_type, "token": token},
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=JobCreateResponse(job_id=job.id).model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# POST /api/v1/broll/jobs/upload  — multipart audio upload
# ---------------------------------------------------------------------------

@router.post(
    "/broll/jobs/upload",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobCreateResponse,
    summary="Submit a new b-roll generation job (multipart audio upload)",
    responses={
        202: {"description": "Job accepted"},
        409: {"description": "Idempotency key conflict"},
        413: {"description": "Audio file too large"},
        415: {"description": "Unsupported audio file type"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def submit_job_upload(
    db: DbSession,
    token: OptionalToken,
    audio: UploadFile = File(..., description="Audio file to transcribe"),
    voice: str | None = Form(None),
    aspect_ratios: str = Form("16:9", description="Comma-separated aspect ratios, e.g. '16:9,9:16'"),
    style: str | None = Form(None),
    avoid: str = Form("", description="Comma-separated terms to exclude"),
    burn_in: bool = Form(False),
    srt: bool = Form(False),
    callback_url: str | None = Form(None),
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> JSONResponse:
    s = get_settings()

    # Validate filename / extension
    filename = audio.filename or "upload.bin"
    validate_audio_filename(filename)

    # Read file with size cap
    data = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise WrongFileTypeError(
            f"Audio file exceeds the {MAX_UPLOAD_BYTES // 1_048_576} MB size limit"
        )

    # Parse aspect ratios
    ratios = [r.strip() for r in aspect_ratios.split(",") if r.strip()]
    validate_aspect_ratios(ratios)

    # Upload raw audio to object storage
    storage = get_storage_client()
    job_id = uuid.uuid4()
    object_name = f"jobs/{job_id}/audio/{filename}"
    await storage.upload_bytes(
        s.STORAGE_BUCKET_RAW,
        object_name,
        data,
        content_type=audio.content_type or "application/octet-stream",
    )

    # Build a deterministic hash for idempotency (based on filename + ratios + options)
    import hashlib, json as _json
    body_hash = hashlib.sha256(
        _json.dumps({
            "filename": filename,
            "size": len(data),
            "aspect_ratios": ratios,
            "voice": voice,
            "burn_in": burn_in,
            "srt": srt,
        }, sort_keys=True).encode()
    ).hexdigest()

    # Idempotency check
    replay = await _handle_idempotency(idempotency_key, body_hash, db)
    if replay:
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content=replay.model_dump(mode="json"),
        )

    avoid_list = [t.strip() for t in avoid.split(",") if t.strip()]

    job = await create_job(
        db,
        input_type="audio_upload",
        audio_object=object_name,
        voice=voice,
        aspect_ratios=ratios,
        style=style,
        avoid=avoid_list,
        captions={"burn_in": burn_in, "srt": srt},
        callback_url=callback_url,
        idempotency_key=idempotency_key,
        request_hash=body_hash,
        owner_token=token,
        job_id=job_id,
    )

    logger.info("job_upload_created", extra={"job_id": str(job.id), "upload_filename": filename})

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content=JobCreateResponse(job_id=job.id).model_dump(mode="json"),
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs  — list recent jobs
# ---------------------------------------------------------------------------

@router.get(
    "/jobs",
    summary="List recent b-roll generation jobs",
)
async def list_recent_jobs(
    db: DbSession,
    limit: int = 25,
    offset: int = 0,
) -> JSONResponse:
    jobs = await list_jobs(db, limit=limit, offset=offset)
    items = []
    for j in jobs:
        v_url = f"/api/v1/jobs/{j.id}/video" if j.status == "succeeded" else None
        s_url = f"/api/v1/jobs/{j.id}/srt" if j.status == "succeeded" else None
        items.append({
            "job_id": str(j.id),
            "status": j.status,
            "stage": j.stage,
            "progress": j.progress,
            "input_type": j.input_type,
            "script": j.script,
            "scenes": j.scenes,
            "voice": j.voice,
            "aspect_ratios": j.aspect_ratios,
            "style": j.style,
            "warnings": j.warnings or [],
            "result": {
                "video_url": v_url,
                "srt_url": s_url,
                "selected_clips": (j.result or {}).get("selected_clips", 0),
                "keywords": (j.result or {}).get("keywords", []),
            } if j.status == "succeeded" else None,
            "error_code": j.error_code,
            "error_message": j.error_message,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
        })
    return JSONResponse(status_code=status.HTTP_200_OK, content=items)


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status and result links",
    responses={
        200: {"description": "Job found"},
        404: {"description": "Job not found"},
    },
)
async def get_job_status(
    job_id: uuid.UUID,
    db: DbSession,
    token: OptionalToken,
) -> JSONResponse:
    job = await get_job_or_404(job_id, db)

    # If succeeded, ensure video_url and srt_url are present
    if job.status == JobStatus.SUCCEEDED.value and job.result:
        result_with_urls = dict(job.result)
        # Default to relative API endpoints for streaming / downloading
        if not result_with_urls.get("video_url"):
            result_with_urls["video_url"] = f"/api/v1/jobs/{job.id}/video"
        if not result_with_urls.get("srt_url"):
            result_with_urls["srt_url"] = f"/api/v1/jobs/{job.id}/srt"
        job.result = result_with_urls

    response = _job_to_status_response(job)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response.model_dump(mode="json"),
    )


@router.get(
    "/jobs/{job_id}/video",
    summary="Download or stream the generated video file",
)
async def get_job_video(job_id: uuid.UUID) -> FileResponse:
    media_base = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
    video_path = os.path.join(media_base, str(job_id), "video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found or not rendered yet")
    return FileResponse(video_path, media_type="video/mp4", filename=f"broll_{job_id}.mp4")


@router.get(
    "/jobs/{job_id}/srt",
    summary="Download the generated subtitles (SRT) file",
)
async def get_job_srt(job_id: uuid.UUID) -> FileResponse:
    media_base = os.environ.get("MEDIA_DIR", os.path.join(os.getcwd(), "media"))
    srt_path = os.path.join(media_base, str(job_id), "captions.srt")
    if not os.path.exists(srt_path):
        raise HTTPException(status_code=404, detail="SRT file not found")
    return FileResponse(srt_path, media_type="text/plain", filename=f"captions_{job_id}.srt")
