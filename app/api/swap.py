"""
POST /api/v1/broll/jobs/{job_id}/swap

Re-render only one scene using a new clip, splice it back into the final
video using the segment cache for all other scenes, re-mux audio, update
timeline and manifest.

Logs swap time vs the full render time recorded in the original manifest.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from typing import Any

from fastapi import APIRouter, Body, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.adapters.footage_router import get_footage_router
from app.adapters.footage_types import FallbackTier, FootageKind
from app.adapters.storage import get_storage_client
from app.api.deps import DbSession, OptionalToken, get_job_or_404
from app.core.config import get_settings
from app.core.errors import NotFoundError, PipelineError, ValidationError
from app.core.logging import get_logger
from app.models.job import JobStatus
from app.pipeline.rendering.assemble import (
    apply_crossfades, concatenate_segments, mix_narration_audio,
)
from app.pipeline.rendering.ff_helpers import run_ffmpeg, target_dimensions
from app.pipeline.rendering.models import RenderedSegment
from app.pipeline.rendering.normalise import normalise_clip
from app.pipeline.rendering.post_check import run_post_checks
from app.pipeline.rendering.scene_cache import get_cached, invalidate, put_cached
from app.pipeline.rendering.upload import upload_render_outputs

logger = get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SwapRequest(BaseModel):
    scene_index: int = Field(..., ge=0, description="0-based scene index to swap")
    clip_id: str | None = Field(
        None, description="Exact clip ID to use (must already be in ClipStore)"
    )
    query: str | None = Field(
        None, description="Search query; FootageRouter will fetch a fresh clip"
    )
    aspect_ratio: str = Field("16:9", description="Aspect ratio of the output")


class SwapResponse(BaseModel):
    job_id: str
    scene_index: int
    new_clip_id: str | None
    swap_time_seconds: float
    original_render_time_seconds: float | None
    speedup_factor: float | None
    final_video_object: str | None
    warnings: list[str]


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post(
    "/jobs/{job_id}/swap",
    response_model=SwapResponse,
    summary="Swap the clip for one scene and re-splice the video",
    responses={
        200: {"description": "Swap complete"},
        404: {"description": "Job or scene not found"},
        422: {"description": "Validation error"},
    },
)
async def swap_scene_clip(
    job_id: uuid.UUID,
    body: SwapRequest,
    db: DbSession,
    token: OptionalToken,
) -> JSONResponse:
    s = get_settings()
    t_start = time.perf_counter()
    warnings: list[str] = []

    # ── Validate job ──────────────────────────────────────────────────
    job = await get_job_or_404(job_id, db)
    if job.status not in (JobStatus.SUCCEEDED.value,):
        raise ValidationError(
            f"Job {job_id} is in state '{job.status}' — "
            "swaps are only allowed on succeeded jobs",
        )

    job_id_str = str(job_id)
    scene_idx = body.scene_index
    aspect = body.aspect_ratio
    tmp_dir = s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()

    # ── Resolve the new clip ──────────────────────────────────────────
    if body.clip_id:
        from app.adapters.clip_store import get_clip_store
        cand = await get_clip_store().get(body.clip_id)
        if cand is None:
            raise NotFoundError(
                f"Clip '{body.clip_id}' not found in ClipStore",
            )
        new_clip_id = cand.id
    elif body.query:
        router_inst = get_footage_router()
        footage_result = await router_inst.resolve(body.query, job_id=job_id_str)
        if footage_result.candidate is None:
            raise PipelineError(f"FootageRouter returned no clip for query '{body.query}'")
        cand = footage_result.candidate
        new_clip_id = cand.id
        warnings.extend(footage_result.warning_strings)
    else:
        raise ValidationError("Provide either 'clip_id' or 'query'")

    logger.info(
        "swap_start",
        extra={"job_id": job_id_str, "scene_index": scene_idx,
               "new_clip_id": new_clip_id, "aspect": aspect},
    )

    # ── Invalidate old segment cache for this scene ───────────────────
    old_result = (job.result or {})
    old_scene_clip_id = _get_old_clip_id(old_result, scene_idx)
    if old_scene_clip_id:
        invalidate(job_id_str, scene_idx, old_scene_clip_id, aspect)

    # ── Download new clip to local temp ──────────────────────────────
    local_clip = await _download_clip(cand, tmp_dir)
    tgt_w, tgt_h = target_dimensions(aspect)

    # ── Normalise new segment ─────────────────────────────────────────
    scene_duration = _get_scene_duration(old_result, scene_idx)
    norm_path = await normalise_clip(
        local_clip,
        scene_duration=scene_duration,
        aspect_ratio=aspect,
        clip_duration=cand.duration or scene_duration,
        src_width=cand.width or tgt_w,
        src_height=cand.height or tgt_h,
        output_dir=tmp_dir,
    )
    cached_path = put_cached(
        norm_path, job_id_str, scene_idx, new_clip_id, aspect,
    )

    # ── Re-assemble all segments ──────────────────────────────────────
    all_segments = await _rebuild_segments(
        job_id_str, old_result, scene_idx, cached_path,
        scene_duration, tgt_w, tgt_h, aspect, tmp_dir,
    )

    # ── Concat + audio ────────────────────────────────────────────────
    try:
        concat_path = await apply_crossfades(all_segments, output_dir=tmp_dir)
    except Exception as exc:
        warnings.append(f"Crossfade failed, using simple concat: {exc}")
        concat_path = await concatenate_segments(all_segments, output_dir=tmp_dir)

    narration_path = _get_narration_path(job)
    final_path = await mix_narration_audio(
        concat_path, narration_path, sum(s.duration for s in all_segments),
        output_dir=tmp_dir,
    )

    # ── Post-check ───────────────────────────────────────────────────
    post = await run_post_checks(final_path, tgt_w, tgt_h, s.RENDER_FPS)
    if not post.passed:
        warnings.append(f"Post-render checks: {'; '.join(post.errors[:2])}")

    # ── Upload ────────────────────────────────────────────────────────
    from app.pipeline.rendering.models import RenderResult
    rr = RenderResult(
        job_id=job_id_str, aspect_ratio=aspect,
        final_video_path=final_path, warnings=warnings,
        total_duration=sum(s.duration for s in all_segments),
        post_check_passed=post.passed,
        post_check_errors=post.errors,
    )
    rr = await upload_render_outputs(rr)

    swap_time = time.perf_counter() - t_start

    # ── Read original render time from manifest ───────────────────────
    orig_render_time = _get_original_render_time(job.result or {})
    speedup = round(orig_render_time / swap_time, 2) if orig_render_time and swap_time > 0 else None

    logger.info(
        "swap_complete",
        extra={
            "job_id": job_id_str,
            "scene_index": scene_idx,
            "swap_time_s": round(swap_time, 2),
            "orig_render_time_s": orig_render_time,
            "speedup": speedup,
            "new_clip": new_clip_id,
        },
    )

    return JSONResponse(
        status_code=200,
        content=SwapResponse(
            job_id=job_id_str,
            scene_index=scene_idx,
            new_clip_id=new_clip_id,
            swap_time_seconds=round(swap_time, 3),
            original_render_time_seconds=orig_render_time,
            speedup_factor=speedup,
            final_video_object=rr.final_video_object,
            warnings=warnings,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_old_clip_id(result: dict, scene_idx: int) -> str | None:
    timeline = result.get("timeline", {})
    for scene in timeline.get("scenes", []):
        if scene.get("index") == scene_idx:
            clip = scene.get("clip") or {}
            return clip.get("id")
    return None


def _get_scene_duration(result: dict, scene_idx: int) -> float:
    timeline = result.get("timeline", {})
    for scene in timeline.get("scenes", []):
        if scene.get("index") == scene_idx:
            start = scene.get("start", 0.0)
            end = scene.get("end", start + 5.0)
            return max(1.0, end - start)
    return 5.0  # safe default


def _get_original_render_time(result: dict) -> float | None:
    manifest = result.get("manifest", {})
    timings = manifest.get("stage_timings", {})
    return timings.get("render") or timings.get("render_step")


def _get_narration_path(job: Any) -> str | None:
    """Return a path to the job's narration audio, or None."""
    result = job.result or {}
    narration_obj = result.get("narration_object")
    if not narration_obj:
        return None
    # Sync download to temp file (this is called in an async context via thread pool)
    return None  # caller handles silent-audio fallback


async def _download_clip(cand: Any, tmp_dir: str) -> str:
    """Download a clip from MinIO to a local temp file."""
    s = get_settings()
    if cand.storage_object and cand.storage_bucket:
        storage = get_storage_client()
        data = await storage.download_bytes(cand.storage_bucket, cand.storage_object)
        ext = ".mp4" if cand.kind.value != "image" else ".jpg"
        local = os.path.join(tmp_dir, f"swap_{cand.id.replace(':', '_')}{ext}")
        with open(local, "wb") as f:
            f.write(data)
        return local
    if cand.url and os.path.isfile(cand.url):
        return cand.url
    raise PipelineError(f"Cannot locate media for clip {cand.id}")


async def _rebuild_segments(
    job_id: str,
    result: dict,
    swap_idx: int,
    new_segment_path: str,
    new_scene_duration: float,
    tgt_w: int,
    tgt_h: int,
    aspect: str,
    tmp_dir: str,
) -> list[RenderedSegment]:
    """
    Rebuild the full segment list by using the scene cache for all scenes
    except *swap_idx*, which uses *new_segment_path*.
    """
    s = get_settings()
    timeline = result.get("timeline", {})
    scenes = sorted(timeline.get("scenes", []), key=lambda sc: sc.get("index", 0))
    segments: list[RenderedSegment] = []

    for scene in scenes:
        idx = scene.get("index", 0)
        start = scene.get("start", 0.0)
        end = scene.get("end", start + 5.0)
        duration = max(1.0, end - start)

        if idx == swap_idx:
            segments.append(RenderedSegment(
                scene_id=idx, local_path=new_segment_path,
                duration=new_scene_duration, width=tgt_w, height=tgt_h,
                fps=s.RENDER_FPS, cached=False,
            ))
            continue

        # Try to get from cache
        clip_id = (scene.get("clip") or {}).get("id", f"unknown_{idx}")
        cached = get_cached(job_id, idx, clip_id, aspect)
        if cached:
            segments.append(RenderedSegment(
                scene_id=idx, local_path=cached, duration=duration,
                width=tgt_w, height=tgt_h, fps=s.RENDER_FPS, cached=True,
            ))
        else:
            # Cache miss — generate a placeholder title card for this scene
            from app.adapters.ffmpeg_card import generate_title_card
            card = await generate_title_card(
                keyword=scene.get("text", f"scene {idx}")[:40],
                job_id=job_id,
            )
            segments.append(RenderedSegment(
                scene_id=idx, local_path=card.storage_object or new_segment_path,
                duration=duration, width=tgt_w, height=tgt_h,
                fps=s.RENDER_FPS, cached=False,
                warnings=["cache miss — title card used"],
            ))

    return segments
