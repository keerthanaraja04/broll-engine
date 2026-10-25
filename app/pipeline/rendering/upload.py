"""
Upload rendering outputs (final video + SRT) to MinIO.
"""

from __future__ import annotations

import os

from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.rendering.models import RenderResult

logger = get_logger(__name__)


async def upload_render_outputs(result: RenderResult) -> RenderResult:
    """
    Upload the final video and SRT (if any) to MinIO.

    Updates result.final_video_object and result.srt_object in-place.
    """
    s = get_settings()
    storage = get_storage_client()

    # Upload final video
    if result.final_video_path and os.path.isfile(result.final_video_path):
        video_object = f"jobs/{result.job_id}/output_{result.aspect_ratio.replace(':', 'x')}.mp4"
        await storage.ensure_bucket(s.STORAGE_BUCKET_FINAL)
        await storage.upload_file(
            s.STORAGE_BUCKET_FINAL,
            video_object,
            result.final_video_path,
            content_type="video/mp4",
        )
        result.final_video_object = video_object
        logger.info(
            "final_video_uploaded",
            extra={"object": video_object, "bucket": s.STORAGE_BUCKET_FINAL},
        )

    # Upload SRT
    if result.srt_path and os.path.isfile(result.srt_path):
        srt_object = f"jobs/{result.job_id}/subtitles_{result.aspect_ratio.replace(':', 'x')}.srt"
        await storage.upload_file(
            s.STORAGE_BUCKET_FINAL,
            srt_object,
            result.srt_path,
            content_type="application/x-subrip",
        )
        result.srt_object = srt_object
        logger.info("srt_uploaded", extra={"object": srt_object})

    return result
