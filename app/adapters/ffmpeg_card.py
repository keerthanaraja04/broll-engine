"""
ffmpeg_card — generate a title/data card as a video clip using FFmpeg.

This is the tier-4 (last-resort) fallback in the FootageRouter.  When all
other tiers fail for a keyword, a short colour-background card with the
keyword as text is synthesised locally with no external API calls.

Output
------
A short MP4 clip (default 5 s) at the configured resolution, with:
  - Solid background colour (configurable)
  - Keyword text centred with a semi-transparent box behind it
  - Silent audio track (so the clip mixes cleanly with narration)

The clip is uploaded to MinIO and a Candidate returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import tempfile
import uuid
from typing import Any

from app.adapters.footage_types import Candidate, DownloadedClip, FallbackTier, FootageKind, RenditionInfo
from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.errors import MediaProcessingError, StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal FFmpeg command builder
# ---------------------------------------------------------------------------

def _build_ffmpeg_cmd(
    output_path: str,
    text: str,
    width: int,
    height: int,
    duration: float,
    bg_color: str,
    font_size: int,
    font_color: str,
    font_dir: str,
) -> list[str]:
    """
    Build the ffmpeg command list for the title card.

    Uses the ``lavfi`` colour source and ``drawtext`` filter.
    Produces a silent H.264 MP4.

    Text escaping: single-quotes and backslashes in *text* are escaped for
    the drawtext filter syntax.
    """
    s = get_settings()

    # Escape text for ffmpeg drawtext: escape \ ' : and newlines
    safe_text = (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace(":", "\\:")
        .replace("\n", " ")
        [:120]  # cap at 120 chars
    )

    # Try to find a Noto font in the configured font directory
    font_file = ""
    if font_dir and os.path.isdir(font_dir):
        for candidate_font in ("NotoSans-Regular.ttf", "NotoSans.ttf", "DejaVuSans.ttf"):
            full = os.path.join(font_dir, candidate_font)
            if os.path.exists(full):
                font_file = full
                break

    drawtext_opts: list[str] = [
        f"text='{safe_text}'",
        f"fontsize={font_size}",
        f"fontcolor={font_color}",
        "x=(w-text_w)/2",
        "y=(h-text_h)/2",
        "box=1",
        "boxcolor=black@0.5",
        "boxborderw=20",
    ]
    if font_file:
        drawtext_opts.insert(0, f"fontfile={font_file}")

    joined_opts = "\\:".join(drawtext_opts)
    drawtext = f"drawtext={joined_opts}"

    # video filter: colour source → scale → drawtext → fps
    vf = (
        f"color=c={bg_color}:s={width}x{height}:d={duration},"
        f"fps=25,"
        f"{drawtext}"
    )

    cmd = [
        s.FFMPEG_BINARY,
        "-y",                         # overwrite
        "-f", "lavfi",
        "-i", f"color=c={bg_color}:s={width}x{height}:d={duration}:r=25",
        "-f", "lavfi",
        "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", drawtext,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "128k",
        "-t", str(duration),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_title_card(
    keyword: str,
    job_id: str | None = None,
    bucket: str | None = None,
) -> Candidate:
    """
    Generate a title-card MP4 for *keyword* and upload it to MinIO.

    Args:
        keyword: The search term / text to display on the card.
        job_id:  Optional job ID for the object key path.
        bucket:  MinIO bucket to upload to; defaults to STORAGE_BUCKET_PROCESSED.

    Returns:
        A Candidate with kind=CARD, storage_object set, and fallback_tier=4.

    Raises:
        MediaProcessingError: If FFmpeg fails.
        StorageError: If MinIO upload fails.
    """
    s = get_settings()
    target_bucket = bucket or s.STORAGE_BUCKET_PROCESSED

    width = s.TITLE_CARD_WIDTH
    height = s.TITLE_CARD_HEIGHT
    duration = s.TITLE_CARD_DURATION_SECONDS

    clip_uid = hashlib.sha256(keyword.encode()).hexdigest()[:16]
    object_key = f"footage/cards/{job_id or 'shared'}/{clip_uid}.mp4"

    loop = asyncio.get_running_loop()

    # Build and run ffmpeg in a thread pool to avoid blocking the event loop
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = _build_ffmpeg_cmd(
            output_path=tmp_path,
            text=keyword,
            width=width,
            height=height,
            duration=duration,
            bg_color=s.TITLE_CARD_BG_COLOR,
            font_size=s.TITLE_CARD_FONT_SIZE,
            font_color=s.TITLE_CARD_FONT_COLOR,
            font_dir=s.FONT_DIR,
        )

        def _run_ffmpeg() -> subprocess.CompletedProcess:  # type: ignore[type-arg]
            return subprocess.run(
                cmd,
                capture_output=True,
                timeout=60,
            )

        result = await loop.run_in_executor(None, _run_ffmpeg)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:600]
            raise MediaProcessingError(
                f"FFmpeg title card generation failed (rc={result.returncode})",
                details={"stderr": stderr, "keyword": keyword},
            )

        # Read generated file
        with open(tmp_path, "rb") as f:
            data = f.read()

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    checksum = hashlib.sha256(data).hexdigest()

    # Upload to MinIO
    storage = get_storage_client()
    await storage.ensure_bucket(target_bucket)
    await storage.upload_bytes(target_bucket, object_key, data, "video/mp4")

    candidate = Candidate(
        id=f"card:{clip_uid}",
        source="ffmpeg_card",
        kind=FootageKind.CARD,
        url="",                 # local — no remote URL
        thumb_url="",
        width=width,
        height=height,
        duration=duration,
        license="Generated",
        commercial_use=True,
        attribution="",
        page_url="",
        tags=[keyword.lower()],
        is_ai_generated=False,
        renditions=[
            RenditionInfo(
                url=object_key,
                width=width,
                height=height,
                size_bytes=len(data),
                label="card",
                content_type="video/mp4",
            )
        ],
        storage_object=object_key,
        storage_bucket=target_bucket,
        checksum_sha256=checksum,
        file_size_bytes=len(data),
        fallback_tier=FallbackTier.TITLE_CARD,
    )

    logger.info(
        "title_card_generated",
        extra={"keyword": keyword, "object": object_key, "bytes": len(data)},
    )
    return candidate


async def ken_burns_from_image(
    image_data: bytes,
    keyword: str,
    job_id: str | None = None,
    bucket: str | None = None,
) -> Candidate:
    """
    Animate a still image with a Ken Burns (zoompan) effect.

    Produces a short MP4 using FFmpeg's ``zoompan`` filter.
    Duration and zoom parameters come from KEN_BURNS_* config values.

    Args:
        image_data: Raw JPEG/PNG/WebP bytes of the source image.
        keyword:    Keyword used for naming and tagging.
        job_id:     Optional job ID for the object path.
        bucket:     Target MinIO bucket.

    Returns:
        A Candidate with kind=IMAGE (animated) and fallback_tier=KEN_BURNS.

    Raises:
        MediaProcessingError: If FFmpeg fails.
        StorageError: If MinIO upload fails.
    """
    s = get_settings()
    target_bucket = bucket or s.STORAGE_BUCKET_PROCESSED
    duration = s.KEN_BURNS_DURATION_SECONDS
    zoom_start = s.KEN_BURNS_ZOOM_START
    zoom_end = s.KEN_BURNS_ZOOM_END
    width = s.TITLE_CARD_WIDTH
    height = s.TITLE_CARD_HEIGHT
    fps = 25

    clip_uid = hashlib.sha256(image_data[:1024] + keyword.encode()).hexdigest()[:16]
    object_key = f"footage/ken_burns/{job_id or 'shared'}/{clip_uid}.mp4"

    loop = asyncio.get_running_loop()

    # Write image to temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_tmp:
        img_tmp.write(image_data)
        img_path = img_tmp.name

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_tmp:
        out_path = out_tmp.name

    try:
        total_frames = int(fps * duration)
        # zoompan: zoom from zoom_start to zoom_end over total_frames
        zoom_expr = (
            f"zoom='min(zoom+{(zoom_end - zoom_start) / total_frames:.6f},{zoom_end})'"
            f":x='iw/2-(iw/zoom/2)'"
            f":y='ih/2-(ih/zoom/2)'"
            f":d={total_frames}"
            f":s={width}x{height}"
            f":fps={fps}"
        )
        vf = f"zoompan={zoom_expr},format=yuv420p"

        cmd = [
            s.FFMPEG_BINARY,
            "-y",
            "-loop", "1",
            "-i", img_path,
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-c:a", "aac",
            "-b:a", "128k",
            "-t", str(duration),
            "-movflags", "+faststart",
            out_path,
        ]

        def _run() -> subprocess.CompletedProcess:  # type: ignore[type-arg]
            return subprocess.run(cmd, capture_output=True, timeout=120)

        result = await loop.run_in_executor(None, _run)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:600]
            raise MediaProcessingError(
                f"Ken Burns FFmpeg failed (rc={result.returncode})",
                details={"stderr": stderr, "keyword": keyword},
            )

        with open(out_path, "rb") as f:
            data = f.read()

    finally:
        for p in (img_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass

    checksum = hashlib.sha256(data).hexdigest()
    storage = get_storage_client()
    await storage.ensure_bucket(target_bucket)
    await storage.upload_bytes(target_bucket, object_key, data, "video/mp4")

    candidate = Candidate(
        id=f"kb:{clip_uid}",
        source="ken_burns",
        kind=FootageKind.IMAGE,
        url="",
        thumb_url="",
        width=width,
        height=height,
        duration=duration,
        license="Pixabay Content License",
        commercial_use=True,
        attribution="",
        page_url="",
        tags=[keyword.lower()],
        is_ai_generated=False,
        renditions=[
            RenditionInfo(
                url=object_key,
                width=width,
                height=height,
                size_bytes=len(data),
                label="ken_burns",
                content_type="video/mp4",
            )
        ],
        storage_object=object_key,
        storage_bucket=target_bucket,
        checksum_sha256=checksum,
        file_size_bytes=len(data),
        fallback_tier=FallbackTier.KEN_BURNS,
    )

    logger.info(
        "ken_burns_generated",
        extra={"keyword": keyword, "object": object_key, "bytes": len(data)},
    )
    return candidate
