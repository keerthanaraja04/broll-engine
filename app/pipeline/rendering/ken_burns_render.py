"""
Ken Burns rendering for image clips.

Uses FFmpeg's zoompan filter to animate a still image with a slow zoom/pan.
Output is a normalised MP4 at the target resolution and fps.
"""

from __future__ import annotations

import os
import tempfile

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.rendering.ff_helpers import run_ffmpeg, target_dimensions

logger = get_logger(__name__)


async def render_ken_burns(
    image_path: str,
    duration: float,
    aspect_ratio: str,
    output_dir: str | None = None,
) -> str:
    """
    Animate a still image with a Ken Burns zoom effect.

    Args:
        image_path:   Path to the source image (JPEG/PNG).
        duration:     Target video duration in seconds.
        aspect_ratio: "16:9", "9:16", or "1:1".
        output_dir:   Directory for the output file.

    Returns:
        Path to the generated MP4.
    """
    s = get_settings()
    tgt_w, tgt_h = target_dimensions(aspect_ratio)
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(tmp_dir, f"{base}_kenburns.mp4")

    fps = s.RENDER_FPS
    total_frames = int(fps * duration)

    zoom_start = s.RENDER_KB_ZOOM_START
    zoom_end = s.RENDER_KB_ZOOM_END

    # zoompan: zoom from zoom_start to zoom_end over total_frames
    zoom_step = (zoom_end - zoom_start) / max(total_frames, 1)
    zoom_expr = f"min(zoom+{zoom_step:.6f},{zoom_end})"

    vf = (
        f"scale={tgt_w * 2}:{tgt_h * 2}:force_original_aspect_ratio=increase,"
        f"crop={tgt_w * 2}:{tgt_h * 2},"
        f"zoompan=z='{zoom_expr}'"
        f":x='iw/2-(iw/zoom/2)'"
        f":y='ih/2-(ih/zoom/2)'"
        f":d={total_frames}"
        f":s={tgt_w}x{tgt_h}"
        f":fps={fps},"
        f"format={s.RENDER_PIXEL_FORMAT},"
        f"setsar={s.RENDER_SAR}"
    )

    args = [
        "-loop", "1",
        "-i", image_path,
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
        "-vf", vf,
        "-c:v", s.MEDIA_VIDEO_CODEC,
        "-preset", s.MEDIA_PRESET,
        "-crf", str(s.MEDIA_CRF),
        "-t", f"{duration:.3f}",
        "-an",  # no audio from image
        "-movflags", "+faststart",
        out_path,
    ]

    await run_ffmpeg(args)
    logger.info(
        "ken_burns_rendered",
        extra={"image": image_path, "out": out_path,
               "duration": duration, "resolution": f"{tgt_w}x{tgt_h}"},
    )
    return out_path
