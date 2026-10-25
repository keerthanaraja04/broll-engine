"""
Normalise a clip to the target format:
  - Trim / loop / slow to match scene duration
  - Scale to target resolution (with smart reframe for non-16:9)
  - 30 fps, yuv420p, SAR 1
  - Brightness / colour match to a configurable target luma
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.rendering.ff_helpers import run_ffmpeg, target_dimensions

logger = get_logger(__name__)


def _build_trim_loop_slow_filter(
    clip_duration: float,
    scene_duration: float,
    needs_slow: bool,
    needs_loop: int,
) -> tuple[list[str], str]:
    """
    Return (input_args, filter_suffix) to make the clip fill scene_duration.

    For looping we use the -stream_loop input option.
    For slowing we use the setpts filter.
    Trimming is done with -t on the output.
    """
    input_args: list[str] = []
    filter_parts: list[str] = []

    if needs_loop > 0:
        input_args = ["-stream_loop", str(needs_loop)]

    if needs_slow:
        # Slow down: setpts scales presentation timestamps
        speed_factor = clip_duration / scene_duration
        filter_parts.append(f"setpts={speed_factor:.4f}*PTS")

    filter_suffix = ",".join(filter_parts) if filter_parts else "null"
    return input_args, filter_suffix


def _build_scale_filter(
    src_w: int, src_h: int,
    tgt_w: int, tgt_h: int,
    reframe_method: str,
    crop_x: float | None = None,
    crop_y: float | None = None,
    blur_strength: int = 20,
) -> str:
    """
    Build the video filter chain for scaling + reframing.

    For 16:9 target from 16:9 source: simple scale.
    For 9:16 / 1:1 targets: smart crop (if vlm_bbox/saliency) or
    blurred-background fit.
    """
    src_ratio = src_w / max(src_h, 1)
    tgt_ratio = tgt_w / max(tgt_h, 1)

    if abs(src_ratio - tgt_ratio) < 0.02:
        # Same aspect ratio — just scale
        return f"scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease," \
               f"pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2," \
               f"setsar={get_settings().RENDER_SAR}"

    if reframe_method == "blurred_bg":
        # Blurred background: scale to cover, then overlay a blurred copy
        return (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=increase,"
            f"boxblur={blur_strength}:{1}[bgblur];"
            f"[fg]scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease[fgscaled];"
            f"[bgblur][fgscaled]overlay=(W-w)/2:(H-h)/2,"
            f"setsar={get_settings().RENDER_SAR}"
        )

    # Crop-based reframe (center or VLM bbox)
    if reframe_method in ("vlm_bbox", "saliency", "center"):
        # Crop to target aspect ratio, then scale to exact target size
        if src_ratio > tgt_ratio:
            # Source is wider — crop horizontally
            crop_w = int(src_h * tgt_ratio)
            crop_h = src_h
        else:
            # Source is taller — crop vertically
            crop_w = src_w
            crop_h = int(src_w / tgt_ratio)

        # Apply VLM bbox offset if available
        if crop_x is not None and crop_y is not None:
            # crop_x, crop_y are normalised [0,1] center of subject
            cx = int(crop_x * src_w - crop_w / 2)
            cy = int(crop_y * src_h - crop_h / 2)
            cx = max(0, min(cx, src_w - crop_w))
            cy = max(0, min(cy, src_h - crop_h))
        else:
            cx = max(0, (src_w - crop_w) // 2)
            cy = max(0, (src_h - crop_h) // 2)

        return (
            f"crop={crop_w}:{crop_h}:{cx}:{cy},"
            f"scale={tgt_w}:{tgt_h},"
            f"setsar={get_settings().RENDER_SAR}"
        )

    # Fallback: scale with padding
    return (
        f"scale={tgt_w}:{tgt_h}:force_original_aspect_ratio=decrease,"
        f"pad={tgt_w}:{tgt_h}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar={get_settings().RENDER_SAR}"
    )


def _build_luma_filter(target_luma: float, current_luma: float) -> str:
    """
    Build an eq filter to shift brightness toward target_luma.

    eq filter: brightness range is [-1, 1], where 0 = no change.
    We compute a delta and map it to the eq brightness parameter.
    """
    delta = target_luma - current_luma
    # Clamp to [-0.5, 0.5] to avoid extreme shifts
    brightness = max(-0.5, min(0.5, delta * 2.0))
    contrast = 1.0  # leave contrast alone
    saturation = 1.0
    return f"eq=brightness={brightness:.3f}:contrast={contrast:.3f}:saturation={saturation:.3f}"


async def normalise_clip(
    clip_path: str,
    scene_duration: float,
    aspect_ratio: str,
    clip_duration: float = 0.0,
    src_width: int = 0,
    src_height: int = 0,
    needs_slow: bool = False,
    needs_loop: int = 0,
    reframe_method: str = "center",
    crop_x: float | None = None,
    crop_y: float | None = None,
    target_luma: float | None = None,
    current_luma: float | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Normalise *clip_path* to the target format and return the output path.
    """
    s = get_settings()
    tgt_w, tgt_h = target_dimensions(aspect_ratio)
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    base = os.path.splitext(os.path.basename(clip_path))[0]
    out_path = os.path.join(tmp_dir, f"{base}_norm.mp4")

    # Build trim/loop/slow
    input_args, pts_filter = _build_trim_loop_slow_filter(
        clip_duration or scene_duration, scene_duration, needs_slow, needs_loop,
    )

    # Build scale/reframe filter
    scale_filter = _build_scale_filter(
        src_width or tgt_w, src_height or tgt_h,
        tgt_w, tgt_h, reframe_method, crop_x, crop_y,
        s.RENDER_BLUR_BG_STRENGTH,
    )

    # Build luma filter if brightness matching enabled
    vf_parts = [pts_filter, scale_filter]
    if (
        s.RENDER_BRIGHTNESS_MATCH
        and target_luma is not None
        and current_luma is not None
        and abs(target_luma - current_luma) > s.RENDER_LUMA_TOLERANCE
    ):
        vf_parts.append(_build_luma_filter(target_luma, current_luma))

    # fps + pix_fmt
    vf_parts.append(f"fps={s.RENDER_FPS}")
    vf_chain = ",".join(vf_parts)

    args = list(input_args) + [
        "-i", clip_path,
        "-vf", vf_chain,
        "-t", f"{scene_duration:.3f}",
        "-c:v", s.MEDIA_VIDEO_CODEC,
        "-preset", s.MEDIA_PRESET,
        "-crf", str(s.MEDIA_CRF),
        "-pix_fmt", s.RENDER_PIXEL_FORMAT,
        "-an",   # no audio from clip — narration is mixed later
        "-movflags", "+faststart",
        out_path,
    ]

    await run_ffmpeg(args)
    logger.info(
        "clip_normalised",
        extra={"clip": clip_path, "out": out_path, "duration": scene_duration,
               "resolution": f"{tgt_w}x{tgt_h}"},
    )
    return out_path
