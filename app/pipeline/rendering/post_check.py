"""
Post-render quality checks.

Uses ffprobe and ffmpeg filters to verify:
  1. Codec = H.264, fps = target, resolution = target, pix_fmt = yuv420p
  2. blackdetect — no sustained black frames
  3. freezedetect — no frozen/sustained-static frames
  4. Audio/video sync drift within tolerance

All checks produce structured results with pass/fail + details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.errors import PostRenderCheckError
from app.core.logging import get_logger
from app.pipeline.rendering.ff_helpers import probe_video, run_ffmpeg

logger = get_logger(__name__)


@dataclass
class PostCheckResult:
    """Aggregated result of all post-render checks."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


async def check_codec_fps_resolution(
    video_path: str,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
) -> tuple[bool, list[str]]:
    """Verify codec, fps, and resolution via ffprobe."""
    probe = await probe_video(video_path)
    errors: list[str] = []

    video_stream = next(
        (s for s in probe.get("streams", [])
         if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        errors.append("No video stream found in output")
        return False, errors

    codec = video_stream.get("codec_name", "")
    if codec != "h264":
        errors.append(f"Video codec is '{codec}', expected 'h264'")

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    if width != expected_width or height != expected_height:
        errors.append(
            f"Resolution is {width}x{height}, "
            f"expected {expected_width}x{expected_height}"
        )

    pix_fmt = video_stream.get("pix_fmt", "")
    if pix_fmt != "yuv420p":
        errors.append(f"Pixel format is '{pix_fmt}', expected 'yuv420p'")

    # Check fps from r_frame_rate or avg_frame_rate
    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = fps_str.split("/")
        actual_fps = int(num) / max(int(den), 1)
    except (ValueError, ZeroDivisionError):
        actual_fps = 0.0
    if abs(actual_fps - expected_fps) > 0.5:
        errors.append(
            f"FPS is {actual_fps:.1f}, expected {expected_fps}"
        )

    return len(errors) == 0, errors


async def check_blackdetect(video_path: str) -> tuple[bool, list[str]]:
    """Run ffmpeg blackdetect filter and check for sustained black frames."""
    s = get_settings()
    result = await run_ffmpeg([
        "-i", video_path,
        "-vf", (
            f"blackdetect=d={s.RENDER_BLACK_DETECT_MIN_DURATION}"
            f":pic_th={s.RENDER_BLACK_DETECT_THRESHOLD}"
        ),
        "-an", "-f", "null", "-",
    ])

    stderr = result.stderr.decode(errors="replace")
    black_segments = re.findall(
        r"blackdetect: black_start:\s*([\d.]+).*black_end:\s*([\d.]+).*black_duration:\s*([\d.]+)",
        stderr,
    )

    errors: list[str] = []
    for start, end, dur in black_segments:
        if float(dur) >= s.RENDER_BLACK_DETECT_MIN_DURATION:
            errors.append(
                f"Black frame detected: {float(start):.1f}s–{float(end):.1f}s "
                f"(duration {float(dur):.1f}s)"
            )

    return len(errors) == 0, errors


async def check_freezedetect(video_path: str) -> tuple[bool, list[str]]:
    """Run ffmpeg freezedetect filter and check for frozen frames."""
    s = get_settings()
    result = await run_ffmpeg([
        "-i", video_path,
        "-vf", (
            f"freezedetect=n={s.RENDER_FREEZE_DETECT_THRESHOLD}"
            f":d={s.RENDER_FREEZE_DETECT_MIN_DURATION}"
        ),
        "-an", "-f", "null", "-",
    ])

    stderr = result.stderr.decode(errors="replace")
    freeze_segments = re.findall(
        r"freezedetect: freeze_start:\s*([\d.]+).*freeze_end:\s*([\d.]+).*freeze_duration:\s*([\d.]+)",
        stderr,
    )

    errors: list[str] = []
    for start, end, dur in freeze_segments:
        if float(dur) >= s.RENDER_FREEZE_DETECT_MIN_DURATION:
            errors.append(
                f"Freeze detected: {float(start):.1f}s–{float(end):.1f}s "
                f"(duration {float(dur):.1f}s)"
            )

    return len(errors) == 0, errors


async def check_audio_video_drift(video_path: str) -> tuple[bool, list[str]]:
    """
    Check audio/video sync drift using ffprobe duration comparison.

    If the audio and video durations differ by more than RENDER_MAX_DRIFT_MS,
    flag it as an error.
    """
    s = get_settings()
    probe = await probe_video(video_path)

    audio_dur = 0.0
    video_dur = 0.0

    for stream in probe.get("streams", []):
        dur_str = stream.get("duration", "0")
        try:
            dur = float(dur_str)
        except (TypeError, ValueError):
            dur = 0.0
        if stream.get("codec_type") == "video":
            video_dur = dur
        elif stream.get("codec_type") == "audio":
            audio_dur = dur

    if video_dur == 0.0 or audio_dur == 0.0:
        return True, []  # can't check if one is missing

    drift_ms = abs(video_dur - audio_dur) * 1000
    errors: list[str] = []

    if drift_ms > s.RENDER_MAX_DRIFT_MS:
        errors.append(
            f"Audio/video drift: {drift_ms:.1f}ms "
            f"(video={video_dur:.3f}s, audio={audio_dur:.3f}s, "
            f"max={s.RENDER_MAX_DRIFT_MS}ms)"
        )

    return len(errors) == 0, errors


async def run_post_checks(
    video_path: str,
    expected_width: int,
    expected_height: int,
    expected_fps: int,
) -> PostCheckResult:
    """
    Run all post-render quality checks.

    Returns a PostCheckResult with aggregated pass/fail status.
    """
    s = get_settings()
    if not s.RENDER_POST_CHECK_ENABLED:
        return PostCheckResult(passed=True, warnings=["Post-render checks disabled"])

    result = PostCheckResult(passed=True)
    all_errors: list[str] = []

    # 1. Codec/fps/resolution
    ok, errors = await check_codec_fps_resolution(
        video_path, expected_width, expected_height, expected_fps,
    )
    all_errors.extend(errors)
    result.details["codec_check"] = ok

    # 2. Blackdetect
    ok, errors = await check_blackdetect(video_path)
    all_errors.extend(errors)
    result.details["blackdetect"] = ok

    # 3. Freezedetect
    ok, errors = await check_freezedetect(video_path)
    all_errors.extend(errors)
    result.details["freezedetect"] = ok

    # 4. Audio/video drift
    ok, errors = await check_audio_video_drift(video_path)
    all_errors.extend(errors)
    result.details["drift_check"] = ok

    result.errors = all_errors
    result.passed = len(all_errors) == 0

    logger.info(
        "post_checks_complete",
        extra={
            "passed": result.passed,
            "errors": len(all_errors),
            "video": video_path,
        },
    )
    return result
