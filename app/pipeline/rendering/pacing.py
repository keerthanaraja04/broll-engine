"""
Pacing guard — analyses a list of RenderedSegments and flags violations:

1. Too many consecutive scenes from the same source (all title cards, etc.)
2. Insufficient duration variety (all scenes near the same length)
3. Returns structured warnings that bubble up into the RenderResult.

Global colour pass — per-segment luma measurement + correction so the
assembled video has consistent visual brightness across all clips.

Both run purely in Python / FFmpeg with no external API calls.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.rendering.ff_helpers import run_ffmpeg, run_ffprobe
from app.pipeline.rendering.models import RenderedSegment

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pacing guard
# ---------------------------------------------------------------------------

@dataclass
class PacingReport:
    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def check_pacing(segments: list[RenderedSegment]) -> PacingReport:
    """
    Analyse *segments* for pacing violations.

    Returns a PacingReport — never raises.
    """
    s = get_settings()
    report = PacingReport()

    if not segments:
        return report

    # ── 1. Consecutive same-source check ─────────────────────────────
    max_consec = s.PACING_MAX_CONSECUTIVE_SAME_SOURCE
    current_source = segments[0].reframe_method  # use reframe_method as proxy for source type
    run_length = 1
    max_seen = 1
    for seg in segments[1:]:
        if seg.reframe_method == current_source:
            run_length += 1
            max_seen = max(max_seen, run_length)
        else:
            current_source = seg.reframe_method
            run_length = 1

    if max_seen > max_consec:
        msg = (
            f"Pacing: {max_seen} consecutive scenes with the same clip type "
            f"(max allowed: {max_consec}) — consider adding more variety"
        )
        report.warnings.append(msg)
        report.passed = False

    # ── 2. Duration variety check ─────────────────────────────────────
    durations = [seg.duration for seg in segments if seg.duration > 0]
    if len(durations) >= 3:
        mean_dur = statistics.mean(durations)
        stdev_dur = statistics.stdev(durations)
        cv = stdev_dur / mean_dur if mean_dur > 0 else 0.0
        target_cv = s.PACING_TARGET_CV
        report.stats["duration_cv"] = round(cv, 3)
        report.stats["mean_duration"] = round(mean_dur, 2)
        if cv < target_cv * 0.3 and len(durations) > 5:
            report.warnings.append(
                f"Pacing: low duration variety (CV={cv:.2f}, target≥{target_cv:.2f}) "
                f"— all scenes are nearly the same length"
            )

    report.stats["max_consecutive"] = max_seen
    report.stats["segment_count"] = len(segments)

    logger.info(
        "pacing_check",
        extra={"passed": report.passed, "warnings": len(report.warnings), **report.stats},
    )
    return report


# ---------------------------------------------------------------------------
# Per-segment luma measurement (ffprobe + signalstats)
# ---------------------------------------------------------------------------

async def measure_luma(segment_path: str) -> float | None:
    """
    Measure the average luma of a video segment using ffmpeg signalstats.

    Returns a float in [0, 1] (where 0=black, 1=white), or None on error.
    """
    try:
        import asyncio
        import subprocess
        s = get_settings()
        cmd = [
            s.FFMPEG_BINARY, "-i", segment_path,
            "-vf", "signalstats=stat=tout,metadata=print",
            "-f", "null", "-",
        ]
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, timeout=30, check=False),
        )
        stderr = result.stderr.decode(errors="replace")
        # Parse: Yuv.Y.AVERAGE from signalstats output
        import re
        m = re.search(r"Yuv\.Y\.AVERAGE=(\d+\.?\d*)", stderr)
        if m:
            yuv_y = float(m.group(1))
            # YUV Y range is 16–235 for limited range; normalise to [0, 1]
            return max(0.0, min(1.0, (yuv_y - 16) / 219))
    except Exception as exc:
        logger.warning("luma_measure_failed", extra={"path": segment_path, "error": str(exc)})
    return None


# ---------------------------------------------------------------------------
# Global colour pass
# ---------------------------------------------------------------------------

async def apply_global_colour_pass(
    segments: list[RenderedSegment],
    output_dir: str,
) -> tuple[list[RenderedSegment], list[str]]:
    """
    Measure per-segment luma and apply an eq= correction to bring each
    segment close to the configured COLOUR_PASS_TARGET_LUMA.

    Returns (corrected_segments, warnings).
    Segments that do not need correction are returned as-is.
    """
    s = get_settings()
    if not s.COLOUR_PASS_ENABLED:
        return segments, []

    import os, tempfile
    target = s.COLOUR_PASS_TARGET_LUMA
    max_corr = s.COLOUR_PASS_MAX_CORRECTION
    target_sat = s.COLOUR_PASS_TARGET_SATURATION
    warnings: list[str] = []
    corrected: list[RenderedSegment] = []

    for seg in segments:
        luma = await measure_luma(seg.local_path)
        if luma is None:
            corrected.append(seg)
            continue

        delta = target - luma
        if abs(delta) <= s.RENDER_LUMA_TOLERANCE:
            # Within tolerance — no correction needed
            corrected.append(seg)
            continue

        # Clamp correction to max_corr
        brightness_adj = max(-max_corr, min(max_corr, delta * 1.5))
        eq_filter = (
            f"eq=brightness={brightness_adj:.3f}"
            f":saturation={target_sat:.2f}"
        )

        base = os.path.splitext(os.path.basename(seg.local_path))[0]
        out_path = os.path.join(output_dir, f"{base}_colour.mp4")

        try:
            await run_ffmpeg([
                "-i", seg.local_path,
                "-vf", eq_filter,
                "-c:v", s.MEDIA_VIDEO_CODEC,
                "-preset", "ultrafast",
                "-crf", str(s.MEDIA_CRF),
                "-pix_fmt", s.RENDER_PIXEL_FORMAT,
                "-an",
                "-movflags", "+faststart",
                out_path,
            ])
            import dataclasses
            new_seg = dataclasses.replace(
                seg, local_path=out_path, brightness_adjusted=True
            )
            corrected.append(new_seg)
            logger.debug(
                "colour_pass_applied",
                extra={"scene": seg.scene_id, "luma_before": round(luma, 3),
                       "brightness_adj": round(brightness_adj, 3)},
            )
        except Exception as exc:
            warnings.append(f"Scene {seg.scene_id}: colour pass failed ({exc})")
            corrected.append(seg)

    logger.info(
        "colour_pass_complete",
        extra={"adjusted": sum(1 for s in corrected if s.brightness_adjusted),
               "total": len(corrected)},
    )
    return corrected, warnings
