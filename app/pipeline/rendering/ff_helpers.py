"""Async FFmpeg / ffprobe subprocess helpers for the rendering stage."""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from app.core.config import get_settings
from app.core.errors import MediaProcessingError, RenderError
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_ffmpeg(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run ffmpeg in a thread pool; returns CompletedProcess."""
    s = get_settings()
    cmd = [s.FFMPEG_BINARY, "-y", *args]
    loop = asyncio.get_running_loop()

    def _run() -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)

    try:
        result = await loop.run_in_executor(None, _run)
    except FileNotFoundError as exc:
        raise RenderError(f"ffmpeg not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError(f"ffmpeg timed out after {timeout}s") from exc

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:600]
        raise RenderError(
            f"ffmpeg failed (rc={result.returncode})",
            details={"stderr": stderr, "cmd": " ".join(cmd[:20])},
        )
    return result


async def run_ffprobe(args: list[str], timeout: int = 30) -> dict[str, Any]:
    """Run ffprobe and return parsed JSON output."""
    s = get_settings()
    cmd = [s.FFPROBE_BINARY, "-v", "quiet", "-print_format", "json", *args]
    loop = asyncio.get_running_loop()

    def _run() -> subprocess.CompletedProcess:  # type: ignore[type-arg]
        return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)

    try:
        result = await loop.run_in_executor(None, _run)
    except FileNotFoundError as exc:
        raise RenderError(f"ffprobe not found: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RenderError("ffprobe timed out") from exc

    if result.returncode != 0:
        raise RenderError(
            "ffprobe failed",
            details={"stderr": result.stderr.decode(errors="replace")[:400]},
        )
    try:
        return json.loads(result.stdout)  # type: ignore[no-any-return]
    except json.JSONDecodeError as exc:
        raise RenderError("ffprobe output is not valid JSON") from exc


async def probe_video(path: str) -> dict[str, Any]:
    """Return ffprobe JSON for a media file."""
    return await run_ffprobe(["-show_format", "-show_streams", path])


async def get_duration(path: str) -> float:
    """Get media duration in seconds via ffprobe."""
    probe = await probe_video(path)
    dur = probe.get("format", {}).get("duration")
    if dur is None:
        for stream in probe.get("streams", []):
            if stream.get("codec_type") in ("audio", "video"):
                dur = stream.get("duration")
                break
    try:
        return float(dur or 0.0)
    except (TypeError, ValueError):
        return 0.0


def target_dimensions(aspect_ratio: str) -> tuple[int, int]:
    """Return (width, height) for the given aspect ratio string."""
    s = get_settings()
    mapping = {
        "16:9": (s.RENDER_WIDTH_16_9, s.RENDER_HEIGHT_16_9),
        "9:16": (s.RENDER_WIDTH_9_16, s.RENDER_HEIGHT_9_16),
        "1:1":  (s.RENDER_WIDTH_1_1, s.RENDER_HEIGHT_1_1),
    }
    if aspect_ratio not in mapping:
        raise RenderError(
            f"Unknown aspect ratio '{aspect_ratio}'",
            details={"allowed": list(mapping.keys())},
        )
    return mapping[aspect_ratio]
