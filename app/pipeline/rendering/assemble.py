"""
Final assembly: concatenate segments with crossfades, mix narration
audio (AAC), optionally burn-in captions, and encode the final H.264 MP4.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.rendering.ff_helpers import run_ffmpeg
from app.pipeline.rendering.models import RenderedSegment

logger = get_logger(__name__)


def _build_concat_list(segment_paths: list[str], output_dir: str) -> str:
    """Write an FFmpeg concat demuxer list file and return its path."""
    list_path = os.path.join(output_dir, "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in segment_paths:
            # Escape single quotes in paths
            safe = path.replace("'", r"\'")
            f.write(f"file '{safe}'\n")
    return list_path


async def concatenate_segments(
    segments: list[RenderedSegment],
    output_dir: str | None = None,
) -> str:
    """
    Concatenate normalised segments using the concat demuxer (no re-encode).

    All segments must have the same codec, fps, resolution, and pixel format
    (guaranteed by the normalisation step).

    Returns the path to the concatenated video (no audio).
    """
    s = get_settings()
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    list_path = _build_concat_list([seg.local_path for seg in segments], tmp_dir)
    out_path = os.path.join(tmp_dir, "concat_video.mp4")

    # Use copy codec (fast) since all segments are already normalised
    args = [
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        "-movflags", "+faststart",
        out_path,
    ]

    await run_ffmpeg(args)
    logger.info(
        "segments_concatenated",
        extra={"segments": len(segments), "out": out_path},
    )
    return out_path


async def mix_narration_audio(
    video_path: str,
    narration_path: str | None,
    total_duration: float,
    output_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Mix narration audio into the concatenated video.

    If *narration_path* is None (e.g. text-path with TTS), the narration
    audio bytes should have been written to a temp file by the caller.

    The output is an H.264 video + AAC audio MP4.
    """
    s = get_settings()
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    if output_path is None:
        output_path = os.path.join(tmp_dir, "with_audio.mp4")

    if narration_path is None:
        # No narration — add silent audio track so the container always has audio
        args = [
            "-i", video_path,
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy",
            "-c:a", s.MEDIA_AUDIO_CODEC,
            "-b:a", "128k",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]
    else:
        args = [
            "-i", video_path,
            "-i", narration_path,
            "-c:v", "copy",
            "-c:a", s.MEDIA_AUDIO_CODEC,
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-movflags", "+faststart",
            output_path,
        ]

    await run_ffmpeg(args)
    logger.info("audio_mixed", extra={"video": video_path, "out": output_path})
    return output_path


async def burn_in_captions(
    video_path: str,
    subtitle_filter: str,
    output_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    """
    Burn-in subtitles using the FFmpeg subtitles filter.

    This requires a full re-encode of the video stream.
    """
    s = get_settings()
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
    if output_path is None:
        output_path = os.path.join(tmp_dir, "burned_captions.mp4")

    args = [
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:v", s.MEDIA_VIDEO_CODEC,
        "-preset", s.MEDIA_PRESET,
        "-crf", str(s.MEDIA_CRF),
        "-c:a", "copy",
        "-movflags", "+faststart",
        output_path,
    ]

    await run_ffmpeg(args)
    logger.info("captions_burned", extra={"video": video_path, "out": output_path})
    return output_path


async def apply_crossfades(
    segments: list[RenderedSegment],
    output_dir: str | None = None,
) -> str:
    """
    Apply short crossfades between consecutive segments using xfade filter.

    For simplicity, if there are only 1-2 segments, skip crossfades and
    return the first segment (or a simple concat).

    Returns the path to the crossfaded video.
    """
    s = get_settings()
    tmp_dir = output_dir or s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()

    if len(segments) <= 1:
        return segments[0].local_path if segments else ""

    xfade_dur = s.RENDER_CROSSFADE_DURATION
    if xfade_dur <= 0:
        # No crossfade — just concat
        return await concatenate_segments(segments, output_dir=tmp_dir)

    # Build xfade chain: each xfade overlaps two inputs
    # offset = accumulated_duration - xfade_duration
    offsets: list[float] = []
    accum = 0.0
    for i, seg in enumerate(segments[:-1]):
        accum += seg.duration
        offsets.append(accum - xfade_dur)

    # Build filter complex
    # [0:v][1:v]xfade=transition=fade:duration=D:offset=O[v01]
    # [v01][2:v]xfade=...[v012]
    filter_parts: list[str] = []
    prev_label = "[0:v]"
    for i in range(len(segments) - 1):
        next_label = f"[v{i + 1}]" if i < len(segments) - 2 else "[vout]"
        filter_parts.append(
            f"{prev_label}[{i + 1}:v]xfade=transition=fade"
            f":duration={xfade_dur:.3f}:offset={offsets[i]:.3f}{next_label}"
        )
        prev_label = next_label

    filter_complex = ";".join(filter_parts)
    out_path = os.path.join(tmp_dir, "crossfaded.mp4")

    input_args: list[str] = []
    for seg in segments:
        input_args.extend(["-i", seg.local_path])

    args = input_args + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", s.MEDIA_VIDEO_CODEC,
        "-preset", s.MEDIA_PRESET,
        "-crf", str(s.MEDIA_CRF),
        "-pix_fmt", s.RENDER_PIXEL_FORMAT,
        "-an",
        "-movflags", "+faststart",
        out_path,
    ]

    await run_ffmpeg(args)
    logger.info("crossfades_applied", extra={"segments": len(segments), "out": out_path})
    return out_path
