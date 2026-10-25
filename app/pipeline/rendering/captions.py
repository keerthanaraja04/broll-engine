"""
Caption / subtitle generation and burn-in.

SRT generation
--------------
``generate_srt()`` takes a Transcript's word timings and produces a valid
SRT file with at most ``RENDER_CAPTION_MAX_LINES`` lines per cue and at most
``RENDER_CAPTION_MAX_CHARS_PER_LINE`` characters per line.

Burn-in
-------
``build_subtitle_filter()`` returns an FFmpeg ``subtitles`` filter string
that burns the SRT into the video using the appropriate Noto font for the
content language (Latin, Devanagari, or Tamil).
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SRT timestamp formatting
# ---------------------------------------------------------------------------

def _format_srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm (SRT format)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# SRT generation from word timings
# ---------------------------------------------------------------------------

def generate_srt(
    words: list[Any],   # list of WordTiming from transcript
    max_lines: int | None = None,
    max_chars: int | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate an SRT subtitle file from word timestamps.

    Groups words into cues of at most *max_lines* lines and
    *max_chars* characters per line.

    Args:
        words:         List of objects with .word, .start, .end attributes.
        max_lines:     Max lines per cue (defaults to config).
        max_chars:     Max chars per line (defaults to config).
        output_path:   Where to write the .srt file. If None, a temp path
                       is returned but the content is still generated.

    Returns:
        The path to the written SRT file.
    """
    s = get_settings()
    ml = max_lines or s.RENDER_CAPTION_MAX_LINES
    mc = max_chars or s.RENDER_CAPTION_MAX_CHARS_PER_LINE

    if not words:
        logger.warning("srt_no_words")
        return _write_srt([], output_path)

    # Group words into cues
    cues: list[dict[str, Any]] = []
    current_words: list[str] = []
    current_start: float = words[0].start
    current_end: float = words[0].end
    current_line_len = 0

    for w in words:
        word_text = w.word
        # Check if adding this word would exceed line limit
        if current_line_len + len(word_text) + 1 > mc and current_words:
            # Flush current line, start new one
            line = " ".join(current_words)
            cues.append({
                "start": current_start,
                "end": current_end,
                "text": line,
            })
            current_words = []
            current_start = w.start
            current_line_len = 0

        current_words.append(word_text)
        current_line_len += len(word_text) + 1
        current_end = w.end

    # Flush remaining
    if current_words:
        line = " ".join(current_words)
        cues.append({
            "start": current_start,
            "end": current_end,
            "text": line,
        })

    # Now wrap cues to respect max_lines: split long lines into multiple cues
    final_cues: list[dict[str, Any]] = []
    for cue in cues:
        wrapped = textwrap.wrap(cue["text"], width=mc)
        # Group into chunks of max_lines
        for i in range(0, len(wrapped), ml):
            chunk_lines = wrapped[i:i + ml]
            chunk_text = "\n".join(chunk_lines)
            if i == 0:
                final_cues.append({
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": chunk_text,
                })
            else:
                # Continuation cue — same end time
                final_cues.append({
                    "start": cue["start"],
                    "end": cue["end"],
                    "text": chunk_text,
                })

    return _write_srt(final_cues, output_path)


def _write_srt(cues: list[dict[str, Any]], output_path: str | None) -> str:
    """Write cues to an SRT file and return the path."""
    import tempfile
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)

    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(cue['start'])} --> {_format_srt_time(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")  # blank line between cues

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("srt_generated", extra={"cues": len(cues), "path": output_path})
    return output_path


# ---------------------------------------------------------------------------
# Burn-in filter
# ---------------------------------------------------------------------------

def _select_font(language: str) -> str:
    """Select the appropriate Noto font name for *language*."""
    s = get_settings()
    # Devanagari: Hindi, Marathi, Nepali, Sanskrit
    if language in ("hi", "mr", "ne", "sa"):
        return s.RENDER_CAPTION_FONT_DEVA
    # Tamil
    if language == "ta":
        return s.RENDER_CAPTION_FONT_TAMIL
    # Default: Latin
    return s.RENDER_CAPTION_FONT_LATIN


def build_subtitle_filter(
    srt_path: str,
    language: str = "en",
    video_width: int = 1920,
    video_height: int = 1080,
) -> str:
    """
    Build an FFmpeg ``subtitles`` filter that burns the SRT into the video.

    The filter selects the correct Noto font for the content language
    and positions subtitles at the bottom with a semi-transparent box.
    """
    s = get_settings()
    font_name = _select_font(language)
    font_size = s.RENDER_CAPTION_FONT_SIZE
    margin_v = s.RENDER_CAPTION_MARGIN_V
    opacity = s.RENDER_CAPTION_BOX_OPACITY

    # Escape colons in the SRT path for the filter syntax
    escaped_path = srt_path.replace(":", r"\:").replace("'", r"\'")

    # Build subtitles filter with styling
    # Force_style overrides ASS style properties
    style_parts = [
        f"FontName={font_name}",
        f"FontSize={font_size}",
        f"MarginV={margin_v}",
        f"BorderStyle=4",       # box border
        f"Outline=1",
        f"Shadow=0",
    ]

    style_str = ",".join(style_parts)
    filter_str = f"subtitles='{escaped_path}':force_style='{style_str}'"

    logger.info(
        "subtitle_filter_built",
        extra={"font": font_name, "language": language, "srt": srt_path},
    )
    return filter_str
