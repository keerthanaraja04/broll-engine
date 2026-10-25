"""
Text processor for the input stage.

Two sub-paths
-------------
script path
    Input:  a plain-text narration script string.
    Output: TTS audio + word timings → Transcript(source="script").

scenes path
    Input:  list of scene dicts with keys:
              text          — narration / subtitle text  (required)
              visual_hint   — optional b-roll search hint  (optional)
              duration_hint — optional target duration      (optional)
    Output: TTS for the joined text → Transcript(source="scenes").
            Each Sentence in the transcript carries the original
            visual_hint and scene_index so downstream stages can
            make per-scene footage decisions.

Word-timing alignment for scenes
---------------------------------
After TTS, word timings are assigned back to scenes proportionally:
words are distributed to scene i if their midpoint timestamp falls
within scene i's time slice (proportional to character count).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.adapters.base import TTSResult
from app.adapters.tts_adapter import get_tts_adapter
from app.core.config import get_settings
from app.core.errors import EmptyScriptError
from app.core.logging import get_logger
from app.pipeline.input_stage.transcript import (
    Sentence,
    Transcript,
    WordTiming,
    sentences_from_words,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SceneInput:
    """Normalised representation of one entry from the scenes[] payload."""
    text: str
    visual_hint: str | None = None
    duration_hint: float | None = None
    index: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_script(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        raise EmptyScriptError("Script is empty or contains only whitespace")
    return stripped


def _split_into_tts_calls(text: str, max_chars: int) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters, breaking
    only at sentence boundaries (`.!?`) or newlines.
    Never splits a word in half.
    """
    if len(text) <= max_chars:
        return [text]

    # Split on sentence-ending punctuation, keeping the delimiter
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current = ""

    for part in parts:
        if not part:
            continue
        # If a single part itself exceeds max_chars, split by words
        if len(part) > max_chars:
            words = part.split()
            for word in words:
                if len(current) + len(word) + 1 > max_chars and current:
                    chunks.append(current.strip())
                    current = word + " "
                else:
                    current += word + " "
        elif len(current) + len(part) + 1 > max_chars and current:
            chunks.append(current.strip())
            current = part + " "
        else:
            current += part + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _tts_words_to_pipeline(tts_result: TTSResult) -> list[WordTiming]:
    return [
        WordTiming(word=w.word, start=w.start, end=w.end)
        for w in tts_result.word_timings
    ]


def _assign_words_to_scenes(
    words: list[WordTiming],
    scenes: list[SceneInput],
    total_duration: float,
) -> list[list[WordTiming]]:
    """
    Assign each word to its scene based on proportional time slices.

    Scene i spans [scene_start_i, scene_end_i] where the boundaries are
    proportional to the cumulative character count.
    """
    if not scenes:
        return []

    total_chars = sum(len(sc.text) for sc in scenes)
    if total_chars == 0:
        return [[] for _ in scenes]

    # Build scene time boundaries
    boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    for sc in scenes:
        fraction = len(sc.text) / total_chars
        end = cursor + fraction * total_duration
        boundaries.append((cursor, end))
        cursor = end

    # Assign words
    scene_words: list[list[WordTiming]] = [[] for _ in scenes]
    for w in words:
        midpoint = (w.start + w.end) / 2.0
        assigned = False
        for i, (s_start, s_end) in enumerate(boundaries):
            if s_start <= midpoint <= s_end:
                scene_words[i].append(w)
                assigned = True
                break
        if not assigned and scene_words:
            # Assign to last scene as a fallback
            scene_words[-1].append(w)

    return scene_words


async def _run_tts(text: str, voice: str | None) -> TTSResult:
    """Call TTS adapter, handling multi-chunk texts."""
    s = get_settings()
    tts = get_tts_adapter()

    chunks = _split_into_tts_calls(text, s.TTS_MAX_CHARS_PER_CALL)
    if len(chunks) == 1:
        return await tts.synthesize(text, voice=voice)

    # Multiple chunks — synthesise and concatenate audio + word timings
    results: list[TTSResult] = []
    for chunk in chunks:
        r = await tts.synthesize(chunk, voice=voice)
        results.append(r)

    # Merge: concatenate audio bytes and offset word timings
    merged_audio = b"".join(r.audio_bytes for r in results)
    merged_timings = []
    offset = 0.0
    for r in results:
        for w in r.word_timings:
            from app.adapters.base import WordTimingTTS
            merged_timings.append(
                WordTimingTTS(word=w.word, start=w.start + offset, end=w.end + offset)
            )
        offset += r.duration

    total_duration = sum(r.duration for r in results)
    return TTSResult(
        audio_bytes=merged_audio,
        audio_format=results[0].audio_format,
        sample_rate=results[0].sample_rate,
        duration=total_duration,
        word_timings=merged_timings,
        voice=results[0].voice,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def process_script(
    script: str,
    voice: str | None = None,
    language: str | None = None,
) -> Transcript:
    """
    Synthesise *script* with TTS and return a Transcript.

    Args:
        script:   Plain-text narration script.
        voice:    TTS voice identifier (None → TTS_DEFAULT_VOICE from config).
        language: ISO 639-1 hint stored in the transcript.

    Returns:
        Transcript(source="script") with words, sentences, and TTS audio.
    """
    clean = _validate_script(script)

    tts_result = await _run_tts(clean, voice=voice)
    words = _tts_words_to_pipeline(tts_result)
    sentences = sentences_from_words(words)

    transcript = Transcript(
        language=language or "en",
        duration=tts_result.duration,
        words=words,
        sentences=sentences,
        source="script",
        tts_audio_bytes=tts_result.audio_bytes,
    )

    logger.info(
        "script_processed",
        extra={
            "words": transcript.word_count,
            "sentences": len(sentences),
            "duration": tts_result.duration,
        },
    )
    return transcript


async def process_scenes(
    scenes: list[dict[str, Any]],
    voice: str | None = None,
    language: str | None = None,
) -> Transcript:
    """
    Synthesise the joined text of *scenes* with TTS and return a Transcript.

    Each Sentence in the result carries visual_hint and scene_index so that
    downstream stages can select per-scene b-roll footage.

    Args:
        scenes:   List of scene dicts with keys:
                    text          (str, required)
                    visual_hint   (str, optional)
                    duration_hint (float, optional, seconds)
        voice:    TTS voice identifier.
        language: ISO 639-1 hint.

    Returns:
        Transcript(source="scenes").
    """
    s = get_settings()

    # Normalise inputs
    scene_inputs: list[SceneInput] = []
    for i, sc in enumerate(scenes):
        text = sc.get("text", "").strip() if isinstance(sc, dict) else str(sc).strip()
        if not text:
            continue
        scene_inputs.append(
            SceneInput(
                text=text,
                visual_hint=sc.get("visual_hint") if isinstance(sc, dict) else None,
                duration_hint=sc.get("duration_hint") if isinstance(sc, dict) else None,
                index=i,
            )
        )

    if not scene_inputs:
        raise EmptyScriptError(
            "All scenes have empty text — nothing to synthesise"
        )

    # Join scenes for TTS
    joined = s.SCENE_JOIN_SEPARATOR.join(sc.text for sc in scene_inputs)

    tts_result = await _run_tts(joined, voice=voice)
    all_words = _tts_words_to_pipeline(tts_result)

    # Assign words to scenes proportionally
    scene_word_lists = _assign_words_to_scenes(
        all_words, scene_inputs, tts_result.duration
    )

    # Build per-scene sentences
    sentences: list[Sentence] = []
    sentence_id = 0
    for sc, sw in zip(scene_inputs, scene_word_lists):
        if not sw:
            # Scene had no words assigned; synthesise empty sentence with estimated timing
            total_chars = sum(len(x.text) for x in scene_inputs)
            fraction = len(sc.text) / max(total_chars, 1)
            est_start = sc.index * tts_result.duration / max(len(scene_inputs), 1)
            est_end = est_start + fraction * tts_result.duration
            sentences.append(
                Sentence(
                    id=sentence_id,
                    text=sc.text,
                    start=est_start,
                    end=est_end,
                    visual_hint=sc.visual_hint,
                    scene_index=sc.index,
                )
            )
        else:
            # Split scene words further at punctuation
            sub_sentences = sentences_from_words(sw)
            for sub in sub_sentences:
                sub.id = sentence_id
                sub.visual_hint = sc.visual_hint
                sub.scene_index = sc.index
                sentences.append(sub)
                sentence_id += 1
            if not sub_sentences:
                # Fallback: single sentence from all words
                sentences.append(
                    Sentence(
                        id=sentence_id,
                        text=sc.text,
                        start=sw[0].start,
                        end=sw[-1].end,
                        words=sw,
                        visual_hint=sc.visual_hint,
                        scene_index=sc.index,
                    )
                )
        sentence_id += 1

    transcript = Transcript(
        language=language or "en",
        duration=tts_result.duration,
        words=all_words,
        sentences=sentences,
        source="scenes",
        tts_audio_bytes=tts_result.audio_bytes,
    )

    logger.info(
        "scenes_processed",
        extra={
            "n_scenes": len(scene_inputs),
            "words": transcript.word_count,
            "sentences": len(sentences),
            "duration": tts_result.duration,
        },
    )
    return transcript
