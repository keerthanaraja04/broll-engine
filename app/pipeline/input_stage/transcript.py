"""
Canonical Transcript model used by every downstream pipeline stage.

Produced by both the audio path (STT → Transcript) and the text path
(script/scenes → TTS → Transcript).  Downstream stages never need to
know which path produced it.

Data model
----------
Transcript
  language        — ISO 639-1 code, e.g. "en", "hi", "ta"
  duration        — total audio duration in seconds
  words           — flat list of every word with start/end timestamps
  sentences       — list of Sentence objects (one per segment / scene)
  source          — "audio" | "script" | "scenes"
  audio_path      — path to the 16 kHz mono WAV used for STT (if audio path)
  tts_audio_bytes — raw WAV bytes produced by TTS (if text path)
  warnings        — non-fatal processing notes accumulated during ingestion

Sentence
  id              — 0-based index
  text            — plain-text content
  start / end     — timestamps in seconds
  words           — words that fall within this sentence
  visual_hint     — optional free-text hint for b-roll search (scenes input only)
  scene_index     — source scene index (scenes input only)

WordTiming  (used only here; distinct from the adapter-layer WordTiming in
            app/adapters/base.py to keep the pipeline model self-contained)
  word, start, end
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class WordTiming:
    """A single word with its audio timestamp."""
    word: str
    start: float   # seconds from start of the audio
    end: float     # seconds from start of the audio

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"WordTiming: end ({self.end}) < start ({self.start})"
            )


@dataclass
class Sentence:
    """
    A contiguous group of words forming a sentence / segment / scene.

    When the source is "audio", sentences map 1:1 to Whisper segments.
    When the source is "script" or "scenes", sentences are split at
    terminal punctuation from the TTS word-timing output.
    """
    id: int
    text: str
    start: float
    end: float
    words: list[WordTiming] = field(default_factory=list)
    # Set only for scenes[] input
    visual_hint: str | None = None
    scene_index: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Transcript:
    """
    Unified transcript produced by the input stage.

    Consumed by all downstream stages: keyword extraction, VLM scoring,
    subtitle burning, etc.
    """
    language: str
    duration: float
    words: list[WordTiming]
    sentences: list[Sentence]
    source: Literal["audio", "script", "scenes"]

    # Provenance — set whichever is relevant for the path taken
    audio_path: str | None = None          # path to 16 kHz mono WAV
    tts_audio_bytes: bytes | None = None   # raw WAV from TTS (text path)

    # Accumulator for non-fatal notes
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Full plain-text concatenation of all sentences."""
        return " ".join(s.text for s in self.sentences).strip()

    @property
    def word_count(self) -> int:
        return len(self.words)

    def words_in_range(self, start: float, end: float) -> list[WordTiming]:
        """Return words whose midpoint falls within [start, end]."""
        return [
            w for w in self.words
            if start <= (w.start + w.end) / 2 <= end
        ]

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    # ------------------------------------------------------------------
    # Serialisation (for PipelineContext storage)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "source": self.source,
            "audio_path": self.audio_path,
            "warnings": self.warnings,
            "word_count": self.word_count,
            "sentence_count": len(self.sentences),
            "text_preview": self.text[:200],
            "words": [
                {"word": w.word, "start": w.start, "end": w.end}
                for w in self.words
            ],
            "sentences": [
                {
                    "id": s.id,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "visual_hint": s.visual_hint,
                    "scene_index": s.scene_index,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in s.words
                    ],
                }
                for s in self.sentences
            ],
        }


# ---------------------------------------------------------------------------
# Builder helpers shared by audio_processor and text_processor
# ---------------------------------------------------------------------------

def sentences_from_words(
    words: list[WordTiming],
    min_gap_seconds: float = 0.5,
    max_sentence_words: int = 40,
) -> list[Sentence]:
    """
    Segment *words* into sentences using two heuristics:

    1. Terminal punctuation (.!?) on the preceding word triggers a split.
    2. A silence gap >= *min_gap_seconds* between consecutive words splits.
    3. Hard cap at *max_sentence_words* words per sentence.

    Returns a list of Sentence objects with words populated.
    """
    if not words:
        return []

    sentences: list[Sentence] = []
    current: list[WordTiming] = []
    sentence_id = 0

    def _flush() -> None:
        nonlocal sentence_id
        if not current:
            return
        text = " ".join(w.word for w in current).strip()
        sentences.append(
            Sentence(
                id=sentence_id,
                text=text,
                start=current[0].start,
                end=current[-1].end,
                words=list(current),
            )
        )
        sentence_id += 1
        current.clear()

    for i, w in enumerate(words):
        current.append(w)

        # Check split conditions
        should_split = False

        # Terminal punctuation
        clean = w.word.rstrip()
        if clean and clean[-1] in ".!?":
            should_split = True

        # Silence gap to next word
        if not should_split and i + 1 < len(words):
            gap = words[i + 1].start - w.end
            if gap >= min_gap_seconds:
                should_split = True

        # Hard cap
        if len(current) >= max_sentence_words:
            should_split = True

        if should_split:
            _flush()

    _flush()  # final sentence
    return sentences
