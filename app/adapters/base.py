"""
Abstract adapter interfaces for all AI services.

Every concrete adapter (real or mock) must implement exactly one of these
abstract base classes.  Callers depend only on these interfaces — never on
a specific vendor implementation.

Shared dataclasses
------------------
WordTiming        — a single word with start/end timestamps (seconds)
TranscriptSegment — a phrase-level segment from the STT result
TranscriptResult  — full STTAdapter.transcribe() return value
VLMScore          — a single image score from VLMAdapter.score()
TTSResult         — TTSAdapter.synthesize() return value

Factories
---------
Each adapter module exposes a get_<type>_adapter() function that reads
<TYPE>_ADAPTER from Settings and returns the appropriate implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Shared dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WordTiming:
    """A single recognised word with precise timestamps."""
    word: str
    start: float   # seconds from audio start
    end: float     # seconds from audio start


@dataclass
class TranscriptSegment:
    """A contiguous phrase-level segment (usually a sentence or clause)."""
    id: int
    text: str
    start: float
    end: float
    words: list[WordTiming] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptResult:
    """
    Full output of STTAdapter.transcribe().

    Attributes
    ----------
    words:    Word-level tokens with timestamps.  May be empty if the backend
              does not support word timestamps.
    segments: Phrase-level segments.
    language: ISO 639-1 code detected or supplied (e.g. "en", "hi", "ta").
    duration: Total audio duration in seconds.
    text:     Full concatenated transcript string.
    """
    words: list[WordTiming]
    segments: list[TranscriptSegment]
    language: str
    duration: float
    text: str


@dataclass
class BoundingBox:
    """Normalised bounding box [0, 1] — (x1, y1, x2, y2) top-left origin."""
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def full_frame(cls) -> "BoundingBox":
        return cls(x1=0.0, y1=0.0, x2=1.0, y2=1.0)


@dataclass
class VLMScore:
    """
    Score for one image from VLMAdapter.score().

    Attributes
    ----------
    score:        Relevance score 1–5 (5 = highly relevant to the text).
    reason:       One-sentence human-readable explanation from the model.
    subject_bbox: Normalised bounding box of the primary subject in the image.
    flags:        List of content advisory flags e.g. ["nsfw", "text_overlay"].
    image_index:  Index of the image in the original input list.
    """
    score: int                          # 1–5
    reason: str
    subject_bbox: BoundingBox
    flags: list[str]
    image_index: int = 0


@dataclass
class WordTimingTTS:
    """Word timing produced by the TTS adapter."""
    word: str
    start: float
    end: float


@dataclass
class TTSResult:
    """
    Output of TTSAdapter.synthesize().

    Attributes
    ----------
    audio_bytes:   Raw audio bytes in the format specified by TTS_AUDIO_FORMAT.
    audio_format:  "wav" | "mp3" | "flac" | "opus".
    sample_rate:   Sample rate in Hz.
    duration:      Duration of the synthesised audio in seconds.
    word_timings:  Per-word timings.  Always populated (via Whisper fallback
                   if the TTS backend does not return them natively).
    voice:         The voice identifier that was used.
    """
    audio_bytes: bytes
    audio_format: str
    sample_rate: int
    duration: float
    word_timings: list[WordTimingTTS]
    voice: str


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class STTAdapter(ABC):
    """Speech-to-text adapter interface."""

    @abstractmethod
    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptResult:
        """
        Transcribe an audio file.

        Args:
            audio_path: Absolute path to the audio file on disk.
            language:   Optional ISO 639-1 hint (e.g. "en", "hi").

        Returns:
            TranscriptResult with words, segments, detected language, and full text.
        """
        ...


class LLMAdapter(ABC):
    """Large-language-model adapter interface."""

    @abstractmethod
    async def json_complete(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run a JSON-mode completion and validate/coerce the response against *schema*.

        Args:
            system: System prompt (persona / instructions).
            prompt: User-turn prompt.
            schema: JSON Schema dict the response MUST conform to.

        Returns:
            Validated dict that matches *schema*.

        Raises:
            LLMSchemaError: If validation fails even after repair retries.
            GroqServiceError / ExternalServiceError: On API failure.
        """
        ...


class VLMAdapter(ABC):
    """Vision-language-model adapter interface."""

    @abstractmethod
    async def score(
        self,
        images: list[bytes],
        text: str,
    ) -> list[VLMScore]:
        """
        Score each image for relevance to *text*.

        Args:
            images: List of raw image bytes (JPEG / PNG / WebP).
            text:   The query / narration text to score against.

        Returns:
            One VLMScore per input image, in the same order.
        """
        ...


class TTSAdapter(ABC):
    """Text-to-speech adapter interface."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> TTSResult:
        """
        Synthesise speech from *text*.

        Args:
            text:  The text to speak.
            voice: Voice identifier. Falls back to TTS_DEFAULT_VOICE from config.

        Returns:
            TTSResult with audio bytes, format, duration, and word timings.
        """
        ...
