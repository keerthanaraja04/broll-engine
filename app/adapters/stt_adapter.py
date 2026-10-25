"""
STT (Speech-to-Text) adapter.

Real  : GroqSTTAdapter  — Groq Whisper API with word timestamps
Mock  : MockSTTAdapter  — deterministic, no API key needed

Factory: get_stt_adapter() reads STT_ADAPTER from settings ("groq" | "mock").
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx

from app.adapters._cache import AdapterCache, make_cache_key_bytes
from app.adapters._ratelimiter import get_bucket
from app.adapters._retry import parse_retry_after, with_retry
from app.adapters.base import (
    STTAdapter,
    TranscriptResult,
    TranscriptSegment,
    WordTiming,
)
from app.core.config import get_settings
from app.core.errors import GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _to_cache(result: TranscriptResult) -> dict[str, Any]:
    return {
        "text": result.text,
        "language": result.language,
        "duration": result.duration,
        "words": [{"word": w.word, "start": w.start, "end": w.end}
                   for w in result.words],
        "segments": [
            {"id": s.id, "text": s.text, "start": s.start, "end": s.end,
             "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
             "words": [{"word": w.word, "start": w.start, "end": w.end}
                        for w in s.words]}
            for s in result.segments
        ],
    }


def _from_cache(d: dict[str, Any]) -> TranscriptResult:
    words = [WordTiming(word=w["word"], start=w["start"], end=w["end"])
             for w in d.get("words", [])]
    segments = [
        TranscriptSegment(
            id=seg["id"], text=seg["text"], start=seg["start"], end=seg["end"],
            avg_logprob=seg.get("avg_logprob", 0.0),
            no_speech_prob=seg.get("no_speech_prob", 0.0),
            words=[WordTiming(word=w["word"], start=w["start"], end=w["end"])
                   for w in seg.get("words", [])],
        )
        for seg in d.get("segments", [])
    ]
    return TranscriptResult(
        text=d["text"], language=d["language"],
        duration=d["duration"], words=words, segments=segments,
    )


def _parse_groq_response(body: dict[str, Any]) -> TranscriptResult:
    """Convert a Groq Whisper verbose_json response to TranscriptResult."""
    text = body.get("text", "")
    language = body.get("language", "en")
    duration = float(body.get("duration", 0.0))

    raw_words = body.get("words", [])
    words = [
        WordTiming(
            word=w.get("word", "").strip(),
            start=float(w.get("start", 0.0)),
            end=float(w.get("end", 0.0)),
        )
        for w in raw_words if w.get("word", "").strip()
    ]

    raw_segs = body.get("segments", [])
    segments: list[TranscriptSegment] = []
    for seg in raw_segs:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        seg_words = [w for w in words
                     if w.start >= seg_start and w.end <= seg_end + 0.05]
        segments.append(
            TranscriptSegment(
                id=int(seg.get("id", len(segments))),
                text=seg.get("text", "").strip(),
                start=seg_start, end=seg_end,
                avg_logprob=float(seg.get("avg_logprob", 0.0)),
                no_speech_prob=float(seg.get("no_speech_prob", 0.0)),
                words=seg_words,
            )
        )

    return TranscriptResult(
        text=text, language=language, duration=duration,
        words=words, segments=segments,
    )


class GroqSTTAdapter(STTAdapter):
    """Speech-to-text via Groq Whisper API with caching + rate limiting."""

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._base_url = str(s.GROQ_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(s.STT_TIMEOUT_SECONDS)
        self._cache = AdapterCache("stt", ttl=s.STT_CACHE_TTL_SECONDS,
                                   enabled=s.STT_CACHE_ENABLED)
        self._bucket = get_bucket("stt", capacity=s.STT_RATE_LIMIT_TOKENS,
                                  refill_rate=s.STT_RATE_LIMIT_REFILL_RATE)

    async def transcribe(
        self, audio_path: str, language: str | None = None,
    ) -> TranscriptResult:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        cache_key = make_cache_key_bytes(audio_bytes, language or "")
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("stt_cache_hit", extra={"path": audio_path})
            return _from_cache(cached)

        await self._bucket.acquire()

        async def _call() -> TranscriptResult:
            return await self._call_groq(audio_bytes, audio_path, language)

        result = await with_retry(
            fn=_call, max_retries=self._s.STT_MAX_RETRIES,
            adapter_name="stt",
            retryable_exceptions=(httpx.TimeoutException, httpx.ConnectError,
                                  GroqServiceError),
        )
        await self._cache.set(cache_key, _to_cache(result))
        return result

    async def _call_groq(
        self, audio_bytes: bytes, filename: str, language: str | None,
    ) -> TranscriptResult:
        s = self._s
        url = f"{self._base_url}/audio/transcriptions"
        basename = os.path.basename(filename)
        files = {"file": (basename, audio_bytes, "audio/mpeg")}
        data: dict[str, str] = {
            "model": s.STT_MODEL,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        if language:
            data["language"] = language

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {s.GROQ_API_KEY}"},
                    data=data, files=files,
                )
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"STT timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"STT connect error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Groq STT rate limit",
                                 details={"retry_after_seconds":
                                          parse_retry_after(response)})
        if not response.is_success:
            raise GroqServiceError(
                f"Groq STT HTTP {response.status_code}",
                details={"body": response.text[:400]})
        try:
            return _parse_groq_response(response.json())
        except Exception as exc:
            raise GroqServiceError("STT parse error") from exc


class MockSTTAdapter(STTAdapter):
    """
    Deterministic mock STT adapter for testing.

    Returns a fixed transcript derived from the audio file path hash
    so different inputs produce different but stable outputs.
    """

    async def transcribe(
        self, audio_path: str, language: str | None = None,
    ) -> TranscriptResult:
        basename = os.path.basename(audio_path)
        h = int(hashlib.sha256(basename.encode()).hexdigest(), 16)
        word_pool = [
            "the", "city", "skyline", "glows", "at", "dusk",
            "people", "walk", "along", "the", "waterfront",
            "technology", "shapes", "our", "future", "every", "day",
        ]
        n_words = 8 + (h % 6)
        words_text = [word_pool[(h + i * 7) % len(word_pool)]
                      for i in range(n_words)]
        text = " ".join(words_text)

        words = [
            WordTiming(word=w, start=round(i * 0.4, 2),
                       end=round(i * 0.4 + 0.35, 2))
            for i, w in enumerate(words_text)
        ]
        seg = TranscriptSegment(
            id=0, text=text, start=0.0,
            end=round(len(words_text) * 0.4, 2),
            words=words, avg_logprob=-0.15, no_speech_prob=0.02,
        )
        return TranscriptResult(
            text=text, language=language or "en",
            duration=seg.end + 0.2,
            words=words, segments=[seg],
        )


_adapter: STTAdapter | None = None


def get_stt_adapter() -> STTAdapter:
    global _adapter
    if _adapter is None:
        impl = get_settings().STT_ADAPTER.lower()
        _adapter = MockSTTAdapter() if impl == "mock" else GroqSTTAdapter()
    return _adapter


def reset_stt_adapter() -> None:
    global _adapter
    _adapter = None
