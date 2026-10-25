"""
TTS (Text-to-Speech) adapter.

Real  : GroqTTSAdapter  — Groq TTS API; falls back to Whisper for word timings
Mock  : MockTTSAdapter  — returns a minimal valid WAV + deterministic timings

Factory: get_tts_adapter() reads TTS_ADAPTER from settings ("groq" | "mock").
"""

from __future__ import annotations

import os
import struct
import tempfile
from typing import Any

import httpx

from app.adapters._cache import AdapterCache, make_cache_key_bytes
from app.adapters._ratelimiter import get_bucket
from app.adapters._retry import parse_retry_after, with_retry
from app.adapters.base import TTSAdapter, TTSResult, WordTimingTTS
from app.core.config import get_settings
from app.core.errors import GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _to_cache(result: TTSResult) -> dict[str, Any]:
    return {
        "audio_bytes": result.audio_bytes.hex(),
        "audio_format": result.audio_format,
        "sample_rate": result.sample_rate,
        "duration": result.duration,
        "voice": result.voice,
        "word_timings": [
            {"word": w.word, "start": w.start, "end": w.end}
            for w in result.word_timings
        ],
    }


def _from_cache(d: dict[str, Any]) -> TTSResult:
    return TTSResult(
        audio_bytes=bytes.fromhex(d["audio_bytes"]),
        audio_format=d["audio_format"],
        sample_rate=d["sample_rate"],
        duration=d["duration"],
        voice=d["voice"],
        word_timings=[
            WordTimingTTS(word=w["word"], start=w["start"], end=w["end"])
            for w in d.get("word_timings", [])
        ],
    )


def _timings_from_text(text: str, duration: float) -> list[WordTimingTTS]:
    """Evenly-spaced synthetic word timings as a last resort."""
    words = text.split()
    if not words:
        return []
    gap = duration / len(words)
    return [
        WordTimingTTS(word=w, start=round(i * gap, 3),
                      end=round((i + 1) * gap - 0.05, 3))
        for i, w in enumerate(words)
    ]


class GroqTTSAdapter(TTSAdapter):
    """Synthesise audio via Groq TTS; obtain word timings via Whisper."""

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._base_url = str(s.GROQ_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(s.TTS_TIMEOUT_SECONDS)
        self._cache = AdapterCache("tts", ttl=s.TTS_CACHE_TTL_SECONDS,
                                   enabled=s.TTS_CACHE_ENABLED)
        self._bucket = get_bucket("tts", capacity=s.TTS_RATE_LIMIT_TOKENS,
                                  refill_rate=s.TTS_RATE_LIMIT_REFILL_RATE)

    async def synthesize(self, text: str,
                         voice: str | None = None) -> TTSResult:
        s = self._s
        used_voice = voice or s.TTS_DEFAULT_VOICE
        cache_key = make_cache_key_bytes(text, used_voice, s.TTS_MODEL,
                                         s.TTS_AUDIO_FORMAT)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("tts_cache_hit")
            return _from_cache(cached)

        await self._bucket.acquire()

        async def _call() -> bytes:
            return await self._call_groq_tts(text, used_voice)

        audio_bytes = await with_retry(
            fn=_call, max_retries=s.TTS_MAX_RETRIES,
            adapter_name="tts",
            retryable_exceptions=(httpx.TimeoutException, httpx.ConnectError,
                                  GroqServiceError),
        )

        if s.TTS_AUDIO_FORMAT == "wav":
            duration = len(audio_bytes) / (s.TTS_SAMPLE_RATE * 2 * 2)
        else:
            duration = len(audio_bytes) / 16000

        word_timings = await self._get_word_timings(audio_bytes, text)

        result = TTSResult(
            audio_bytes=audio_bytes, audio_format=s.TTS_AUDIO_FORMAT,
            sample_rate=s.TTS_SAMPLE_RATE, duration=duration,
            word_timings=word_timings, voice=used_voice,
        )
        await self._cache.set(cache_key, _to_cache(result))
        return result

    async def _call_groq_tts(self, text: str, voice: str) -> bytes:
        s = self._s
        url = f"{self._base_url}/audio/speech"
        payload = {
            "model": s.TTS_MODEL, "input": text,
            "voice": voice, "response_format": s.TTS_AUDIO_FORMAT,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {s.GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"TTS timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"TTS connect error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Groq TTS rate limit",
                                 details={"retry_after_seconds":
                                          parse_retry_after(response)})
        if not response.is_success:
            raise GroqServiceError(
                f"Groq TTS HTTP {response.status_code}",
                details={"body": response.text[:300]})
        return response.content

    async def _get_word_timings(
        self, audio_bytes: bytes, text: str,
    ) -> list[WordTimingTTS]:
        """Run audio through Whisper to get word timestamps."""
        s = self._s
        if not s.TTS_FALLBACK_WORD_TIMINGS:
            dur = len(audio_bytes) / max(1, s.TTS_SAMPLE_RATE * 2 * 2)
            return _timings_from_text(text, duration=dur)

        from app.adapters.stt_adapter import GroqSTTAdapter
        try:
            ext = "." + s.TTS_AUDIO_FORMAT
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            stt = GroqSTTAdapter()
            transcript = await stt.transcribe(tmp_path)
            os.unlink(tmp_path)
            return [
                WordTimingTTS(word=w.word, start=w.start, end=w.end)
                for w in transcript.words
            ]
        except Exception as exc:
            logger.warning("tts_whisper_fallback_failed",
                           extra={"error": str(exc)})
            return _timings_from_text(text, duration=5.0)


def _minimal_wav(text: str) -> bytes:
    """Generate a tiny valid silent WAV for mock use."""
    sample_rate = 24000
    n_words = max(1, len(text.split()))
    duration_s = n_words * 0.35 + 0.5
    n_samples = int(sample_rate * duration_s)
    pcm = b"\x00\x00" * n_samples
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", data_size,
    )
    return header + pcm


class MockTTSAdapter(TTSAdapter):
    """Deterministic mock: silent WAV with evenly-spaced word timings."""

    async def synthesize(self, text: str,
                         voice: str | None = None) -> TTSResult:
        s = get_settings()
        used_voice = voice or s.TTS_DEFAULT_VOICE
        audio = _minimal_wav(text)
        words = text.split()
        duration = len(words) * 0.35 + 0.5
        timings = _timings_from_text(text, duration)
        return TTSResult(
            audio_bytes=audio, audio_format="wav",
            sample_rate=24000, duration=duration,
            word_timings=timings, voice=used_voice,
        )


_adapter: TTSAdapter | None = None


def get_tts_adapter() -> TTSAdapter:
    global _adapter
    if _adapter is None:
        impl = get_settings().TTS_ADAPTER.lower()
        _adapter = MockTTSAdapter() if impl == "mock" else GroqTTSAdapter()
    return _adapter


def reset_tts_adapter() -> None:
    global _adapter
    _adapter = None
