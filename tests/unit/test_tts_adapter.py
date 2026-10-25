"""
Unit tests for TTSAdapter — synthesis, word timings, Whisper fallback.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("TTS_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


def _is_valid_wav(data: bytes) -> bool:
    return (len(data) > 44 and data[:4] == b"RIFF"
            and data[8:12] == b"WAVE")


class TestMockTTSAdapter:
    @pytest.mark.asyncio
    async def test_returns_tts_result(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("Hello world")
        assert r.audio_bytes
        assert r.audio_format == "wav"
        assert r.sample_rate > 0
        assert r.duration > 0
        assert isinstance(r.word_timings, list)
        assert isinstance(r.voice, str)

    @pytest.mark.asyncio
    async def test_audio_is_valid_wav(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("The quick brown fox")
        assert _is_valid_wav(r.audio_bytes)

    @pytest.mark.asyncio
    async def test_word_timings_count_matches(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        text = "one two three four"
        r = await MockTTSAdapter().synthesize(text)
        assert len(r.word_timings) == len(text.split())

    @pytest.mark.asyncio
    async def test_word_timings_ordered(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("alpha beta gamma delta")
        starts = [w.start for w in r.word_timings]
        assert starts == sorted(starts)

    @pytest.mark.asyncio
    async def test_word_end_gt_start(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("testing timing values here")
        for w in r.word_timings:
            assert w.end > w.start

    @pytest.mark.asyncio
    async def test_voice_override(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("hello", voice="custom")
        assert r.voice == "custom"

    @pytest.mark.asyncio
    async def test_default_voice(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        from app.core.config import get_settings
        r = await MockTTSAdapter().synthesize("hello")
        assert r.voice == get_settings().TTS_DEFAULT_VOICE

    @pytest.mark.asyncio
    async def test_empty_text_handled(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        r = await MockTTSAdapter().synthesize("")
        assert isinstance(r.audio_bytes, bytes)
        assert r.word_timings == []

    @pytest.mark.asyncio
    async def test_longer_text_longer_audio(self) -> None:
        from app.adapters.tts_adapter import MockTTSAdapter
        a = MockTTSAdapter()
        r_short = await a.synthesize("hi")
        r_long = await a.synthesize(
            "this is a much longer sentence with many words")
        assert len(r_long.audio_bytes) > len(r_short.audio_bytes)


class TestTTSFactory:
    def test_mock_selector(self) -> None:
        from app.adapters.tts_adapter import (
            MockTTSAdapter, get_tts_adapter, reset_tts_adapter)
        reset_tts_adapter()
        assert isinstance(get_tts_adapter(), MockTTSAdapter)
        reset_tts_adapter()

    def test_singleton(self) -> None:
        from app.adapters.tts_adapter import get_tts_adapter, reset_tts_adapter
        reset_tts_adapter()
        assert get_tts_adapter() is get_tts_adapter()
        reset_tts_adapter()


class TestTimingsFromText:
    def test_evenly_spaced(self) -> None:
        from app.adapters.tts_adapter import _timings_from_text
        t = _timings_from_text("a b c", duration=3.0)
        assert len(t) == 3
        assert t[0].start < t[1].start < t[2].start

    def test_empty_text(self) -> None:
        from app.adapters.tts_adapter import _timings_from_text
        assert _timings_from_text("", duration=2.0) == []


class TestGroqTTSWhisperFallback:
    @pytest.mark.asyncio
    async def test_whisper_fallback_called(self) -> None:
        from app.adapters.tts_adapter import GroqTTSAdapter
        from app.adapters.base import TranscriptResult, WordTiming

        mock_transcript = TranscriptResult(
            text="hello world", language="en", duration=1.0,
            words=[WordTiming(word="hello", start=0.0, end=0.4),
                   WordTiming(word="world", start=0.5, end=0.9)],
            segments=[],
        )
        fake_audio = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100

        with patch.dict(os.environ, {"TTS_FALLBACK_WORD_TIMINGS": "true",
                                     "TTS_ADAPTER": "groq"}):
            from app.core.config import get_settings
            get_settings.cache_clear()

            async def _fake_tts(self_ref: Any, text: str,
                                voice: str) -> bytes:
                return fake_audio

            from app.adapters import stt_adapter as stt_mod

            async def _fake_stt(self_ref: Any, path: str,
                                lang: Any = None) -> Any:
                return mock_transcript

            with patch.object(stt_mod.GroqSTTAdapter, "transcribe",
                              _fake_stt), \
                 patch.object(GroqTTSAdapter, "_call_groq_tts", _fake_tts):
                r = await GroqTTSAdapter().synthesize(
                    "hello world", voice="test")
            assert len(r.word_timings) == 2
            assert r.word_timings[0].word == "hello"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_synthetic_timings_when_whisper_fails(self) -> None:
        from app.adapters.tts_adapter import GroqTTSAdapter
        fake_audio = b"RIFF\x00\x00\x00\x00WAVEfmt " + b"\x00" * 100

        with patch.dict(os.environ, {"TTS_FALLBACK_WORD_TIMINGS": "true",
                                     "TTS_ADAPTER": "groq"}):
            from app.core.config import get_settings
            get_settings.cache_clear()

            async def _fake_tts(self_ref: Any, text: str,
                                voice: str) -> bytes:
                return fake_audio

            from app.adapters import stt_adapter as stt_mod

            async def _broken_stt(self_ref: Any, path: str,
                                 lang: Any = None) -> Any:
                raise RuntimeError("whisper unavailable")

            with patch.object(stt_mod.GroqSTTAdapter, "transcribe",
                              _broken_stt), \
                 patch.object(GroqTTSAdapter, "_call_groq_tts", _fake_tts):
                r = await GroqTTSAdapter().synthesize("alpha beta gamma")
            assert len(r.word_timings) == 3
        get_settings.cache_clear()
