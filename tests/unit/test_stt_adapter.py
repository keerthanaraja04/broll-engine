"""
Unit tests for STTAdapter — runs with MockSTTAdapter (no API keys).
Also tests GroqSTTAdapter response parser and cache helpers in isolation.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("STT_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")

from tests.fixtures import AUDIO_WAV


class TestMockSTTAdapter:
    @pytest.mark.asyncio
    async def test_returns_transcript_result(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        result = await MockSTTAdapter().transcribe(AUDIO_WAV)
        assert result.text
        assert isinstance(result.words, list)
        assert isinstance(result.segments, list)
        assert isinstance(result.language, str)
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_same_path_same_result(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        a = MockSTTAdapter()
        r1 = await a.transcribe(AUDIO_WAV)
        r2 = await a.transcribe(AUDIO_WAV)
        assert r1.text == r2.text
        assert r1.language == r2.language
        assert len(r1.words) == len(r2.words)

    @pytest.mark.asyncio
    async def test_language_hint_propagated(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        result = await MockSTTAdapter().transcribe(AUDIO_WAV, language="hi")
        assert result.language == "hi"

    @pytest.mark.asyncio
    async def test_default_language_is_en(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        result = await MockSTTAdapter().transcribe(AUDIO_WAV)
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_word_timings_ordered(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        result = await MockSTTAdapter().transcribe(AUDIO_WAV)
        starts = [w.start for w in result.words]
        assert starts == sorted(starts)

    @pytest.mark.asyncio
    async def test_word_end_gt_start(self) -> None:
        from app.adapters.stt_adapter import MockSTTAdapter
        result = await MockSTTAdapter().transcribe(AUDIO_WAV)
        for w in result.words:
            assert w.end > w.start


class TestSTTFactory:
    def test_mock_selector(self) -> None:
        from app.adapters.stt_adapter import (
            MockSTTAdapter, get_stt_adapter, reset_stt_adapter)
        reset_stt_adapter()
        assert isinstance(get_stt_adapter(), MockSTTAdapter)
        reset_stt_adapter()

    def test_singleton(self) -> None:
        from app.adapters.stt_adapter import get_stt_adapter, reset_stt_adapter
        reset_stt_adapter()
        assert get_stt_adapter() is get_stt_adapter()
        reset_stt_adapter()


class TestSTTResponseParser:
    def test_parse_full_response(self) -> None:
        from app.adapters.stt_adapter import _parse_groq_response
        body = {
            "text": "hello world", "language": "en", "duration": 2.5,
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.4},
                {"word": "world", "start": 0.5, "end": 0.9},
            ],
            "segments": [
                {"id": 0, "text": "hello world", "start": 0.0, "end": 0.9,
                 "avg_logprob": -0.2, "no_speech_prob": 0.01},
            ],
        }
        r = _parse_groq_response(body)
        assert r.text == "hello world"
        assert r.language == "en"
        assert len(r.words) == 2
        assert r.words[0].word == "hello"

    def test_parse_missing_words_key(self) -> None:
        from app.adapters.stt_adapter import _parse_groq_response
        body = {"text": "t", "language": "en", "duration": 1.0,
                "segments": []}
        assert _parse_groq_response(body).words == []

    def test_parse_empty_word_strings_filtered(self) -> None:
        from app.adapters.stt_adapter import _parse_groq_response
        body = {
            "text": "hi", "language": "en", "duration": 1.0,
            "words": [
                {"word": "  ", "start": 0.0, "end": 0.3},
                {"word": "hi", "start": 0.3, "end": 0.7},
            ],
            "segments": [],
        }
        assert len(_parse_groq_response(body).words) == 1


class TestSTTCacheHelpers:
    def test_roundtrip(self) -> None:
        from app.adapters.stt_adapter import _to_cache, _from_cache
        from app.adapters.base import (
            TranscriptResult, TranscriptSegment, WordTiming)
        orig = TranscriptResult(
            text="hello world", language="en", duration=1.5,
            words=[WordTiming(word="hello", start=0.0, end=0.4),
                   WordTiming(word="world", start=0.5, end=0.9)],
            segments=[TranscriptSegment(
                id=0, text="hello world", start=0.0, end=0.9)])
        restored = _from_cache(_to_cache(orig))
        assert restored.text == orig.text
        assert len(restored.words) == 2
        assert restored.words[0].word == "hello"
