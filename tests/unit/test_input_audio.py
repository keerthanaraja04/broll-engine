"""
Unit tests for app/pipeline/input_stage/audio_processor.py

Covers:
- validate_audio(): extension allow-list, duration limit
- detect_silence(): all-silence triggers AudioSilentError, partial silence ok
- _parse_silence_intervals(): ffmpeg stderr parsing
- chunk_at_silence(): no-op when file fits, splits on silence midpoints
- transcribe_chunks(): timestamp offsetting across chunks, language detection
- check_transcript_quality(): empty text, low logprob, high no-speech fraction
- process_audio(): full pipeline with all I/O mocked
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("STT_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_adapter_result(
    text: str = "hello world test",
    language: str = "en",
    duration: float = 3.0,
    words: list | None = None,
    segments: list | None = None,
    avg_logprob: float = -0.3,
    no_speech_prob: float = 0.05,
) -> Any:
    from app.adapters.base import (
        TranscriptResult,
        TranscriptSegment,
        WordTiming as AdWord,
    )
    if words is None:
        parts = text.split()
        gap = duration / max(len(parts), 1)
        words = [
            AdWord(word=w, start=round(i * gap, 3), end=round((i + 1) * gap - 0.05, 3))
            for i, w in enumerate(parts)
        ]
    if segments is None:
        segments = [
            TranscriptSegment(
                id=0, text=text, start=0.0, end=duration,
                words=words, avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            )
        ]
    return TranscriptResult(
        text=text, language=language, duration=duration,
        words=words, segments=segments,
    )


FFPROBE_OK = {
    "format": {"duration": "5.0"},
    "streams": [{"codec_type": "audio", "duration": "5.0"}],
}

FFPROBE_LONG = {
    "format": {"duration": "700.0"},
    "streams": [{"codec_type": "audio", "duration": "700.0"}],
}


# ── validate_audio ───────────────────────────────────────────────────────────

class TestValidateAudio:
    def test_accepted_extension_returns_duration(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        with patch("app.pipeline.input_stage.audio_processor._probe_duration",
                   return_value=5.0):
            dur = validate_audio("speech.mp3")
        assert dur == 5.0

    @pytest.mark.parametrize("ext", ["mp3", "wav", "m4a", "flac", "ogg", "webm"])
    def test_all_allowed_extensions_accepted(self, ext: str) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        with patch("app.pipeline.input_stage.audio_processor._probe_duration",
                   return_value=1.0):
            validate_audio(f"file.{ext}")

    def test_rejected_extension_raises_wrong_file_type(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        from app.core.errors import WrongFileTypeError
        with pytest.raises(WrongFileTypeError) as exc_info:
            validate_audio("doc.pdf")
        assert exc_info.value.code == "WRONG_FILE_TYPE"

    def test_exe_rejected(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        from app.core.errors import WrongFileTypeError
        with pytest.raises(WrongFileTypeError):
            validate_audio("bad.exe")

    def test_no_extension_rejected(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        from app.core.errors import WrongFileTypeError
        with pytest.raises(WrongFileTypeError):
            validate_audio("audiofile")

    def test_duration_within_limit_ok(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        with patch("app.pipeline.input_stage.audio_processor._probe_duration",
                   return_value=599.0):
            dur = validate_audio("long.mp3")
        assert dur == 599.0

    def test_duration_exceeds_limit_raises_audio_too_long(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        from app.core.errors import AudioTooLongError
        with patch("app.pipeline.input_stage.audio_processor._probe_duration",
                   return_value=700.0):
            with pytest.raises(AudioTooLongError) as exc_info:
                validate_audio("too_long.mp3")
        err = exc_info.value
        assert err.code == "AUDIO_TOO_LONG"
        assert err.retryable is False
        assert err.details["duration_seconds"] == 700.0

    def test_exact_limit_accepted(self) -> None:
        from app.pipeline.input_stage.audio_processor import validate_audio
        from app.core.config import get_settings
        limit = get_settings().AUDIO_MAX_DURATION_SECONDS
        with patch("app.pipeline.input_stage.audio_processor._probe_duration",
                   return_value=limit):
            validate_audio("exact.mp3")


# ── _parse_silence_intervals ─────────────────────────────────────────────────

class TestParseSilenceIntervals:
    def test_parses_start_and_end(self) -> None:
        from app.pipeline.input_stage.audio_processor import _parse_silence_intervals
        stderr = (
            "[silencedetect] silence_start: 0.5\n"
            "[silencedetect] silence_end: 1.2 | silence_duration: 0.7\n"
            "[silencedetect] silence_start: 3.0\n"
            "[silencedetect] silence_end: 3.8 | silence_duration: 0.8\n"
        )
        intervals = _parse_silence_intervals(stderr)
        assert len(intervals) == 2
        assert intervals[0].start == 0.5
        assert intervals[0].end == 1.2
        assert intervals[1].start == 3.0
        assert intervals[1].end == 3.8

    def test_dangling_start_gets_inf_end(self) -> None:
        from app.pipeline.input_stage.audio_processor import _parse_silence_intervals
        stderr = "[silencedetect] silence_start: 4.5\n"
        intervals = _parse_silence_intervals(stderr)
        assert len(intervals) == 1
        assert intervals[0].start == 4.5
        assert intervals[0].end == float("inf")

    def test_no_silence_returns_empty(self) -> None:
        from app.pipeline.input_stage.audio_processor import _parse_silence_intervals
        assert _parse_silence_intervals("no silence here") == []

    def test_silence_duration_property(self) -> None:
        from app.pipeline.input_stage.audio_processor import SilenceInterval
        iv = SilenceInterval(start=1.0, end=2.5)
        assert iv.duration == pytest.approx(1.5)


# ── detect_silence ───────────────────────────────────────────────────────────

class TestDetectSilence:
    def _make_ffmpeg_result(self, stderr: str, returncode: int = 0) -> MagicMock:
        r = MagicMock()
        r.returncode = returncode
        r.stderr = stderr.encode()
        return r

    def test_partial_silence_returns_intervals(self) -> None:
        from app.pipeline.input_stage.audio_processor import detect_silence
        stderr = (
            "[silencedetect] silence_start: 1.0\n"
            "[silencedetect] silence_end: 1.5 | silence_duration: 0.5\n"
        )
        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=self._make_ffmpeg_result(stderr)):
            ivs = detect_silence("speech.wav", duration=5.0)
        assert len(ivs) == 1

    def test_all_silence_raises_audio_silent(self) -> None:
        from app.pipeline.input_stage.audio_processor import detect_silence
        from app.core.errors import AudioSilentError
        # 9.8 s silent in a 10 s file → 98% → above default 95% threshold
        stderr = (
            "[silencedetect] silence_start: 0.0\n"
            "[silencedetect] silence_end: 9.8 | silence_duration: 9.8\n"
        )
        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=self._make_ffmpeg_result(stderr)):
            with pytest.raises(AudioSilentError) as exc_info:
                detect_silence("silent.wav", duration=10.0)
        err = exc_info.value
        assert err.code == "AUDIO_SILENT"
        assert err.retryable is False
        assert err.details["silent_ratio"] >= 0.95

    def test_borderline_silence_below_threshold_ok(self) -> None:
        """90% silent — below 95% default threshold → should not raise."""
        from app.pipeline.input_stage.audio_processor import detect_silence
        stderr = (
            "[silencedetect] silence_start: 0.0\n"
            "[silencedetect] silence_end: 9.0 | silence_duration: 9.0\n"
        )
        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=self._make_ffmpeg_result(stderr)):
            ivs = detect_silence("mostly_silent.wav", duration=10.0)
        assert len(ivs) == 1

    def test_no_silence_returns_empty_list(self) -> None:
        from app.pipeline.input_stage.audio_processor import detect_silence
        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=self._make_ffmpeg_result("")):
            ivs = detect_silence("speech.wav", duration=5.0)
        assert ivs == []


# ── chunk_at_silence ─────────────────────────────────────────────────────────

class TestChunkAtSilence:
    def test_small_file_returns_single_chunk(self) -> None:
        from app.pipeline.input_stage.audio_processor import (
            chunk_at_silence, AudioChunk,
        )
        from app.core.config import get_settings
        result = chunk_at_silence(
            wav_path="test.wav",
            file_size_bytes=1_000,           # well under limit
            silence_intervals=[],
            total_duration=5.0,
        )
        assert len(result) == 1
        assert result[0].path == "test.wav"
        assert result[0].start_offset == 0.0

    def test_large_file_splits_at_silence(self, tmp_path) -> None:
        from app.pipeline.input_stage.audio_processor import (
            chunk_at_silence, SilenceInterval,
        )
        from app.core.config import get_settings

        # Create a dummy WAV to split
        src = tmp_path / "big.wav"
        src.write_bytes(b"\x00" * 100)

        # Two silences at ~60 s and ~120 s in a 180 s file
        silences = [
            SilenceInterval(start=58.0, end=62.0),   # midpoint = 60
            SilenceInterval(start=118.0, end=122.0),  # midpoint = 120
        ]

        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0

        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=ffmpeg_result):
            chunks = chunk_at_silence(
                wav_path=str(src),
                file_size_bytes=30_000_000,   # > 24 MB limit
                silence_intervals=silences,
                total_duration=180.0,
                output_dir=str(tmp_path),
            )
        assert len(chunks) >= 2
        # First chunk starts at 0
        assert chunks[0].start_offset == 0.0
        # Offsets are monotonically increasing
        offsets = [c.start_offset for c in chunks]
        assert offsets == sorted(offsets)

    def test_chunks_cover_full_duration(self, tmp_path) -> None:
        from app.pipeline.input_stage.audio_processor import (
            chunk_at_silence, SilenceInterval,
        )
        src = tmp_path / "audio.wav"
        src.write_bytes(b"\x00" * 100)

        silences = [SilenceInterval(start=29.0, end=31.0)]
        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0

        with patch("app.pipeline.input_stage.audio_processor._run_ffmpeg",
                   return_value=ffmpeg_result):
            chunks = chunk_at_silence(
                wav_path=str(src),
                file_size_bytes=30_000_000,
                silence_intervals=silences,
                total_duration=60.0,
                output_dir=str(tmp_path),
            )
        total = sum(c.duration for c in chunks)
        assert total == pytest.approx(60.0, abs=1.0)


# ── transcribe_chunks (timestamp merging) ────────────────────────────────────

class TestTranscribeChunks:
    @pytest.mark.asyncio
    async def test_single_chunk_passthrough(self) -> None:
        from app.pipeline.input_stage.audio_processor import (
            transcribe_chunks, AudioChunk,
        )
        mock_result = _make_adapter_result("hello world", duration=2.0)
        with patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_factory:
            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(return_value=mock_result)
            mock_factory.return_value = mock_stt

            result = await transcribe_chunks(
                [AudioChunk(path="a.wav", start_offset=0.0, duration=2.0)]
            )
        assert result.text == "hello world"
        assert len(result.words) == 2
        assert result.words[0].start == pytest.approx(0.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_two_chunks_timestamps_offset(self) -> None:
        from app.pipeline.input_stage.audio_processor import (
            transcribe_chunks, AudioChunk,
        )
        from app.adapters.base import TranscriptResult, TranscriptSegment, WordTiming as AW

        chunk1 = AudioChunk(path="a.wav", start_offset=0.0, duration=5.0)
        chunk2 = AudioChunk(path="b.wav", start_offset=5.0, duration=5.0)

        r1 = TranscriptResult(
            text="hello world",
            language="en", duration=5.0,
            words=[AW(word="hello", start=0.0, end=0.4),
                   AW(word="world", start=0.5, end=0.9)],
            segments=[TranscriptSegment(id=0, text="hello world",
                                         start=0.0, end=0.9,
                                         avg_logprob=-0.2, no_speech_prob=0.01)],
        )
        r2 = TranscriptResult(
            text="foo bar",
            language="en", duration=5.0,
            words=[AW(word="foo", start=0.0, end=0.3),
                   AW(word="bar", start=0.4, end=0.8)],
            segments=[TranscriptSegment(id=0, text="foo bar",
                                         start=0.0, end=0.8,
                                         avg_logprob=-0.3, no_speech_prob=0.02)],
        )

        with patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_f:
            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(side_effect=[r1, r2])
            mock_f.return_value = mock_stt

            merged = await transcribe_chunks([chunk1, chunk2])

        # Total 4 words
        assert len(merged.words) == 4
        # Chunk-2 words offset by 5 s
        foo_word = next(w for w in merged.words if w.word == "foo")
        assert foo_word.start == pytest.approx(5.0, abs=0.01)
        bar_word = next(w for w in merged.words if w.word == "bar")
        assert bar_word.start == pytest.approx(5.4, abs=0.01)

    @pytest.mark.asyncio
    async def test_segment_ids_non_overlapping(self) -> None:
        from app.pipeline.input_stage.audio_processor import (
            transcribe_chunks, AudioChunk,
        )
        r1 = _make_adapter_result("one two", duration=2.0)
        r2 = _make_adapter_result("three four", duration=2.0)
        r1.segments[0].id = 0
        r2.segments[0].id = 0  # same id in chunk — should be renumbered

        with patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_f:
            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(side_effect=[r1, r2])
            mock_f.return_value = mock_stt

            merged = await transcribe_chunks([
                AudioChunk("a.wav", 0.0, 2.0),
                AudioChunk("b.wav", 2.0, 2.0),
            ])

        seg_ids = [s.id for s in merged.segments]
        assert seg_ids == sorted(set(seg_ids)), "Segment IDs must be unique"

    @pytest.mark.asyncio
    async def test_language_detected_from_non_english_chunk(self) -> None:
        from app.pipeline.input_stage.audio_processor import (
            transcribe_chunks, AudioChunk,
        )
        r1 = _make_adapter_result("bonjour monde", language="fr", duration=2.0)
        with patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_f:
            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(return_value=r1)
            mock_f.return_value = mock_stt

            result = await transcribe_chunks(
                [AudioChunk("a.wav", 0.0, 2.0)], language=None
            )
        assert result.language == "fr"


# ── check_transcript_quality ─────────────────────────────────────────────────

class TestCheckTranscriptQuality:
    def test_good_transcript_returns_no_warnings(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        r = _make_adapter_result("hello world this is fine", avg_logprob=-0.2)
        warnings = check_transcript_quality(r)
        assert warnings == []

    def test_empty_text_raises_audio_no_speech(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        from app.core.errors import AudioNoSpeechError
        r = _make_adapter_result("")
        r.text = ""
        with pytest.raises(AudioNoSpeechError) as exc_info:
            check_transcript_quality(r)
        assert exc_info.value.code == "AUDIO_NO_SPEECH"

    def test_whitespace_only_text_raises(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        from app.core.errors import AudioNoSpeechError
        r = _make_adapter_result("   \n  ")
        r.text = "   "
        with pytest.raises(AudioNoSpeechError):
            check_transcript_quality(r)

    def test_low_avg_logprob_raises(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        from app.core.errors import AudioNoSpeechError
        r = _make_adapter_result("blah", avg_logprob=-2.0)
        with pytest.raises(AudioNoSpeechError) as exc_info:
            check_transcript_quality(r)
        assert "avg_logprob" in str(exc_info.value.details)

    def test_high_no_speech_fraction_raises(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        from app.core.errors import AudioNoSpeechError
        from app.adapters.base import TranscriptResult, TranscriptSegment, WordTiming as AW

        # 3 out of 4 segments have high no_speech_prob → 75% > 70% threshold
        segs = [
            TranscriptSegment(id=i, text="x", start=float(i), end=float(i+1),
                               avg_logprob=-0.2, no_speech_prob=0.95)
            for i in range(3)
        ]
        segs.append(TranscriptSegment(id=3, text="real speech",
                                       start=3.0, end=4.0,
                                       avg_logprob=-0.2, no_speech_prob=0.05))
        words = [AW(word="x", start=0.0, end=0.1)]
        r = TranscriptResult(
            text="x x x real speech", language="en",
            duration=4.0, words=words, segments=segs,
        )
        with pytest.raises(AudioNoSpeechError):
            check_transcript_quality(r)

    def test_moderate_no_speech_adds_warning(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        from app.adapters.base import TranscriptResult, TranscriptSegment, WordTiming as AW

        # 2 out of 5 segments noisy → 40% > 30% → warning only
        segs = [
            TranscriptSegment(id=0, text="good", start=0.0, end=1.0,
                               avg_logprob=-0.2, no_speech_prob=0.1),
            TranscriptSegment(id=1, text="good2", start=1.0, end=2.0,
                               avg_logprob=-0.2, no_speech_prob=0.1),
            TranscriptSegment(id=2, text="noisy", start=2.0, end=3.0,
                               avg_logprob=-0.2, no_speech_prob=0.9),
            TranscriptSegment(id=3, text="good3", start=3.0, end=4.0,
                               avg_logprob=-0.2, no_speech_prob=0.1),
            TranscriptSegment(id=4, text="noisy2", start=4.0, end=5.0,
                               avg_logprob=-0.2, no_speech_prob=0.9),
        ]
        words = [AW(word="w", start=0.0, end=0.1)]
        r = TranscriptResult(
            text="good good2 noisy good3 noisy2",
            language="en", duration=5.0, words=words, segments=segs,
        )
        warnings = check_transcript_quality(r)
        assert len(warnings) > 0
        assert any("no_speech" in w.lower() for w in warnings)

    def test_few_words_adds_warning(self) -> None:
        from app.pipeline.input_stage.audio_processor import check_transcript_quality
        r = _make_adapter_result("hi", duration=0.5)
        # Override words to be just 1
        from app.adapters.base import WordTiming as AW
        r.words = [AW(word="hi", start=0.0, end=0.3)]
        r.text = "hi"
        warnings = check_transcript_quality(r)
        assert any("word" in w.lower() for w in warnings)


# ── process_audio (full pipeline) ────────────────────────────────────────────

class TestProcessAudio:
    @pytest.mark.asyncio
    async def test_returns_transcript_with_source_audio(self) -> None:
        from app.pipeline.input_stage.audio_processor import process_audio

        mock_result = _make_adapter_result("hello world", duration=3.0)

        with patch("app.pipeline.input_stage.audio_processor.validate_audio",
                   return_value=3.0), \
             patch("app.pipeline.input_stage.audio_processor.convert_to_wav",
                   return_value="/tmp/test_16k.wav"), \
             patch("os.path.getsize", return_value=1_000), \
             patch("app.pipeline.input_stage.audio_processor.detect_silence",
                   return_value=[]), \
             patch("app.pipeline.input_stage.audio_processor.chunk_at_silence",
                   return_value=[
                       __import__(
                           "app.pipeline.input_stage.audio_processor",
                           fromlist=["AudioChunk"]
                       ).AudioChunk(path="/tmp/test_16k.wav",
                                     start_offset=0.0, duration=3.0)
                   ]), \
             patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_f:

            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(return_value=mock_result)
            mock_f.return_value = mock_stt

            transcript = await process_audio("speech.wav")

        assert transcript.source == "audio"
        assert transcript.language == "en"
        assert transcript.word_count == 2
        assert len(transcript.sentences) >= 1
        assert transcript.audio_path == "/tmp/test_16k.wav"

    @pytest.mark.asyncio
    async def test_silence_error_propagates(self) -> None:
        from app.pipeline.input_stage.audio_processor import process_audio
        from app.core.errors import AudioSilentError

        with patch("app.pipeline.input_stage.audio_processor.validate_audio",
                   return_value=10.0), \
             patch("app.pipeline.input_stage.audio_processor.convert_to_wav",
                   return_value="/tmp/silent.wav"), \
             patch("os.path.getsize", return_value=1_000), \
             patch("app.pipeline.input_stage.audio_processor.detect_silence",
                   side_effect=AudioSilentError("all silent")):
            with pytest.raises(AudioSilentError):
                await process_audio("silent.wav")

    @pytest.mark.asyncio
    async def test_no_speech_error_propagates(self) -> None:
        from app.pipeline.input_stage.audio_processor import process_audio, AudioChunk
        from app.core.errors import AudioNoSpeechError

        empty_result = _make_adapter_result("")
        empty_result.text = ""

        with patch("app.pipeline.input_stage.audio_processor.validate_audio",
                   return_value=2.0), \
             patch("app.pipeline.input_stage.audio_processor.convert_to_wav",
                   return_value="/tmp/nospeech.wav"), \
             patch("os.path.getsize", return_value=1_000), \
             patch("app.pipeline.input_stage.audio_processor.detect_silence",
                   return_value=[]), \
             patch("app.pipeline.input_stage.audio_processor.chunk_at_silence",
                   return_value=[AudioChunk("/tmp/nospeech.wav", 0.0, 2.0)]), \
             patch("app.pipeline.input_stage.audio_processor.get_stt_adapter") as mock_f:

            mock_stt = MagicMock()
            mock_stt.transcribe = AsyncMock(return_value=empty_result)
            mock_f.return_value = mock_stt

            with pytest.raises(AudioNoSpeechError):
                await process_audio("nospeech.wav")
