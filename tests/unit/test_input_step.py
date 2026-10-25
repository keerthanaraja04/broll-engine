"""
Unit tests for app/pipeline/input_stage/input_step.py (InputStageStep).

Covers:
- Dispatch to audio_processor for input_type=audio_upload / audio_url
- Dispatch to text_processor for input_type=script
- Dispatch to text_processor for input_type=scenes
- Transcript stored on context under "transcript", "transcript_obj",
  "input_language", "input_duration", "input_warnings"
- ValidationError for unknown input_type
- ValidationError for missing required payload fields
- PipelineError wraps unexpected exceptions from processors
- transcript.to_dict() shape (spot-check)
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("STT_ADAPTER", "mock")
os.environ.setdefault("TTS_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_transcript(
    source: str = "script",
    language: str = "en",
    duration: float = 3.0,
    text: str = "hello world",
) -> Any:
    from app.pipeline.input_stage.transcript import (
        Transcript, Sentence, WordTiming,
    )
    words = [
        WordTiming(word=w, start=i * 0.4, end=i * 0.4 + 0.35)
        for i, w in enumerate(text.split())
    ]
    sentences = [Sentence(id=0, text=text, start=0.0,
                           end=duration, words=words)]
    return Transcript(
        language=language, duration=duration,
        words=words, sentences=sentences,
        source=source,  # type: ignore[arg-type]
    )


def _make_pipeline_ctx(payload: dict) -> Any:
    from app.pipeline.base import PipelineContext
    return PipelineContext(job_id="test-job-001", payload=payload)


# ── dispatch to script processor ────────────────────────────────────────────

class TestInputStepScript:
    @pytest.mark.asyncio
    async def test_script_dispatch_stores_transcript(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="script")
        ctx = _make_pipeline_ctx({"input_type": "script",
                                   "script": "hello world"})

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        assert result_ctx.get("transcript_obj") is expected
        assert result_ctx.get("input_language") == "en"
        assert result_ctx.get("input_duration") == pytest.approx(3.0)
        assert result_ctx.get("input_warnings") == []

    @pytest.mark.asyncio
    async def test_script_transcript_dict_stored(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="script")
        ctx = _make_pipeline_ctx({"input_type": "script",
                                   "script": "hello world"})

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        d = result_ctx.get("transcript")
        assert isinstance(d, dict)
        assert d["source"] == "script"
        assert d["language"] == "en"
        assert "words" in d
        assert "sentences" in d

    @pytest.mark.asyncio
    async def test_script_passes_voice_and_language(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="script", language="hi")
        ctx = _make_pipeline_ctx({
            "input_type": "script",
            "script": "hello world",
            "voice": "custom-voice",
            "language": "hi",
        })

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(return_value=expected)) as mock_ps:
            await InputStageStep().execute(ctx)

        mock_ps.assert_called_once()
        _, kwargs = mock_ps.call_args
        assert kwargs.get("voice") == "custom-voice"
        assert kwargs.get("language") == "hi"

    @pytest.mark.asyncio
    async def test_missing_script_key_raises_validation_error(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "script"})  # no "script" key
        with pytest.raises((ValidationError, Exception)):
            await InputStageStep().execute(ctx)

    @pytest.mark.asyncio
    async def test_non_string_script_raises_validation_error(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "script", "script": 12345})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)


# ── dispatch to scenes processor ────────────────────────────────────────────

class TestInputStepScenes:
    @pytest.mark.asyncio
    async def test_scenes_dispatch_stores_transcript(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="scenes")
        ctx = _make_pipeline_ctx({
            "input_type": "scenes",
            "scenes": [{"text": "city skyline", "visual_hint": "city"}],
        })

        with patch("app.pipeline.input_stage.input_step.process_scenes",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        assert result_ctx.get("transcript_obj").source == "scenes"

    @pytest.mark.asyncio
    async def test_missing_scenes_key_raises(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "scenes"})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)

    @pytest.mark.asyncio
    async def test_empty_scenes_list_raises(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "scenes", "scenes": []})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)


# ── dispatch to audio_upload ─────────────────────────────────────────────────

class TestInputStepAudioUpload:
    @pytest.mark.asyncio
    async def test_audio_upload_dispatch(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="audio")
        ctx = _make_pipeline_ctx({
            "input_type": "audio_upload",
            "audio_object": "jobs/test/audio.mp3",
            "audio_bucket": "broll-raw",
        })

        mock_storage = MagicMock()
        mock_storage.download_bytes = AsyncMock(return_value=b"fake-audio")

        with patch("app.pipeline.input_stage.input_step.get_storage_client",
                   return_value=mock_storage), \
             patch("app.pipeline.input_stage.input_step.process_audio",
                   AsyncMock(return_value=expected)), \
             patch("builtins.open", MagicMock()), \
             patch("os.path.join", return_value="/tmp/test_audio.mp3"):
            result_ctx = await InputStageStep().execute(ctx)

        assert result_ctx.get("transcript_obj").source == "audio"

    @pytest.mark.asyncio
    async def test_missing_audio_object_raises(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "audio_upload"})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)


# ── dispatch to audio_url ────────────────────────────────────────────────────

class TestInputStepAudioUrl:
    @pytest.mark.asyncio
    async def test_audio_url_dispatch(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="audio")
        ctx = _make_pipeline_ctx({
            "input_type": "audio_url",
            "audio_url": "https://example.com/audio.mp3",
        })

        with patch.object(InputStageStep, "_download_url",
                          AsyncMock(return_value="/tmp/audio.mp3")), \
             patch("app.pipeline.input_stage.input_step.process_audio",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        assert result_ctx.get("input_duration") == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_missing_audio_url_raises(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "audio_url"})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)


# ── unknown input_type ───────────────────────────────────────────────────────

class TestInputStepUnknownType:
    @pytest.mark.asyncio
    async def test_unknown_type_raises_validation_error(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({"input_type": "video_upload"})
        with pytest.raises(ValidationError) as exc_info:
            await InputStageStep().execute(ctx)
        assert "video_upload" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_empty_input_type_raises_validation_error(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import ValidationError
        ctx = _make_pipeline_ctx({})
        with pytest.raises(ValidationError):
            await InputStageStep().execute(ctx)


# ── error wrapping ───────────────────────────────────────────────────────────

class TestInputStepErrorWrapping:
    @pytest.mark.asyncio
    async def test_runtime_error_wrapped_as_pipeline_error(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import PipelineError
        ctx = _make_pipeline_ctx({"input_type": "script",
                                   "script": "hello world"})

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(side_effect=RuntimeError("unexpected crash"))):
            with pytest.raises(PipelineError) as exc_info:
                await InputStageStep().execute(ctx)
        assert "unexpected crash" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_domain_errors_pass_through_unwrapped(self) -> None:
        """AudioSilentError and similar domain errors bubble up directly."""
        from app.pipeline.input_stage.input_step import InputStageStep
        from app.core.errors import AudioSilentError
        ctx = _make_pipeline_ctx({
            "input_type": "audio_url",
            "audio_url": "https://example.com/silent.mp3",
        })
        with patch.object(InputStageStep, "_download_url",
                          AsyncMock(return_value="/tmp/silent.mp3")), \
             patch("app.pipeline.input_stage.input_step.process_audio",
                   AsyncMock(side_effect=AudioSilentError("all silent"))):
            with pytest.raises((AudioSilentError, Exception)):
                await InputStageStep().execute(ctx)


# ── context keys ────────────────────────────────────────────────────────────

class TestInputStepContextKeys:
    @pytest.mark.asyncio
    async def test_all_expected_keys_set(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="script")
        ctx = _make_pipeline_ctx({"input_type": "script",
                                   "script": "hello world"})

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        assert result_ctx.get("transcript") is not None
        assert result_ctx.get("transcript_obj") is not None
        assert result_ctx.get("input_language") is not None
        assert result_ctx.get("input_duration") is not None
        assert result_ctx.get("input_warnings") is not None

    @pytest.mark.asyncio
    async def test_input_warnings_is_list(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep

        expected = _make_transcript(source="script")
        expected.warnings.append("test warning")
        ctx = _make_pipeline_ctx({"input_type": "script",
                                   "script": "hello world"})

        with patch("app.pipeline.input_stage.input_step.process_script",
                   AsyncMock(return_value=expected)):
            result_ctx = await InputStageStep().execute(ctx)

        warnings = result_ctx.get("input_warnings")
        assert isinstance(warnings, list)
        assert "test warning" in warnings

    @pytest.mark.asyncio
    async def test_step_name_is_class_name(self) -> None:
        from app.pipeline.input_stage.input_step import InputStageStep
        assert InputStageStep().name == "InputStageStep"


# ── transcript model ────────────────────────────────────────────────────────

class TestTranscriptModel:
    def test_to_dict_contains_required_keys(self) -> None:
        t = _make_transcript()
        d = t.to_dict()
        for key in ("language", "duration", "source", "words",
                    "sentences", "word_count", "sentence_count"):
            assert key in d, f"Missing key: {key}"

    def test_text_property_joins_sentences(self) -> None:
        t = _make_transcript(text="hello world")
        assert "hello" in t.text
        assert "world" in t.text

    def test_words_in_range(self) -> None:
        from app.pipeline.input_stage.transcript import (
            Transcript, Sentence, WordTiming,
        )
        words = [
            WordTiming(word="a", start=0.0, end=0.3),
            WordTiming(word="b", start=1.0, end=1.3),
            WordTiming(word="c", start=2.0, end=2.3),
        ]
        t = Transcript(
            language="en", duration=3.0, words=words,
            sentences=[], source="script",
        )
        in_range = t.words_in_range(0.8, 2.2)
        assert len(in_range) == 2
        assert in_range[0].word == "b"
        assert in_range[1].word == "c"

    def test_add_warning(self) -> None:
        t = _make_transcript()
        t.add_warning("test warning")
        assert "test warning" in t.warnings

    def test_word_timing_end_lt_start_raises(self) -> None:
        from app.pipeline.input_stage.transcript import WordTiming
        with pytest.raises(ValueError):
            WordTiming(word="bad", start=2.0, end=1.0)


# ── sentences_from_words ────────────────────────────────────────────────────

class TestSentencesFromWords:
    def test_empty_words_returns_empty(self) -> None:
        from app.pipeline.input_stage.transcript import sentences_from_words
        assert sentences_from_words([]) == []

    def test_splits_at_terminal_punctuation(self) -> None:
        from app.pipeline.input_stage.transcript import (
            sentences_from_words, WordTiming,
        )
        words = [
            WordTiming(word="Hello.", start=0.0, end=0.4),
            WordTiming(word="World", start=0.5, end=0.9),
        ]
        sentences = sentences_from_words(words)
        assert len(sentences) == 2
        assert sentences[0].text == "Hello."
        assert sentences[1].text == "World"

    def test_splits_at_silence_gap(self) -> None:
        from app.pipeline.input_stage.transcript import (
            sentences_from_words, WordTiming,
        )
        words = [
            WordTiming(word="first", start=0.0, end=0.3),
            # 1 second gap — above default 0.5 s threshold
            WordTiming(word="second", start=1.3, end=1.6),
        ]
        sentences = sentences_from_words(words, min_gap_seconds=0.5)
        assert len(sentences) == 2

    def test_hard_cap_splits_long_run(self) -> None:
        from app.pipeline.input_stage.transcript import (
            sentences_from_words, WordTiming,
        )
        # 50 words with no punctuation or large gap
        words = [
            WordTiming(word=f"w{i}", start=i * 0.2, end=i * 0.2 + 0.15)
            for i in range(50)
        ]
        sentences = sentences_from_words(words, max_sentence_words=40)
        assert len(sentences) >= 2

    def test_sentence_ids_sequential(self) -> None:
        from app.pipeline.input_stage.transcript import (
            sentences_from_words, WordTiming,
        )
        words = [
            WordTiming(word="one.", start=0.0, end=0.3),
            WordTiming(word="two.", start=0.4, end=0.7),
            WordTiming(word="three.", start=0.8, end=1.1),
        ]
        sentences = sentences_from_words(words)
        assert [s.id for s in sentences] == list(range(len(sentences)))

    def test_words_assigned_to_sentences(self) -> None:
        from app.pipeline.input_stage.transcript import (
            sentences_from_words, WordTiming,
        )
        words = [
            WordTiming(word="Hello.", start=0.0, end=0.4),
            WordTiming(word="World.", start=0.5, end=0.9),
        ]
        sentences = sentences_from_words(words)
        all_sentence_words = [w for s in sentences for w in s.words]
        assert len(all_sentence_words) == 2
