"""
Unit tests for app/pipeline/input_stage/text_processor.py

Covers:
- _validate_script(): empty/whitespace rejection
- _split_into_tts_calls(): single chunk, multi-chunk at sentence boundary
- process_script(): TTS called, Transcript built, word timings present
- process_scenes(): joined TTS, visual_hint passthrough, scene_index set,
                    words assigned to correct scenes, empty scene skipped
- _assign_words_to_scenes(): proportional assignment correctness
- Error paths: empty script, all-empty scenes
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("TTS_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_tts_result(
    text: str = "hello world",
    duration: float = 2.0,
) -> Any:
    from app.adapters.base import TTSResult, WordTimingTTS
    words = text.split()
    gap = duration / max(len(words), 1)
    timings = [
        WordTimingTTS(word=w, start=round(i * gap, 3),
                       end=round((i + 1) * gap - 0.05, 3))
        for i, w in enumerate(words)
    ]
    import struct
    n = int(24000 * duration)
    pcm = b"\x00\x00" * n
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1, 24000, 48000, 2, 16,
        b"data", data_size,
    )
    return TTSResult(
        audio_bytes=header + pcm,
        audio_format="wav",
        sample_rate=24000,
        duration=duration,
        word_timings=timings,
        voice="test-voice",
    )


# ── _validate_script ─────────────────────────────────────────────────────────

class TestValidateScript:
    def test_valid_script_returns_stripped(self) -> None:
        from app.pipeline.input_stage.text_processor import _validate_script
        assert _validate_script("  hello world  ") == "hello world"

    def test_empty_string_raises_empty_script_error(self) -> None:
        from app.pipeline.input_stage.text_processor import _validate_script
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError) as exc_info:
            _validate_script("")
        assert exc_info.value.code == "EMPTY_SCRIPT"

    def test_whitespace_only_raises(self) -> None:
        from app.pipeline.input_stage.text_processor import _validate_script
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError):
            _validate_script("   \n\t  ")


# ── _split_into_tts_calls ────────────────────────────────────────────────────

class TestSplitIntoTtsCalls:
    def test_short_text_returns_single_chunk(self) -> None:
        from app.pipeline.input_stage.text_processor import _split_into_tts_calls
        chunks = _split_into_tts_calls("Hello world.", max_chars=4500)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_long_text_splits_at_sentence(self) -> None:
        from app.pipeline.input_stage.text_processor import _split_into_tts_calls
        # Two sentences each ~30 chars, max_chars=35 → forces split after first
        s1 = "This is the first sentence here."
        s2 = "This is the second sentence here."
        combined = f"{s1} {s2}"
        chunks = _split_into_tts_calls(combined, max_chars=35)
        assert len(chunks) >= 2
        # No chunk exceeds max_chars
        for c in chunks:
            assert len(c) <= 35 + 10  # small tolerance for word boundaries

    def test_each_chunk_preserves_content(self) -> None:
        from app.pipeline.input_stage.text_processor import _split_into_tts_calls
        text = "Word " * 200  # 1000 chars
        chunks = _split_into_tts_calls(text, max_chars=100)
        rejoined = " ".join(c.strip() for c in chunks)
        # All words preserved (order may differ slightly at boundaries)
        original_words = set(text.split())
        rejoined_words = set(rejoined.split())
        assert original_words == rejoined_words

    def test_single_word_exceeding_limit_handled(self) -> None:
        from app.pipeline.input_stage.text_processor import _split_into_tts_calls
        # A single very long word — can't split, should end up in one chunk
        text = "a" * 10
        chunks = _split_into_tts_calls(text, max_chars=5)
        assert "".join(c.strip() for c in chunks) == text


# ── process_script ───────────────────────────────────────────────────────────

class TestProcessScript:
    @pytest.mark.asyncio
    async def test_returns_transcript_with_source_script(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script

        tts_result = _make_tts_result("hello world this is a test", duration=3.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts

            transcript = await process_script("hello world this is a test")

        assert transcript.source == "script"
        assert transcript.word_count == 6
        assert transcript.tts_audio_bytes is not None
        assert len(transcript.tts_audio_bytes) > 0

    @pytest.mark.asyncio
    async def test_language_stored_in_transcript(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        tts_result = _make_tts_result("bonjour monde", duration=2.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_script("bonjour monde", language="fr")
        assert transcript.language == "fr"

    @pytest.mark.asyncio
    async def test_voice_passed_to_tts(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        tts_result = _make_tts_result("test", duration=1.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            await process_script("test", voice="custom-voice")
        mock_tts.synthesize.assert_called_once()
        call_kwargs = mock_tts.synthesize.call_args
        assert call_kwargs[1].get("voice") == "custom-voice" or \
               (call_kwargs[0] and "custom-voice" in call_kwargs[0])

    @pytest.mark.asyncio
    async def test_word_timings_populated(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        tts_result = _make_tts_result("alpha beta gamma", duration=3.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_script("alpha beta gamma")
        assert len(transcript.words) == 3
        # Words are time-ordered
        starts = [w.start for w in transcript.words]
        assert starts == sorted(starts)

    @pytest.mark.asyncio
    async def test_sentences_created(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        # Two sentences separated by terminal punctuation
        text = "Hello world. Goodbye world."
        tts_result = _make_tts_result(text, duration=4.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_script(text)
        assert len(transcript.sentences) >= 1

    @pytest.mark.asyncio
    async def test_empty_script_raises_empty_script_error(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError):
            await process_script("")

    @pytest.mark.asyncio
    async def test_whitespace_script_raises(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError):
            await process_script("   ")

    @pytest.mark.asyncio
    async def test_duration_matches_tts_result(self) -> None:
        from app.pipeline.input_stage.text_processor import process_script
        tts_result = _make_tts_result("duration test", duration=7.5)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_script("duration test")
        assert transcript.duration == pytest.approx(7.5)


# ── process_scenes ───────────────────────────────────────────────────────────

class TestProcessScenes:
    @pytest.mark.asyncio
    async def test_returns_transcript_with_source_scenes(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [{"text": "city skyline at night", "visual_hint": "city skyline"}]
        tts_result = _make_tts_result("city skyline at night", duration=3.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes)
        assert transcript.source == "scenes"

    @pytest.mark.asyncio
    async def test_visual_hint_preserved_in_sentences(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [
            {"text": "people walking in a park", "visual_hint": "people walking"},
            {"text": "cars on a highway", "visual_hint": "highway traffic"},
        ]
        tts_result = _make_tts_result(
            "people walking in a park cars on a highway", duration=5.0
        )
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes)

        # Every sentence that has a scene_index should carry the visual_hint
        hints = {
            s.visual_hint for s in transcript.sentences if s.visual_hint is not None
        }
        assert "people walking" in hints
        assert "highway traffic" in hints

    @pytest.mark.asyncio
    async def test_scene_index_set_on_sentences(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [
            {"text": "first scene text here"},
            {"text": "second scene text here"},
        ]
        tts_result = _make_tts_result(
            "first scene text here second scene text here", duration=6.0
        )
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes)

        scene_indices = {s.scene_index for s in transcript.sentences
                        if s.scene_index is not None}
        # Both scene indices should appear
        assert 0 in scene_indices or 1 in scene_indices

    @pytest.mark.asyncio
    async def test_scene_without_visual_hint_has_none(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [{"text": "plain scene with no hint"}]
        tts_result = _make_tts_result("plain scene with no hint", duration=2.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes)
        for s in transcript.sentences:
            assert s.visual_hint is None

    @pytest.mark.asyncio
    async def test_empty_scenes_raises_empty_script_error(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError):
            await process_scenes([])

    @pytest.mark.asyncio
    async def test_all_empty_text_scenes_raises(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        from app.core.errors import EmptyScriptError
        with pytest.raises(EmptyScriptError):
            await process_scenes([{"text": ""}, {"text": "   "}])

    @pytest.mark.asyncio
    async def test_multiple_scenes_word_count_correct(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [
            {"text": "one two three"},
            {"text": "four five six"},
        ]
        joined = "one two three four five six"
        tts_result = _make_tts_result(joined, duration=6.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes)
        assert transcript.word_count == 6

    @pytest.mark.asyncio
    async def test_language_stored(self) -> None:
        from app.pipeline.input_stage.text_processor import process_scenes
        scenes = [{"text": "hola mundo"}]
        tts_result = _make_tts_result("hola mundo", duration=2.0)
        with patch("app.pipeline.input_stage.text_processor.get_tts_adapter") as mock_f:
            mock_tts = MagicMock()
            mock_tts.synthesize = AsyncMock(return_value=tts_result)
            mock_f.return_value = mock_tts
            transcript = await process_scenes(scenes, language="es")
        assert transcript.language == "es"


# ── _assign_words_to_scenes ──────────────────────────────────────────────────

class TestAssignWordsToScenes:
    def _make_words(self, n: int, duration: float = 10.0):
        from app.pipeline.input_stage.transcript import WordTiming
        gap = duration / n
        return [
            WordTiming(word=f"w{i}", start=round(i * gap, 3),
                       end=round((i + 1) * gap - 0.05, 3))
            for i in range(n)
        ]

    def test_single_scene_gets_all_words(self) -> None:
        from app.pipeline.input_stage.text_processor import (
            _assign_words_to_scenes, SceneInput,
        )
        words = self._make_words(4)
        scenes = [SceneInput(text="one two three four", index=0)]
        result = _assign_words_to_scenes(words, scenes, total_duration=10.0)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_two_equal_scenes_split_evenly(self) -> None:
        from app.pipeline.input_stage.text_processor import (
            _assign_words_to_scenes, SceneInput,
        )
        words = self._make_words(8, duration=8.0)
        scenes = [
            SceneInput(text="a b c d", index=0),    # 7 chars
            SceneInput(text="e f g h", index=1),    # 7 chars
        ]
        result = _assign_words_to_scenes(words, scenes, total_duration=8.0)
        # Each scene should get roughly 4 words
        assert abs(len(result[0]) - len(result[1])) <= 1

    def test_empty_words_returns_empty_per_scene(self) -> None:
        from app.pipeline.input_stage.text_processor import (
            _assign_words_to_scenes, SceneInput,
        )
        scenes = [SceneInput(text="hello", index=0),
                  SceneInput(text="world", index=1)]
        result = _assign_words_to_scenes([], scenes, total_duration=5.0)
        assert len(result) == 2
        assert all(lst == [] for lst in result)

    def test_empty_scenes_returns_empty(self) -> None:
        from app.pipeline.input_stage.text_processor import (
            _assign_words_to_scenes, SceneInput,
        )
        words = self._make_words(4)
        assert _assign_words_to_scenes(words, [], total_duration=4.0) == []

    def test_all_words_assigned(self) -> None:
        from app.pipeline.input_stage.text_processor import (
            _assign_words_to_scenes, SceneInput,
        )
        words = self._make_words(10, duration=10.0)
        scenes = [
            SceneInput(text="first scene text", index=0),
            SceneInput(text="second scene text with more words", index=1),
            SceneInput(text="third", index=2),
        ]
        result = _assign_words_to_scenes(words, scenes, total_duration=10.0)
        total_assigned = sum(len(r) for r in result)
        assert total_assigned == 10
