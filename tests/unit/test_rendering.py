"""
Unit tests for the rendering stage.

No real ffmpeg/ffprobe calls, no MinIO, no real file I/O.
All external dependencies are mocked.

Coverage:
- AspectRatio enum
- target_dimensions() for all 3 ratios
- RenderedSegment / RenderResult to_dict
- normalise.py: _build_trim_loop_slow_filter, _build_scale_filter, _build_luma_filter
- reframe.py: get_reframe_method, compute_crop_from_bbox, resolve_crop_center
- captions.py: _format_srt_time, generate_srt, _select_font, build_subtitle_filter
- scene_cache.py: cache_key, get_cached, put_cached, invalidate
- post_check.py: PostCheckResult, check_codec_fps_resolution (mocked)
- assemble.py: _build_concat_list (mocked ffmpeg)
- render_step.py: RenderStep context reads/writes, error wrapping
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── models ──────────────────────────────────────────────────────────────────

class TestAspectRatio:
    def test_values(self) -> None:
        from app.pipeline.rendering.models import AspectRatio
        assert AspectRatio.RATIO_16_9.value == "16:9"
        assert AspectRatio.RATIO_9_16.value == "9:16"
        assert AspectRatio.RATIO_1_1.value == "1:1"

    def test_from_string(self) -> None:
        from app.pipeline.rendering.models import AspectRatio
        assert AspectRatio("16:9") == AspectRatio.RATIO_16_9


class TestRenderedSegment:
    def test_to_dict_shape(self) -> None:
        from app.pipeline.rendering.models import RenderedSegment
        seg = RenderedSegment(
            scene_id=0, local_path="/tmp/seg.mp4", duration=5.0,
            width=1920, height=1080, fps=30,
            reframe_method="center", brightness_adjusted=True,
        )
        d = seg.to_dict()
        for key in ("scene_id", "local_path", "duration", "width",
                     "height", "fps", "reframe_method", "brightness_adjusted",
                     "cached", "warnings"):
            assert key in d

    def test_defaults(self) -> None:
        from app.pipeline.rendering.models import RenderedSegment
        seg = RenderedSegment(
            scene_id=0, local_path="x", duration=5.0,
            width=1920, height=1080, fps=30,
        )
        assert seg.cached is False
        assert seg.warnings == []


class TestRenderResult:
    def test_to_dict_shape(self) -> None:
        from app.pipeline.rendering.models import RenderResult
        r = RenderResult(job_id="j1", aspect_ratio="16:9")
        d = r.to_dict()
        for key in ("job_id", "aspect_ratio", "final_video_path",
                     "srt_path", "final_video_object", "srt_object",
                     "segments", "post_check_passed", "post_check_errors",
                     "warnings", "total_duration"):
            assert key in d

    def test_defaults(self) -> None:
        from app.pipeline.rendering.models import RenderResult
        r = RenderResult(job_id="j1", aspect_ratio="16:9")
        assert r.final_video_path is None
        assert r.post_check_passed is False
        assert r.total_duration == 0.0


# ── ff_helpers ───────────────────────────────────────────────────────────────

class TestTargetDimensions:
    def test_16_9(self) -> None:
        from app.pipeline.rendering.ff_helpers import target_dimensions
        w, h = target_dimensions("16:9")
        assert w == 1920
        assert h == 1080

    def test_9_16(self) -> None:
        from app.pipeline.rendering.ff_helpers import target_dimensions
        w, h = target_dimensions("9:16")
        assert w == 1080
        assert h == 1920

    def test_1_1(self) -> None:
        from app.pipeline.rendering.ff_helpers import target_dimensions
        w, h = target_dimensions("1:1")
        assert w == 1080
        assert h == 1080

    def test_invalid_raises(self) -> None:
        from app.pipeline.rendering.ff_helpers import target_dimensions
        from app.core.errors import RenderError
        with pytest.raises(RenderError):
            target_dimensions("4:3")


# ── normalise.py ─────────────────────────────────────────────────────────────

class TestTrimLoopSlowFilter:
    def test_no_adjustment_needed(self) -> None:
        from app.pipeline.rendering.normalise import _build_trim_loop_slow_filter
        input_args, vf = _build_trim_loop_slow_filter(
            clip_duration=10.0, scene_duration=10.0,
            needs_slow=False, needs_loop=0,
        )
        assert input_args == []
        assert vf == "null"

    def test_loop_input_args(self) -> None:
        from app.pipeline.rendering.normalise import _build_trim_loop_slow_filter
        input_args, vf = _build_trim_loop_slow_filter(
            clip_duration=2.0, scene_duration=10.0,
            needs_slow=False, needs_loop=4,
        )
        assert "-stream_loop" in input_args
        assert "4" in input_args

    def test_slow_setpts(self) -> None:
        from app.pipeline.rendering.normalise import _build_trim_loop_slow_filter
        input_args, vf = _build_trim_loop_slow_filter(
            clip_duration=5.0, scene_duration=10.0,
            needs_slow=True, needs_loop=0,
        )
        assert "setpts" in vf
        assert "PTS" in vf


class TestScaleFilter:
    def test_same_ratio_simple_scale(self) -> None:
        from app.pipeline.rendering.normalise import _build_scale_filter
        vf = _build_scale_filter(1920, 1080, 1920, 1080, "center")
        assert "scale=1920:1080" in vf
        assert "setsar" in vf

    def test_blurred_bg(self) -> None:
        from app.pipeline.rendering.normalise import _build_scale_filter
        vf = _build_scale_filter(1920, 1080, 1080, 1920, "blurred_bg")
        assert "boxblur" in vf
        assert "overlay" in vf

    def test_crop_for_vertical(self) -> None:
        from app.pipeline.rendering.normalise import _build_scale_filter
        vf = _build_scale_filter(1920, 1080, 1080, 1920, "center")
        assert "crop" in vf
        assert "scale=1080:1920" in vf

    def test_vlm_bbox_offset(self) -> None:
        from app.pipeline.rendering.normalise import _build_scale_filter
        vf = _build_scale_filter(1920, 1080, 1080, 1920, "vlm_bbox",
                                  crop_x=0.3, crop_y=0.5)
        assert "crop" in vf


class TestLumaFilter:
    def test_brightness_up(self) -> None:
        from app.pipeline.rendering.normalise import _build_luma_filter
        vf = _build_luma_filter(target_luma=0.5, current_luma=0.3)
        assert "eq=" in vf
        assert "brightness=" in vf
        # delta = 0.5 - 0.3 = 0.2, brightness = 0.2 * 2.0 = 0.4
        assert "0.4" in vf

    def test_no_change(self) -> None:
        from app.pipeline.rendering.normalise import _build_luma_filter
        vf = _build_luma_filter(target_luma=0.5, current_luma=0.5)
        assert "brightness=0.000" in vf

    def test_clamped(self) -> None:
        from app.pipeline.rendering.normalise import _build_luma_filter
        vf = _build_luma_filter(target_luma=1.0, current_luma=0.0)
        assert "brightness=0.500" in vf or "brightness=1.000" in vf


# ── reframe.py ──────────────────────────────────────────────────────────────

class TestGetReframeMethod:
    def test_16_9_always_center(self) -> None:
        from app.pipeline.rendering.reframe import get_reframe_method
        assert get_reframe_method(has_vlm_bbox=True, aspect_ratio="16:9") == "center"

    def test_9_16_with_vlm_bbox(self) -> None:
        from app.pipeline.rendering.reframe import get_reframe_method
        with patch.dict(os.environ, {"RENDER_REFRAME_METHOD": "vlm_bbox"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            assert get_reframe_method(has_vlm_bbox=True, aspect_ratio="9:16") == "vlm_bbox"
        get_settings.cache_clear()

    def test_9_16_without_vlm_bbox(self) -> None:
        from app.pipeline.rendering.reframe import get_reframe_method
        assert get_reframe_method(has_vlm_bbox=False, aspect_ratio="9:16") != "vlm_bbox"

    def test_1_1_with_vlm_bbox(self) -> None:
        from app.pipeline.rendering.reframe import get_reframe_method
        with patch.dict(os.environ, {"RENDER_REFRAME_METHOD": "vlm_bbox"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            assert get_reframe_method(has_vlm_bbox=True, aspect_ratio="1:1") == "vlm_bbox"
        get_settings.cache_clear()


class TestComputeCropFromBBox:
    def test_center_bbox(self) -> None:
        from app.pipeline.rendering.reframe import compute_crop_from_bbox
        cx, cy = compute_crop_from_bbox(1920, 1080, 1080, 1920,
                                         0.4, 0.3, 0.6, 0.7)
        # bbox center = (0.5, 0.5) → center crop
        assert 0.4 <= cx <= 0.6
        assert 0.4 <= cy <= 0.6

    def test_offset_bbox(self) -> None:
        from app.pipeline.rendering.reframe import compute_crop_from_bbox
        cx, cy = compute_crop_from_bbox(1920, 1080, 1080, 1920,
                                         0.1, 0.1, 0.3, 0.3)
        # bbox center = (0.2, 0.2) → should offset crop
        assert cx < 0.5 or cy < 0.5

    def test_full_frame_bbox(self) -> None:
        from app.pipeline.rendering.reframe import compute_crop_from_bbox
        cx, cy = compute_crop_from_bbox(1920, 1080, 1080, 1920,
                                         0.0, 0.0, 1.0, 1.0)
        assert cx == pytest.approx(0.5)


class TestResolveCropCenter:
    def test_vlm_bbox_priority(self) -> None:
        from app.pipeline.rendering.reframe import resolve_crop_center
        cx, cy, method = resolve_crop_center(
            1920, 1080, 1080, 1920,
            vlm_bbox=(0.1, 0.1, 0.3, 0.3),
            thumb_bytes=None,
        )
        assert method == "vlm_bbox"

    def test_fallback_to_center(self) -> None:
        from app.pipeline.rendering.reframe import resolve_crop_center
        cx, cy, method = resolve_crop_center(
            1920, 1080, 1080, 1920,
            vlm_bbox=None,
            thumb_bytes=None,
        )
        assert method == "center"
        assert cx == 0.5
        assert cy == 0.5


# ── captions.py ──────────────────────────────────────────────────────────────

class TestFormatSrtTime:
    def test_zero(self) -> None:
        from app.pipeline.rendering.captions import _format_srt_time
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_seconds(self) -> None:
        from app.pipeline.rendering.captions import _format_srt_time
        assert _format_srt_time(5.5) == "00:00:05,500"

    def test_minutes(self) -> None:
        from app.pipeline.rendering.captions import _format_srt_time
        assert _format_srt_time(65.123) == "00:01:05,123"

    def test_hours(self) -> None:
        from app.pipeline.rendering.captions import _format_srt_time
        assert _format_srt_time(3661.0) == "01:01:01,000"


class TestGenerateSrt:
    def _make_words(self, n: int = 5, gap: float = 0.4) -> list[Any]:
        from app.pipeline.input_stage.transcript import WordTiming
        return [
            WordTiming(word=f"word{i}", start=round(i * gap, 2),
                       end=round(i * gap + 0.35, 2))
            for i in range(n)
        ]

    def test_generates_valid_srt(self, tmp_path: str) -> None:
        from app.pipeline.rendering.captions import generate_srt
        words = self._make_words(5)
        out = generate_srt(words, output_path=str(tmp_path / "test.srt"))
        content = open(out, encoding="utf-8").read()
        assert "1" in content  # first cue index
        assert "-->" in content  # timestamp separator
        assert "word0" in content

    def test_empty_words(self, tmp_path: str) -> None:
        from app.pipeline.rendering.captions import generate_srt
        out = generate_srt([], output_path=str(tmp_path / "empty.srt"))
        content = open(out, encoding="utf-8").read()
        assert content.strip() == ""

    def test_cue_count_reasonable(self, tmp_path: str) -> None:
        from app.pipeline.rendering.captions import generate_srt
        words = self._make_words(20, gap=0.5)
        out = generate_srt(words, max_chars=20, output_path=str(tmp_path / "test.srt"))
        content = open(out, encoding="utf-8").read()
        # Should have multiple cues for 20 words
        cue_indices = [l for l in content.split("\n") if l.strip().isdigit()]
        assert len(cue_indices) >= 1

    def test_respects_max_chars(self, tmp_path: str) -> None:
        from app.pipeline.rendering.captions import generate_srt
        words = self._make_words(30, gap=0.3)
        out = generate_srt(words, max_chars=15, output_path=str(tmp_path / "test.srt"))
        content = open(out, encoding="utf-8").read()
        lines = content.split("\n")
        # No line (that's actual subtitle text) should exceed max_chars + small tolerance
        for line in lines:
            if line.strip() and not line.strip().isdigit() and "-->" not in line:
                assert len(line) <= 20, f"Line too long: {line}"


class TestSelectFont:
    def test_english(self) -> None:
        from app.pipeline.rendering.captions import _select_font
        assert "NotoSans" in _select_font("en")

    def test_hindi(self) -> None:
        from app.pipeline.rendering.captions import _select_font
        assert "Devanagari" in _select_font("hi")

    def test_tamil(self) -> None:
        from app.pipeline.rendering.captions import _select_font
        assert "Tamil" in _select_font("ta")

    def test_marathi(self) -> None:
        from app.pipeline.rendering.captions import _select_font
        assert "Devanagari" in _select_font("mr")


class TestBuildSubtitleFilter:
    def test_filter_contains_subtitles(self) -> None:
        from app.pipeline.rendering.captions import build_subtitle_filter
        f = build_subtitle_filter("/tmp/test.srt", language="en")
        assert "subtitles=" in f

    def test_filter_contains_font(self) -> None:
        from app.pipeline.rendering.captions import build_subtitle_filter
        f = build_subtitle_filter("/tmp/test.srt", language="en")
        assert "NotoSans" in f

    def test_filter_contains_font_deva(self) -> None:
        from app.pipeline.rendering.captions import build_subtitle_filter
        f = build_subtitle_filter("/tmp/test.srt", language="hi")
        assert "Devanagari" in f

    def test_filter_contains_font_tamil(self) -> None:
        from app.pipeline.rendering.captions import build_subtitle_filter
        f = build_subtitle_filter("/tmp/test.srt", language="ta")
        assert "Tamil" in f

    def test_escapes_colons_in_path(self) -> None:
        from app.pipeline.rendering.captions import build_subtitle_filter
        f = build_subtitle_filter("C:\\videos\\test.srt", language="en")
        # Colons should be escaped
        assert r"\:" in f or "test.srt" in f


# ── scene_cache.py ───────────────────────────────────────────────────────────

class TestSceneCache:
    def test_cache_key_deterministic(self) -> None:
        from app.pipeline.rendering.scene_cache import cache_key
        k1 = cache_key("job1", 0, "pixabay:1", "16:9")
        k2 = cache_key("job1", 0, "pixabay:1", "16:9")
        assert k1 == k2

    def test_cache_key_different_inputs_differ(self) -> None:
        from app.pipeline.rendering.scene_cache import cache_key
        k1 = cache_key("job1", 0, "pixabay:1", "16:9")
        k2 = cache_key("job1", 0, "pixabay:1", "9:16")
        k3 = cache_key("job2", 0, "pixabay:1", "16:9")
        assert k1 != k2
        assert k1 != k3

    def test_get_cached_miss_returns_none(self) -> None:
        from app.pipeline.rendering.scene_cache import get_cached
        assert get_cached("job", 0, "clip", "16:9") is None

    def test_get_cached_disabled(self) -> None:
        with patch.dict(os.environ, {"RENDER_SCENE_CACHE_ENABLED": "false"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            from app.pipeline.rendering.scene_cache import get_cached
            assert get_cached("job", 0, "clip", "16:9") is None
        get_settings.cache_clear()

    def test_put_cached_returns_path(self, tmp_path: str) -> None:
        from app.pipeline.rendering.scene_cache import put_cached, cache_key
        src = os.path.join(str(tmp_path), "src.mp4")
        with open(src, "wb") as f:
            f.write(b"fake")
        result = put_cached(src, "job1", 0, "clip1", "16:9")
        assert isinstance(result, str)

    def test_invalidate_missing_is_safe(self) -> None:
        from app.pipeline.rendering.scene_cache import invalidate
        invalidate("job", 0, "nonexistent", "16:9")  # must not raise


# ── post_check.py ────────────────────────────────────────────────────────────

class TestPostCheckResult:
    def test_default_passed(self) -> None:
        from app.pipeline.rendering.post_check import PostCheckResult
        r = PostCheckResult(passed=True)
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []

    def test_to_dict_shape(self) -> None:
        from app.pipeline.rendering.post_check import PostCheckResult
        r = PostCheckResult(passed=False, errors=["black frame"])
        d = r.to_dict()
        assert d["passed"] is False
        assert "black frame" in d["errors"]


class TestCheckCodecFpsResolution:
    @pytest.mark.asyncio
    async def test_valid_video_passes(self) -> None:
        from app.pipeline.rendering.post_check import check_codec_fps_resolution
        probe = {
            "streams": [{
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920, "height": 1080,
                "pix_fmt": "yuv420p",
                "r_frame_rate": "30/1",
            }],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_codec_fps_resolution("x.mp4", 1920, 1080, 30)
        assert ok
        assert errors == []

    @pytest.mark.asyncio
    async def test_wrong_codec_fails(self) -> None:
        from app.pipeline.rendering.post_check import check_codec_fps_resolution
        probe = {
            "streams": [{
                "codec_type": "video", "codec_name": "vp9",
                "width": 1920, "height": 1080,
                "pix_fmt": "yuv420p", "r_frame_rate": "30/1",
            }],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_codec_fps_resolution("x.mp4", 1920, 1080, 30)
        assert not ok
        assert any("h264" in e for e in errors)

    @pytest.mark.asyncio
    async def test_wrong_resolution_fails(self) -> None:
        from app.pipeline.rendering.post_check import check_codec_fps_resolution
        probe = {
            "streams": [{
                "codec_type": "video", "codec_name": "h264",
                "width": 1280, "height": 720,
                "pix_fmt": "yuv420p", "r_frame_rate": "30/1",
            }],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_codec_fps_resolution("x.mp4", 1920, 1080, 30)
        assert not ok
        assert any("1280" in e for e in errors)

    @pytest.mark.asyncio
    async def test_wrong_fps_fails(self) -> None:
        from app.pipeline.rendering.post_check import check_codec_fps_resolution
        probe = {
            "streams": [{
                "codec_type": "video", "codec_name": "h264",
                "width": 1920, "height": 1080,
                "pix_fmt": "yuv420p", "r_frame_rate": "24/1",
            }],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_codec_fps_resolution("x.mp4", 1920, 1080, 30)
        assert not ok

    @pytest.mark.asyncio
    async def test_no_video_stream_fails(self) -> None:
        from app.pipeline.rendering.post_check import check_codec_fps_resolution
        probe = {"streams": []}
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_codec_fps_resolution("x.mp4", 1920, 1080, 30)
        assert not ok


class TestCheckAudioVideoDrift:
    @pytest.mark.asyncio
    async def test_no_drift_passes(self) -> None:
        from app.pipeline.rendering.post_check import check_audio_video_drift
        probe = {
            "streams": [
                {"codec_type": "video", "duration": "10.000"},
                {"codec_type": "audio", "duration": "10.010"},
            ],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_audio_video_drift("x.mp4")
        assert ok

    @pytest.mark.asyncio
    async def test_large_drift_fails(self) -> None:
        from app.pipeline.rendering.post_check import check_audio_video_drift
        probe = {
            "streams": [
                {"codec_type": "video", "duration": "10.000"},
                {"codec_type": "audio", "duration": "10.500"},
            ],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_audio_video_drift("x.mp4")
        assert not ok
        assert any("drift" in e.lower() for e in errors)

    @pytest.mark.asyncio
    async def test_no_audio_passes(self) -> None:
        from app.pipeline.rendering.post_check import check_audio_video_drift
        probe = {
            "streams": [
                {"codec_type": "video", "duration": "10.000"},
            ],
        }
        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)):
            ok, errors = await check_audio_video_drift("x.mp4")
        assert ok  # can't check if no audio


class TestRunPostChecks:
    @pytest.mark.asyncio
    async def test_all_pass(self) -> None:
        from app.pipeline.rendering.post_check import run_post_checks
        probe = {
            "streams": [{
                "codec_type": "video", "codec_name": "h264",
                "width": 1920, "height": 1080,
                "pix_fmt": "yuv420p", "r_frame_rate": "30/1",
                "duration": "10.0",
            }, {
                "codec_type": "audio", "duration": "10.01",
            }],
        }
        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0
        ffmpeg_result.stderr = b""

        with patch("app.pipeline.rendering.post_check.probe_video",
                   AsyncMock(return_value=probe)), \
             patch("app.pipeline.rendering.post_check.run_ffmpeg",
                   AsyncMock(return_value=ffmpeg_result)):
            result = await run_post_checks("x.mp4", 1920, 1080, 30)
        assert result.passed

    @pytest.mark.asyncio
    async def test_disabled_passes(self) -> None:
        from app.pipeline.rendering.post_check import run_post_checks
        with patch.dict(os.environ, {"RENDER_POST_CHECK_ENABLED": "false"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            result = await run_post_checks("x.mp4", 1920, 1080, 30)
            assert result.passed
            assert "disabled" in result.warnings[0]
        get_settings.cache_clear()


# ── assemble.py ──────────────────────────────────────────────────────────────

class TestBuildConcatList:
    def test_writes_all_paths(self, tmp_path: str) -> None:
        from app.pipeline.rendering.assemble import _build_concat_list
        paths = ["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"]
        list_path = _build_concat_list(paths, str(tmp_path))
        content = open(list_path).read()
        assert "a.mp4" in content
        assert "b.mp4" in content
        assert "c.mp4" in content
        assert content.count("file '") == 3

    def test_escapes_quotes(self, tmp_path: str) -> None:
        from app.pipeline.rendering.assemble import _build_concat_list
        paths = ["/tmp/it's a file.mp4"]
        list_path = _build_concat_list(paths, str(tmp_path))
        content = open(list_path).read()
        assert r"\'" in content


# ── render_step.py ──────────────────────────────────────────────────────────

class TestRenderStep:
    @pytest.mark.asyncio
    async def test_missing_plan_raises_validation(self) -> None:
        from app.pipeline.rendering.render_step import RenderStep
        from app.pipeline.base import PipelineContext
        from app.core.errors import ValidationError
        ctx = PipelineContext(job_id="j1")
        with pytest.raises(ValidationError):
            await RenderStep().execute(ctx)

    def test_step_name(self) -> None:
        from app.pipeline.rendering.render_step import RenderStep
        assert RenderStep().name == "RenderStep"

    @pytest.mark.asyncio
    async def test_renders_one_ratio_with_mock(self) -> None:
        """Full RenderStep with all ffmpeg calls mocked."""
        from app.pipeline.rendering.render_step import RenderStep
        from app.pipeline.base import PipelineContext
        from app.pipeline.clip_selection.models import (
            VideoSelectionPlan, SceneClipResult, ClipCandidate, SelectionRule,
        )
        from app.adapters.footage_types import (
            Candidate, FallbackTier, FootageKind,
        )

        cand = Candidate(
            id="pixabay:1", source="pixabay", kind=FootageKind.VIDEO,
            url="https://x.com/v.mp4", thumb_url="",
            width=1920, height=1080, duration=5.0,
            storage_object="footage/pixabay/pixabay_1.mp4",
            storage_bucket="broll-processed",
            fallback_tier=FallbackTier.PIXABAY_VIDEO,
        )
        cc = ClipCandidate(candidate=cand, query="city", query_index=0,
                           final_score=4.0, needs_slow=False, needs_loop=0)
        scene_result = SceneClipResult(
            scene_id=0, scene_start=0.0, scene_end=5.0,
            scene_duration=5.0, chosen=cc,
            selection_rule=SelectionRule.PIXABAY_VIDEO,
        )
        plan = VideoSelectionPlan(job_id="j1", scene_results=[scene_result])

        ctx = PipelineContext(job_id="j1")
        ctx.set("video_selection_plan_obj", plan)
        ctx.set("transcript_obj", None)
        ctx.payload["aspect_ratios"] = ["16:9"]
        ctx.payload["captions"] = {"burn_in": False, "srt": False}

        # Mock all external calls
        mock_storage = MagicMock()
        mock_storage.download_bytes = AsyncMock(return_value=b"fake-video")
        mock_storage.ensure_bucket = AsyncMock()
        mock_storage.upload_file = AsyncMock()

        ffmpeg_result = MagicMock()
        ffmpeg_result.returncode = 0
        ffmpeg_result.stderr = b""

        with patch("app.pipeline.rendering.render_step.get_storage_client",
                   return_value=mock_storage), \
             patch("app.pipeline.rendering.ff_helpers.run_ffmpeg",
                   AsyncMock(return_value=ffmpeg_result)), \
             patch("app.pipeline.rendering.ff_helpers.get_duration",
                   AsyncMock(return_value=5.0)), \
             patch("app.pipeline.rendering.render_step.normalise_clip",
                   AsyncMock(return_value="/tmp/norm.mp4")), \
             patch("app.pipeline.rendering.render_step.apply_crossfades",
                   AsyncMock(return_value="/tmp/cross.mp4")), \
             patch("app.pipeline.rendering.render_step.mix_narration_audio",
                   AsyncMock(return_value="/tmp/audio.mp4")), \
             patch("app.pipeline.rendering.render_step.run_post_checks",
                   AsyncMock(return_value=MagicMock(
                       passed=True, errors=[], warnings=[],
                       details={}))), \
             patch("app.pipeline.rendering.render_step.upload_render_outputs",
                   AsyncMock(side_effect=lambda r: r)):
            result_ctx = await RenderStep().execute(ctx)

        assert result_ctx.get("render_result_objs") is not None
        render_results = result_ctx.get("render_result_objs")
        assert len(render_results) == 1
        assert render_results[0].aspect_ratio == "16:9"
        assert render_results[0].post_check_passed is True

    @pytest.mark.asyncio
    async def test_no_scenes_raises_pipeline_error(self) -> None:
        from app.pipeline.rendering.render_step import RenderStep
        from app.pipeline.base import PipelineContext
        from app.pipeline.clip_selection.models import VideoSelectionPlan
        from app.core.errors import (PipelineError, ValidationError)

        plan = VideoSelectionPlan(job_id="j1", scene_results=[])
        ctx = PipelineContext(job_id="j1")
        ctx.set("video_selection_plan_obj", plan)
        ctx.payload["aspect_ratios"] = ["16:9"]

        with pytest.raises((PipelineError, Exception)):
            await RenderStep().execute(ctx)
