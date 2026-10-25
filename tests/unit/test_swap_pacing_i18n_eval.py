"""
Tests for:
  1. Swap API helpers (_get_old_clip_id, _get_scene_duration, _get_original_render_time)
  2. Pacing guard (check_pacing)
  3. Global colour pass (apply_global_colour_pass — mocked ffmpeg)
  4. Language utilities (detect_script, get_caption_font, build_stt_language_hint,
     ensure_english_queries)
  5. VLM eval (mock mode, Spearman, MAE, ±1 metrics, human_ratings.csv loading)
  6. Benchmark mock mode (to_markdown roundtrip)
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_segment(scene_id: int = 0, duration: float = 5.0,
                   reframe: str = "center",
                   brightness: bool = False,
                   path: str = "/tmp/seg.mp4") -> Any:
    from app.pipeline.rendering.models import RenderedSegment
    return RenderedSegment(
        scene_id=scene_id, local_path=path,
        duration=duration, width=1920, height=1080, fps=30,
        reframe_method=reframe, brightness_adjusted=brightness,
    )


def _make_timeline(n_scenes: int = 3) -> dict:
    return {
        "job_id": "j1",
        "total_duration": n_scenes * 5.0,
        "scenes": [
            {
                "index": i, "text": f"scene {i}",
                "start": i * 5.0, "end": (i + 1) * 5.0,
                "clip": {
                    "id": f"pixabay:{1000 + i}",
                    "source": "pixabay", "url": "https://cdn.x.com/v.mp4",
                    "page_url": "https://pixabay.com/v/1/",
                    "license": "Pixabay Content License",
                    "commercial_use": True,
                    "attribution": "Video by user on Pixabay",
                    "is_ai_generated": False,
                    "duration_used": 5.0,
                },
                "reason_chosen": "text=3.5, vlm=4.0",
                "relevance_score": 3.8,
                "vlm_score": 4.0,
                "source_rule_fired": "pixabay_video_score_gte_threshold",
                "alternatives_rejected": [],
                "warnings": [],
            }
            for i in range(n_scenes)
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. SWAP API HELPERS
# ═══════════════════════════════════════════════════════════════════════════

class TestSwapHelpers:
    def test_get_old_clip_id(self) -> None:
        from app.api.swap import _get_old_clip_id
        result = {"timeline": _make_timeline(3)}
        assert _get_old_clip_id(result, scene_idx=1) == "pixabay:1001"

    def test_get_old_clip_id_missing_scene(self) -> None:
        from app.api.swap import _get_old_clip_id
        result = {"timeline": _make_timeline(3)}
        assert _get_old_clip_id(result, scene_idx=99) is None

    def test_get_old_clip_id_no_clip(self) -> None:
        from app.api.swap import _get_old_clip_id
        result = {"timeline": {"scenes": [
            {"index": 0, "clip": None}
        ]}}
        assert _get_old_clip_id(result, scene_idx=0) is None

    def test_get_scene_duration(self) -> None:
        from app.api.swap import _get_scene_duration
        result = {"timeline": _make_timeline(3)}
        assert _get_scene_duration(result, scene_idx=0) == pytest.approx(5.0)

    def test_get_scene_duration_missing(self) -> None:
        from app.api.swap import _get_scene_duration
        assert _get_scene_duration({}, scene_idx=5) == pytest.approx(5.0)

    def test_get_original_render_time(self) -> None:
        from app.api.swap import _get_original_render_time
        result = {"manifest": {"stage_timings": {"render": 42.5}}}
        assert _get_original_render_time(result) == pytest.approx(42.5)

    def test_get_original_render_time_missing(self) -> None:
        from app.api.swap import _get_original_render_time
        assert _get_original_render_time({}) is None


# ═══════════════════════════════════════════════════════════════════════════
# 2. PACING GUARD
# ═══════════════════════════════════════════════════════════════════════════

class TestPacingGuard:
    def test_good_variety_passes(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        segments = [
            _make_segment(i, reframe=r)
            for i, r in enumerate(["center", "vlm_bbox", "saliency", "center"])
        ]
        report = check_pacing(segments)
        assert len(report.warnings) == 0

    def test_too_many_consecutive_flags(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        # 5 consecutive "center" segments — above max of 3
        segments = [_make_segment(i, reframe="center") for i in range(5)]
        with patch.dict(os.environ, {"PACING_MAX_CONSECUTIVE_SAME_SOURCE": "3"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            report = check_pacing(segments)
        get_settings.cache_clear()
        assert len(report.warnings) > 0
        assert any("consecutive" in w.lower() for w in report.warnings)

    def test_empty_segments_passes(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        report = check_pacing([])
        assert report.passed

    def test_single_segment_passes(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        report = check_pacing([_make_segment(0)])
        assert report.passed

    def test_pacing_report_has_stats(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        segments = [_make_segment(i) for i in range(3)]
        report = check_pacing(segments)
        assert "segment_count" in report.stats
        assert report.stats["segment_count"] == 3

    def test_duration_variance_recorded(self) -> None:
        from app.pipeline.rendering.pacing import check_pacing
        segments = [
            _make_segment(0, duration=3.0),
            _make_segment(1, duration=8.0),
            _make_segment(2, duration=5.5),
            _make_segment(3, duration=4.0),
        ]
        report = check_pacing(segments)
        assert "duration_cv" in report.stats


# ═══════════════════════════════════════════════════════════════════════════
# 3. COLOUR PASS
# ═══════════════════════════════════════════════════════════════════════════

class TestColourPass:
    @pytest.mark.asyncio
    async def test_disabled_returns_unchanged(self, tmp_path) -> None:
        from app.pipeline.rendering.pacing import apply_global_colour_pass
        with patch.dict(os.environ, {"COLOUR_PASS_ENABLED": "false"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            segs = [_make_segment(0)]
            result, warnings = await apply_global_colour_pass(segs, str(tmp_path))
            assert result == segs
            assert warnings == []
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_luma_within_tolerance_skipped(self, tmp_path) -> None:
        from app.pipeline.rendering.pacing import apply_global_colour_pass
        segs = [_make_segment(0)]
        # If luma measurement returns target value, no correction needed
        with patch("app.pipeline.rendering.pacing.measure_luma",
                   AsyncMock(return_value=0.5)):  # exactly at target
            result, warnings = await apply_global_colour_pass(segs, str(tmp_path))
        assert not result[0].brightness_adjusted

    @pytest.mark.asyncio
    async def test_luma_unavailable_skips_gracefully(self, tmp_path) -> None:
        from app.pipeline.rendering.pacing import apply_global_colour_pass
        segs = [_make_segment(0)]
        with patch("app.pipeline.rendering.pacing.measure_luma",
                   AsyncMock(return_value=None)):
            result, warnings = await apply_global_colour_pass(segs, str(tmp_path))
        assert len(result) == 1
        assert result[0] is segs[0]


# ═══════════════════════════════════════════════════════════════════════════
# 4. LANGUAGE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class TestLanguageUtils:
    def test_detect_devanagari(self) -> None:
        from app.pipeline.input_stage.language import detect_script
        assert detect_script("नमस्ते दुनिया") == "devanagari"

    def test_detect_tamil(self) -> None:
        from app.pipeline.input_stage.language import detect_script
        assert detect_script("வணக்கம் உலகம்") == "tamil"

    def test_detect_latin_returns_none(self) -> None:
        from app.pipeline.input_stage.language import detect_script
        assert detect_script("Hello world") is None

    def test_detect_arabic(self) -> None:
        from app.pipeline.input_stage.language import detect_script
        assert detect_script("مرحبا") == "arabic"

    def test_get_caption_font_english(self) -> None:
        from app.pipeline.input_stage.language import get_caption_font
        font = get_caption_font("en")
        assert "NotoSans" in font
        assert "Devanagari" not in font
        assert "Tamil" not in font

    def test_get_caption_font_hindi(self) -> None:
        from app.pipeline.input_stage.language import get_caption_font
        font = get_caption_font("hi")
        assert "Devanagari" in font

    def test_get_caption_font_marathi(self) -> None:
        from app.pipeline.input_stage.language import get_caption_font
        font = get_caption_font("mr")
        assert "Devanagari" in font

    def test_get_caption_font_tamil(self) -> None:
        from app.pipeline.input_stage.language import get_caption_font
        font = get_caption_font("ta")
        assert "Tamil" in font

    def test_build_language_hint_explicit(self) -> None:
        from app.pipeline.input_stage.language import build_stt_language_hint
        assert build_stt_language_hint("fr") == "fr"

    def test_build_language_hint_from_devanagari_script(self) -> None:
        from app.pipeline.input_stage.language import build_stt_language_hint
        hint = build_stt_language_hint(None, script_text="नमस्ते")
        assert hint == "hi"

    def test_build_language_hint_from_tamil_script(self) -> None:
        from app.pipeline.input_stage.language import build_stt_language_hint
        hint = build_stt_language_hint(None, script_text="வணக்கம்")
        assert hint == "ta"

    def test_build_language_hint_latin_returns_none(self) -> None:
        from app.pipeline.input_stage.language import build_stt_language_hint
        assert build_stt_language_hint(None, script_text="Hello world") is None

    def test_ensure_english_queries_pass_through(self) -> None:
        from app.pipeline.input_stage.language import ensure_english_queries
        queries = ["city skyline", "people walking", "abstract light"]
        result, corrected = ensure_english_queries(queries, "en")
        assert result == queries
        assert not corrected

    def test_ensure_english_queries_replaces_hindi(self) -> None:
        from app.pipeline.input_stage.language import ensure_english_queries
        queries = ["शहर की सड़क", "city streets"]
        result, corrected = ensure_english_queries(queries, "hi")
        assert corrected
        # The Hindi query should be replaced; the English one stays
        assert "city streets" in result

    def test_ensure_english_queries_all_replaced_adds_fallback(self) -> None:
        from app.pipeline.input_stage.language import ensure_english_queries
        queries = ["वर्षा वन", "தமிழ் படம்"]
        result, corrected = ensure_english_queries(queries, "hi")
        assert corrected
        assert len(result) >= 1
        assert all(detect_is_latin(q) for q in result)


def detect_is_latin(text: str) -> bool:
    """Helper: True if text has no non-ASCII chars (is effectively Latin)."""
    return text.encode("ascii", errors="ignore").decode() == text or not text


# ═══════════════════════════════════════════════════════════════════════════
# 5. VLM EVAL
# ═══════════════════════════════════════════════════════════════════════════

class TestVLMEval:
    @pytest.mark.asyncio
    async def test_mock_eval_returns_scores_for_all_scenes(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(5)
        report = await evaluate_timeline(
            timeline, model="mock", api_key="mock",
            n_frames=3, use_mock=True,
        )
        assert len(report.scene_results) == 5
        for sr in report.scene_results:
            assert 1.0 <= sr.eval_score <= 5.0

    @pytest.mark.asyncio
    async def test_mock_eval_deterministic(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(3)
        r1 = await evaluate_timeline(timeline, model="m", api_key="k",
                                      n_frames=3, use_mock=True)
        r2 = await evaluate_timeline(timeline, model="m", api_key="k",
                                      n_frames=3, use_mock=True)
        scores1 = [r.eval_score for r in r1.scene_results]
        scores2 = [r.eval_score for r in r2.scene_results]
        assert scores1 == scores2

    @pytest.mark.asyncio
    async def test_spearman_computed_when_enough_human_ratings(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(10)
        human = {i: float((i % 5) + 1) for i in range(10)}
        report = await evaluate_timeline(
            timeline, model="m", api_key="k",
            human_ratings=human, n_frames=2, use_mock=True,
        )
        assert report.spearman_rho is not None
        assert -1.0 <= report.spearman_rho <= 1.0

    @pytest.mark.asyncio
    async def test_mae_in_valid_range(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(10)
        human = {i: 3.0 for i in range(10)}
        report = await evaluate_timeline(
            timeline, model="m", api_key="k",
            human_ratings=human, n_frames=2, use_mock=True,
        )
        assert report.mae is not None
        assert 0.0 <= report.mae <= 4.0

    @pytest.mark.asyncio
    async def test_pct_within_1_is_percentage(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(10)
        human = {i: 3.0 for i in range(10)}
        report = await evaluate_timeline(
            timeline, model="m", api_key="k",
            human_ratings=human, n_frames=2, use_mock=True,
        )
        assert report.pct_within_1 is not None
        assert 0.0 <= report.pct_within_1 <= 100.0

    def test_spearman_perfect_correlation(self) -> None:
        from eval.vlm_eval import compute_spearman
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        rho = compute_spearman(x, x)
        assert rho == pytest.approx(1.0)

    def test_spearman_inverse_correlation(self) -> None:
        from eval.vlm_eval import compute_spearman
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        rho = compute_spearman(x, y)
        assert rho == pytest.approx(-1.0)

    def test_load_human_ratings_csv(self, tmp_path) -> None:
        from eval.vlm_eval import load_human_ratings
        csv_path = str(tmp_path / "ratings.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene_index", "human_score", "notes"])
            writer.writeheader()
            writer.writerow({"scene_index": "0", "human_score": "4.0", "notes": ""})
            writer.writerow({"scene_index": "1", "human_score": "3", "notes": ""})
        ratings = load_human_ratings(csv_path)
        assert ratings[0] == pytest.approx(4.0)
        assert ratings[1] == pytest.approx(3.0)

    def test_load_human_ratings_missing_file_returns_empty(self) -> None:
        from eval.vlm_eval import load_human_ratings
        assert load_human_ratings("/does/not/exist.csv") == {}

    @pytest.mark.asyncio
    async def test_markdown_table_contains_required_headers(self) -> None:
        from eval.vlm_eval import evaluate_timeline
        timeline = _make_timeline(3)
        report = await evaluate_timeline(
            timeline, model="m", api_key="k",
            n_frames=2, use_mock=True,
        )
        md = report.to_markdown_table()
        assert "VLM Evaluation Results" in md
        assert "Eval Score" in md or "eval" in md.lower()


# ═══════════════════════════════════════════════════════════════════════════
# 6. BENCHMARK MOCK MODE
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchmarkMock:
    def test_mock_benchmark_runs(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        assert report.cold_start is not None
        assert report.warm_start is not None
        assert report.cold_start.turnaround_seconds > report.warm_start.turnaround_seconds

    def test_speedup_is_positive(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        c = report.cold_start.turnaround_seconds
        w = report.warm_start.turnaround_seconds
        assert c > w

    def test_concurrency_results_present(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        assert len(report.concurrency_results) > 0
        assert report.max_clean_concurrency > 0

    def test_to_markdown_contains_headers(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        md = report.to_markdown()
        assert "Turnaround" in md
        assert "Cost" in md or "cost" in md.lower()
        assert "Concurrency" in md

    def test_to_markdown_cold_warm_both_present(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        md = report.to_markdown()
        assert "Cold" in md
        assert "Warm" in md

    def test_cost_per_minute_positive(self) -> None:
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        assert report.cost_per_minute is not None
        assert report.cost_per_minute > 0

    def test_raw_json_serialisable(self) -> None:
        import json
        from benchmarks.bench import _run_mock_benchmark
        report = _run_mock_benchmark()
        raw = {
            "cold_turnaround_s": report.cold_start.turnaround_seconds,
            "warm_turnaround_s": report.warm_start.turnaround_seconds,
            "cost_per_minute_usd": report.cost_per_minute,
            "max_clean_concurrency": report.max_clean_concurrency,
        }
        # Must not raise
        serialised = json.dumps(raw)
        assert "cold" in serialised

    def test_mock_saves_to_file(self, tmp_path) -> None:
        from benchmarks.bench import _run_mock_benchmark
        import json
        report = _run_mock_benchmark()
        md_path = str(tmp_path / "metrics.md")
        raw_path = str(tmp_path / "raw.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        raw = {"cold": report.cold_start.turnaround_seconds}
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        assert Path(md_path).stat().st_size > 0
        assert Path(raw_path).stat().st_size > 0
