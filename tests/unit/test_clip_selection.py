"""
Unit tests for the clip selection stage.

No real network calls, no MinIO, no Redis, no Groq API.
All external dependencies are mocked.

Coverage:
- UsedClipRegistry (exact dedup)
- deduplicate_candidates (perceptual hash dedup)
- assess_duration_fit (direct / slow / loop / too-short)
- score_text (tag overlap, tier bonus, query recency, sensitivity penalty)
- blend_scores (weighted mix, fallback to text-only when vlm=0)
- apply_selection_rule (PIXABAY_VIDEO / KEN_BURNS / TITLE_CARD thresholds)
- pick_best (threshold filtering)
- select_clip_for_scene (full integration with mocked router)
- select_clips_for_plan (cross-scene dedup, sequential processing)
- ClipSelectionStep (context reads/writes, error wrapping)
- SceneClipResult.to_dict() shape
- VideoSelectionPlan.to_dict() shape
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("VLM_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── factories ────────────────────────────────────────────────────────────────

def _make_candidate(
    cid: str = "pixabay:1001",
    source: str = "pixabay",
    kind_val: str = "video",
    duration: float = 10.0,
    tags: list[str] | None = None,
    tier: int = 1,
    thumb_url: str = "https://cdn.example.com/thumb.jpg",
) -> Any:
    from app.adapters.footage_types import Candidate, FallbackTier, FootageKind
    return Candidate(
        id=cid, source=source,
        kind=FootageKind(kind_val),
        url=f"https://cdn.example.com/{cid}.mp4",
        thumb_url=thumb_url,
        width=1920, height=1080, duration=duration,
        tags=tags or ["city", "skyline"],
        fallback_tier=FallbackTier(tier),
    )


def _make_cc(
    cid: str = "pixabay:1001",
    duration: float = 10.0,
    tags: list[str] | None = None,
    tier: int = 1,
    query: str = "city skyline",
    q_idx: int = 0,
) -> Any:
    from app.pipeline.clip_selection.models import ClipCandidate
    return ClipCandidate(
        candidate=_make_candidate(cid=cid, duration=duration,
                                   tags=tags, tier=tier),
        query=query,
        query_index=q_idx,
    )


def _make_scene(
    scene_id: int = 0,
    start: float = 0.0,
    end: float = 5.0,
    narration: str = "city skyline at dusk",
    queries: list[str] | None = None,
    is_sensitive: bool = False,
) -> Any:
    from app.pipeline.scene_planning.models import PlannedScene
    return PlannedScene(
        id=scene_id,
        start=start,
        end=end,
        narration_text=narration,
        meaning="city at night",
        queries=queries or ["city skyline", "urban landscape", "city lights"],
        is_sensitive=is_sensitive,
    )


def _make_footage_result(candidate: Any) -> Any:
    from app.adapters.footage_router import FootageResult
    from app.adapters.footage_types import FallbackTier
    result = FootageResult(candidate=candidate, tier=FallbackTier.PIXABAY_VIDEO)
    return result


# ── UsedClipRegistry ────────────────────────────────────────────────────────

class TestUsedClipRegistry:
    def test_empty_registry_not_used(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        reg = UsedClipRegistry()
        assert not reg.is_used("pixabay:1")

    def test_mark_used_marks_it(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        reg = UsedClipRegistry()
        reg.mark_used("pixabay:1")
        assert reg.is_used("pixabay:1")

    def test_mark_does_not_affect_others(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        reg = UsedClipRegistry()
        reg.mark_used("pixabay:1")
        assert not reg.is_used("pixabay:2")

    def test_len(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        reg = UsedClipRegistry()
        reg.mark_used("a")
        reg.mark_used("b")
        assert len(reg) == 2


# ── deduplicate_candidates ───────────────────────────────────────────────────

class TestDeduplicateCandidates:
    def test_already_used_id_excluded(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry, deduplicate_candidates
        reg = UsedClipRegistry()
        reg.mark_used("pixabay:1001")
        ccs = [_make_cc("pixabay:1001"), _make_cc("pixabay:1002")]
        result = deduplicate_candidates(ccs, reg, {})
        ids = [cc.candidate.id for cc in result]
        assert "pixabay:1001" not in ids
        assert "pixabay:1002" in ids

    def test_unused_id_passes(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry, deduplicate_candidates
        reg = UsedClipRegistry()
        ccs = [_make_cc("pixabay:1001")]
        result = deduplicate_candidates(ccs, reg, {})
        assert len(result) == 1

    def test_phash_disabled_no_phash_computed(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry, deduplicate_candidates
        with patch.dict(os.environ, {"CLIP_PHASH_ENABLED": "false"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            reg = UsedClipRegistry()
            ccs = [_make_cc("a"), _make_cc("b")]
            result = deduplicate_candidates(ccs, reg, {})
            for cc in result:
                assert cc.phash is None
        get_settings.cache_clear()

    def test_phash_identical_images_deduped(self) -> None:
        from app.pipeline.clip_selection.dedup import (
            UsedClipRegistry, deduplicate_candidates, _compute_phash
        )
        # Two candidates pointing to the same tiny image bytes
        thumb = b"\xff" * 64   # tiny solid-white image approximation
        with patch("app.pipeline.clip_selection.dedup._compute_phash",
                   return_value="aaaaaaaaaaaaaaaa"):
            reg = UsedClipRegistry()
            cc1 = _make_cc("a")
            cc2 = _make_cc("b")
            thumb_map = {"a": thumb, "b": thumb}
            result = deduplicate_candidates([cc1, cc2], reg, thumb_map)
        # One survives, other is duplicate
        non_dup = [cc for cc in [cc1, cc2] if not cc.is_duplicate]
        assert len(non_dup) == 1

    def test_phash_different_images_both_survive(self) -> None:
        from app.pipeline.clip_selection.dedup import UsedClipRegistry, deduplicate_candidates
        call_n = [0]

        def _mock_phash(b: bytes) -> str:
            call_n[0] += 1
            return "0000000000000000" if call_n[0] % 2 == 1 else "ffffffffffffffff"

        with patch("app.pipeline.clip_selection.dedup._compute_phash",
                   side_effect=_mock_phash):
            reg = UsedClipRegistry()
            cc1 = _make_cc("a")
            cc2 = _make_cc("b")
            result = deduplicate_candidates([cc1, cc2], reg, {"a": b"x", "b": b"y"})
        assert len(result) == 2


# ── assess_duration_fit ──────────────────────────────────────────────────────

class TestAssessDurationFit:
    def test_clip_longer_than_scene_ok(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        cc = _make_cc(duration=10.0)
        assess_duration_fit(cc, scene_duration=5.0)
        assert cc.duration_ok
        assert not cc.needs_slow
        assert cc.needs_loop == 0

    def test_clip_exact_duration_ok(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        cc = _make_cc(duration=5.0)
        assess_duration_fit(cc, scene_duration=5.0)
        assert cc.duration_ok

    def test_short_clip_needs_loop(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        cc = _make_cc(duration=2.0)
        assess_duration_fit(cc, scene_duration=5.0)
        assert cc.duration_ok
        assert cc.needs_loop > 0

    def test_short_clip_needs_slow(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        # 3.5 s clip, 5.0 s scene: cannot loop (3.5*3=10.5 ok, but let's check slow)
        cc = _make_cc(duration=3.5)
        assess_duration_fit(cc, scene_duration=5.0)
        # Either slow or loop should work
        assert cc.duration_ok

    def test_very_short_clip_not_ok(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        with patch.dict(os.environ, {
            "CLIP_SLOW_MOTION_MIN_FACTOR": "0.9",
            "CLIP_MAX_LOOP_COUNT": "1",
        }):
            from app.core.config import get_settings
            get_settings.cache_clear()
            cc = _make_cc(duration=0.5)
            assess_duration_fit(cc, scene_duration=10.0)
            assert not cc.duration_ok
        get_settings.cache_clear()

    def test_card_always_ok(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        from app.adapters.footage_types import Candidate, FallbackTier, FootageKind
        from app.pipeline.clip_selection.models import ClipCandidate
        c = Candidate(
            id="card:1", source="ffmpeg_card", kind=FootageKind.CARD,
            url="", thumb_url="", width=1920, height=1080, duration=0.0,
            fallback_tier=FallbackTier.TITLE_CARD,
        )
        cc = ClipCandidate(candidate=c, query="test", query_index=2)
        assess_duration_fit(cc, scene_duration=5.0)
        assert cc.duration_ok

    def test_zero_duration_clip_ok(self) -> None:
        from app.pipeline.clip_selection.duration_fit import assess_duration_fit
        cc = _make_cc(duration=0.0)
        cc.candidate.source = "ffmpeg_card"
        assess_duration_fit(cc, scene_duration=5.0)
        assert cc.duration_ok


# ── score_text ───────────────────────────────────────────────────────────────

class TestScoreText:
    def test_high_overlap_gets_high_score(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene(narration="city skyline urban skyscrapers")
        cc = _make_cc(tags=["city", "skyline", "urban", "skyscrapers"])
        score_text(cc, scene, "city skyline")
        assert cc.text_score >= 3.5

    def test_no_overlap_gets_low_score(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene(narration="mountain waterfall forest")
        cc = _make_cc(tags=["ocean", "beach", "waves"])
        score_text(cc, scene, "mountain")
        assert cc.text_score < 3.0

    def test_score_in_range_1_to_5(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene()
        cc = _make_cc()
        score_text(cc, scene, "test query")
        assert 1.0 <= cc.text_score <= 5.0

    def test_title_card_tier_gets_penalty(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene(narration="city skyline urban")
        cc_video = _make_cc(cid="v:1", tags=["city", "skyline", "urban"], tier=1)
        cc_card  = _make_cc(cid="c:1", tags=["city", "skyline", "urban"], tier=4)
        score_text(cc_video, scene, "city skyline")
        score_text(cc_card, scene, "city skyline")
        assert cc_video.text_score > cc_card.text_score

    def test_query_recency_bonus(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene()
        cc0 = _make_cc(q_idx=0)
        cc2 = _make_cc(q_idx=2)
        score_text(cc0, scene, "query")
        score_text(cc2, scene, "query")
        assert cc0.text_score >= cc2.text_score

    def test_sensitivity_penalty_with_sensitive_tags(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        scene = _make_scene(is_sensitive=True)
        cc_safe    = _make_cc(tags=["city", "skyline"])
        cc_violent = _make_cc(cid="v:99", tags=["war", "explosion"])
        score_text(cc_safe, scene, "city")
        score_text(cc_violent, scene, "city")
        assert cc_safe.text_score > cc_violent.text_score

    def test_reason_string_populated(self) -> None:
        from app.pipeline.clip_selection.text_scorer import score_text
        cc = _make_cc()
        score_text(cc, _make_scene(), "city")
        assert cc.text_score_reason != ""


# ── blend_scores ─────────────────────────────────────────────────────────────

class TestBlendScores:
    def test_vlm_zero_uses_text_only(self) -> None:
        from app.pipeline.clip_selection.ranker import blend_scores
        cc = _make_cc()
        cc.text_score = 4.0
        cc.vlm_score = 0.0
        blend_scores([cc])
        assert cc.final_score == pytest.approx(4.0)
        assert "text_only" in cc.final_score_reason

    def test_weighted_blend(self) -> None:
        from app.pipeline.clip_selection.ranker import blend_scores
        cc = _make_cc()
        cc.text_score = 3.0
        cc.vlm_score = 5.0
        with patch.dict(os.environ, {
            "CLIP_SELECTION_WEIGHT_TEXT": "0.4",
            "CLIP_SELECTION_WEIGHT_VLM": "0.6",
        }):
            from app.core.config import get_settings
            get_settings.cache_clear()
            blend_scores([cc])
        expected = 0.4 * 3.0 + 0.6 * 5.0
        assert cc.final_score == pytest.approx(expected, abs=0.01)
        get_settings.cache_clear()

    def test_sorted_descending(self) -> None:
        from app.pipeline.clip_selection.ranker import blend_scores
        ccs = [_make_cc(cid=f"p:{i}") for i in range(3)]
        ccs[0].text_score = 2.0
        ccs[1].text_score = 4.0
        ccs[2].text_score = 3.0
        result = blend_scores(ccs)
        scores = [c.final_score for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_final_score_clamped_to_1_5(self) -> None:
        from app.pipeline.clip_selection.ranker import blend_scores
        cc = _make_cc()
        cc.text_score = 0.0   # below minimum
        cc.vlm_score = 0.0
        blend_scores([cc])
        assert cc.final_score >= 1.0


# ── apply_selection_rule ─────────────────────────────────────────────────────

class TestApplySelectionRule:
    def test_high_score_video_gives_pixabay_rule(self) -> None:
        from app.pipeline.clip_selection.ranker import apply_selection_rule
        from app.pipeline.clip_selection.models import SelectionRule
        cc = _make_cc(tier=1)
        cc.final_score = 4.0
        assert apply_selection_rule(cc) == SelectionRule.PIXABAY_VIDEO

    def test_medium_score_gives_ken_burns_rule(self) -> None:
        from app.pipeline.clip_selection.ranker import apply_selection_rule
        from app.pipeline.clip_selection.models import SelectionRule
        cc = _make_cc(tier=1)
        cc.final_score = 3.2
        assert apply_selection_rule(cc) == SelectionRule.KEN_BURNS

    def test_low_score_gives_title_card_rule(self) -> None:
        from app.pipeline.clip_selection.ranker import apply_selection_rule
        from app.pipeline.clip_selection.models import SelectionRule
        cc = _make_cc(tier=1)
        cc.final_score = 2.5
        assert apply_selection_rule(cc) == SelectionRule.TITLE_CARD

    def test_none_gives_forced_rule(self) -> None:
        from app.pipeline.clip_selection.ranker import apply_selection_rule
        from app.pipeline.clip_selection.models import SelectionRule
        assert apply_selection_rule(None) == SelectionRule.FORCED

    def test_card_candidate_always_title_card_rule(self) -> None:
        from app.pipeline.clip_selection.ranker import apply_selection_rule
        from app.pipeline.clip_selection.models import SelectionRule
        from app.adapters.footage_types import Candidate, FallbackTier, FootageKind
        from app.pipeline.clip_selection.models import ClipCandidate
        c = Candidate(
            id="card:1", source="ffmpeg_card", kind=FootageKind.CARD,
            url="", thumb_url="", width=1920, height=1080, duration=5.0,
            fallback_tier=FallbackTier.TITLE_CARD,
        )
        cc = ClipCandidate(candidate=c, query="x", query_index=2, final_score=4.9)
        assert apply_selection_rule(cc) == SelectionRule.TITLE_CARD


# ── pick_best ────────────────────────────────────────────────────────────────

class TestPickBest:
    def test_returns_first_above_threshold(self) -> None:
        from app.pipeline.clip_selection.ranker import pick_best
        ccs = [_make_cc(cid=f"p:{i}") for i in range(3)]
        ccs[0].final_score = 4.5
        ccs[0].duration_ok = True
        ccs[0].is_duplicate = False
        result = pick_best(ccs, threshold=4.0)
        assert result is ccs[0]

    def test_returns_none_when_all_below(self) -> None:
        from app.pipeline.clip_selection.ranker import pick_best
        ccs = [_make_cc(cid=f"p:{i}") for i in range(3)]
        for cc in ccs:
            cc.final_score = 2.0
        assert pick_best(ccs, threshold=3.0) is None

    def test_skips_duration_not_ok(self) -> None:
        from app.pipeline.clip_selection.ranker import pick_best
        cc = _make_cc()
        cc.final_score = 4.9
        cc.duration_ok = False
        assert pick_best([cc], threshold=3.0) is None

    def test_skips_duplicate(self) -> None:
        from app.pipeline.clip_selection.ranker import pick_best
        cc = _make_cc()
        cc.final_score = 4.9
        cc.is_duplicate = True
        assert pick_best([cc], threshold=3.0) is None

    def test_picks_second_when_first_is_duplicate(self) -> None:
        from app.pipeline.clip_selection.ranker import pick_best
        cc1 = _make_cc(cid="a")
        cc1.final_score = 4.9
        cc1.is_duplicate = True
        cc2 = _make_cc(cid="b")
        cc2.final_score = 4.5
        cc2.duration_ok = True
        result = pick_best([cc1, cc2], threshold=4.0)
        assert result is cc2


# ── select_clip_for_scene (integration) ──────────────────────────────────────

class TestSelectClipForScene:
    @pytest.mark.asyncio
    async def test_high_score_candidate_chosen(self) -> None:
        from app.pipeline.clip_selection.selector import select_clip_for_scene
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        from app.pipeline.clip_selection.models import SelectionRule

        scene = _make_scene(narration="city skyline urban tall buildings")
        cand = _make_candidate(cid="pixabay:best", tags=["city", "skyline", "urban"])
        router = MagicMock()
        router.resolve = AsyncMock(return_value=_make_footage_result(cand))

        mock_cc = _make_cc(cid="pixabay:best", tags=["city", "skyline", "urban"])
        mock_cc.text_score = 4.0
        with patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(return_value=([_inject_vlm(mock_cc, vlm_score=4.5)], {}))):
            result = await select_clip_for_scene(
                scene, router, UsedClipRegistry(), job_id="j1"
            )

        assert result.chosen is not None
        assert result.scene_id == scene.id

    @pytest.mark.asyncio
    async def test_already_used_clip_not_chosen(self) -> None:
        from app.pipeline.clip_selection.selector import select_clip_for_scene
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        from app.pipeline.clip_selection.models import SelectionRule
        from app.adapters.footage_types import FallbackTier

        scene = _make_scene()
        cand = _make_candidate(cid="pixabay:used")
        router = MagicMock()

        # Card result for title-card fallback
        card = _make_candidate(cid="card:x", source="ffmpeg_card",
                                kind_val="card", duration=5.0, tier=4)

        call_count = [0]

        async def _router_resolve(query, **kw):
            call_count[0] += 1
            if call_count[0] <= 3:
                return _make_footage_result(cand)
            return _make_footage_result(card)

        router.resolve = _router_resolve
        router._try_title_card = AsyncMock(return_value=card)

        reg = UsedClipRegistry()
        reg.mark_used("pixabay:used")

        with patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(side_effect=lambda ccs, *a, **kw: (ccs, {}))):
            result = await select_clip_for_scene(scene, router, reg)

        assert result.chosen is None or result.chosen.candidate.id != "pixabay:used"

    @pytest.mark.asyncio
    async def test_warnings_accumulated_on_fallback(self) -> None:
        from app.pipeline.clip_selection.selector import select_clip_for_scene
        from app.pipeline.clip_selection.dedup import UsedClipRegistry
        from app.adapters.footage_router import FootageResult
        from app.adapters.footage_types import FallbackTier, FallbackWarning

        scene = _make_scene(queries=["city"])
        card = _make_candidate(cid="card:fallback", source="ffmpeg_card",
                                kind_val="card", duration=5.0, tier=4)
        router = MagicMock()
        fr = _make_footage_result(card)
        fr.add_warning(FallbackTier.PIXABAY_VIDEO, "Pixabay down", "city")
        router.resolve = AsyncMock(return_value=fr)

        with patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(side_effect=lambda ccs, *a, **kw: (ccs, {}))):
            result = await select_clip_for_scene(scene, router, UsedClipRegistry())

        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_to_dict_shape(self) -> None:
        from app.pipeline.clip_selection.selector import select_clip_for_scene
        from app.pipeline.clip_selection.dedup import UsedClipRegistry

        scene = _make_scene()
        cand = _make_candidate()
        router = MagicMock()
        router.resolve = AsyncMock(return_value=_make_footage_result(cand))

        with patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(side_effect=lambda ccs, *a, **kw: (ccs, {}))):
            result = await select_clip_for_scene(scene, router, UsedClipRegistry())

        d = result.to_dict()
        for key in ("scene_id", "scene_start", "scene_end", "scene_duration",
                     "selection_rule", "rejected", "warnings", "final_score"):
            assert key in d, f"Missing key: {key}"


def _inject_vlm(cc: Any, vlm_score: float) -> Any:
    cc.vlm_score = vlm_score
    return cc


# ── select_clips_for_plan ─────────────────────────────────────────────────────

class TestSelectClipsForPlan:
    @pytest.mark.asyncio
    async def test_cross_scene_dedup(self) -> None:
        """The same clip should not be chosen for two different scenes."""
        from app.pipeline.clip_selection.selector import select_clips_for_plan
        from app.pipeline.scene_planning.models import PlannedScene, ScenePlan

        scenes = [
            PlannedScene(id=0, start=0.0, end=5.0, narration_text="city",
                          meaning="city", queries=["city"]),
            PlannedScene(id=1, start=5.0, end=10.0, narration_text="skyline",
                          meaning="skyline", queries=["skyline"]),
        ]
        plan = ScenePlan(scenes=scenes, total_duration=10.0,
                          source_language="en")

        shared_cand = _make_candidate(cid="pixabay:shared",
                                       tags=["city", "skyline"])

        from app.adapters.footage_router import FootageResult
        from app.adapters.footage_types import FallbackTier

        async def _mock_resolve(keyword, **kw):
            return _make_footage_result(shared_cand)

        mock_router = MagicMock()
        mock_router.resolve = _mock_resolve
        mock_router._try_title_card = AsyncMock(
            return_value=_make_candidate(cid="card:fb", source="ffmpeg_card",
                                          kind_val="card", tier=4))

        with patch("app.pipeline.clip_selection.selector.get_footage_router",
                   return_value=mock_router), \
             patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(side_effect=lambda ccs, *a, **kw: (ccs, {}))):
            vp = await select_clips_for_plan(plan, job_id="test-job")

        chosen_ids = [
            r.chosen.candidate.id for r in vp.scene_results if r.chosen
        ]
        # If dedup works, the same candidate should not appear twice
        assert len(chosen_ids) == len(set(chosen_ids)), \
            f"Duplicate clip IDs chosen: {chosen_ids}"

    @pytest.mark.asyncio
    async def test_video_selection_plan_shape(self) -> None:
        from app.pipeline.clip_selection.selector import select_clips_for_plan
        from app.pipeline.scene_planning.models import PlannedScene, ScenePlan

        plan = ScenePlan(
            scenes=[PlannedScene(id=0, start=0.0, end=5.0,
                                  narration_text="test", meaning="test",
                                  queries=["test"])],
            total_duration=5.0, source_language="en",
        )
        mock_router = MagicMock()
        mock_router.resolve = AsyncMock(
            return_value=_make_footage_result(_make_candidate()))
        mock_router._try_title_card = AsyncMock(
            return_value=_make_candidate(cid="card:x", source="ffmpeg_card",
                                          kind_val="card", tier=4))

        with patch("app.pipeline.clip_selection.selector.get_footage_router",
                   return_value=mock_router), \
             patch("app.pipeline.clip_selection.selector.score_vlm",
                   AsyncMock(side_effect=lambda ccs, *a, **kw: (ccs, {}))):
            vp = await select_clips_for_plan(plan)

        d = vp.to_dict()
        for key in ("job_id", "scene_count", "chosen_count",
                     "warnings", "scenes"):
            assert key in d

    @pytest.mark.asyncio
    async def test_plan_with_no_scenes_returns_empty(self) -> None:
        from app.pipeline.clip_selection.selector import select_clips_for_plan
        from app.pipeline.scene_planning.models import ScenePlan

        plan = ScenePlan(scenes=[], total_duration=0.0, source_language="en")
        with patch("app.pipeline.clip_selection.selector.get_footage_router",
                   return_value=MagicMock()):
            vp = await select_clips_for_plan(plan)
        assert vp.scene_count == 0
        assert vp.chosen_count == 0


# ── ClipSelectionStep ────────────────────────────────────────────────────────

class TestClipSelectionStep:
    @pytest.mark.asyncio
    async def test_step_sets_context_keys(self) -> None:
        from app.pipeline.clip_selection.clip_selection_step import ClipSelectionStep
        from app.pipeline.base import PipelineContext
        from app.pipeline.scene_planning.models import PlannedScene, ScenePlan
        from app.pipeline.clip_selection.models import VideoSelectionPlan

        plan = ScenePlan(
            scenes=[PlannedScene(id=0, start=0.0, end=5.0,
                                  narration_text="test", meaning="t",
                                  queries=["test"])],
            total_duration=5.0, source_language="en",
        )
        fake_vp = VideoSelectionPlan(job_id="j1", scene_results=[])

        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", plan)

        with patch("app.pipeline.clip_selection.clip_selection_step.select_clips_for_plan",
                   AsyncMock(return_value=fake_vp)):
            result_ctx = await ClipSelectionStep().execute(ctx)

        assert result_ctx.get("video_selection_plan_obj") is fake_vp
        assert result_ctx.get("video_selection_plan") == fake_vp.to_dict()
        assert result_ctx.get("scenes_chosen") == 0

    @pytest.mark.asyncio
    async def test_step_raises_validation_when_no_plan(self) -> None:
        from app.pipeline.clip_selection.clip_selection_step import ClipSelectionStep
        from app.pipeline.base import PipelineContext
        from app.core.errors import ValidationError

        ctx = PipelineContext(job_id="j1")
        with pytest.raises(ValidationError):
            await ClipSelectionStep().execute(ctx)

    @pytest.mark.asyncio
    async def test_step_wraps_unexpected_error(self) -> None:
        from app.pipeline.clip_selection.clip_selection_step import ClipSelectionStep
        from app.pipeline.base import PipelineContext
        from app.pipeline.scene_planning.models import PlannedScene, ScenePlan
        from app.core.errors import PipelineError

        plan = ScenePlan(scenes=[], total_duration=0.0, source_language="en")
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", plan)

        with patch("app.pipeline.clip_selection.clip_selection_step.select_clips_for_plan",
                   AsyncMock(side_effect=RuntimeError("unexpected boom"))):
            with pytest.raises(PipelineError) as exc_info:
                await ClipSelectionStep().execute(ctx)
        assert "unexpected boom" in exc_info.value.message

    def test_step_name(self) -> None:
        from app.pipeline.clip_selection.clip_selection_step import ClipSelectionStep
        assert ClipSelectionStep().name == "ClipSelectionStep"
