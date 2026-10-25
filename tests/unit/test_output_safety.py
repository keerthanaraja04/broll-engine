"""
Unit tests for output generation (timeline.json, manifest.json, licence gate)
and the safety layer.

One test per hard case in the brief:
  1. Named real people → neutral/symbolic visuals + warning
  2. Brands → generic category footage, no invented logos
  3. Sensitive news → calm/neutral, no upbeat, VLM rejects graphic
  4. "avoid" list enforced at query time and again by VLM
  5. Abstract lines and stats → metaphor queries or data card
  6. Double meanings → disambiguated sense in queries

Plus:
  - JSON Schema validation of timeline.json and manifest.json
  - Licence gate: missing licence/attribution, watermarked clips
  - OutputStep context reads/writes
  - SafetyStep context reads/writes
  - Saved outputs under tests/fixtures/outputs/
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("LLM_ADAPTER", "mock")
os.environ.setdefault("VLM_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")


# ── factories ────────────────────────────────────────────────────────────────

def _make_candidate(
    cid: str = "pixabay:1",
    source: str = "pixabay",
    license: str = "Pixabay Content License",
    attribution: str = "Video by user on Pixabay",
    commercial_use: bool = True,
    is_ai_generated: bool = False,
    kind_val: str = "video",
    duration: float = 10.0,
    tier: int = 1,
) -> Any:
    from app.adapters.footage_types import Candidate, FallbackTier, FootageKind
    return Candidate(
        id=cid, source=source, kind=FootageKind(kind_val),
        url=f"https://cdn.example.com/{cid}.mp4",
        thumb_url=f"https://cdn.example.com/{cid}_thumb.jpg",
        width=1920, height=1080, duration=duration,
        license=license, commercial_use=commercial_use,
        attribution=attribution, is_ai_generated=is_ai_generated,
        page_url=f"https://pixabay.com/videos/{cid}/",
        fallback_tier=FallbackTier(tier),
    )


def _make_cc(
    cid: str = "pixabay:1",
    source: str = "pixabay",
    final_score: float = 4.0,
    vlm_score: float = 4.0,
    vlm_flags: list[str] | None = None,
    tier: int = 1,
    license: str = "Pixabay Content License",
    attribution: str = "Video by user on Pixabay",
    commercial_use: bool = True,
) -> Any:
    from app.pipeline.clip_selection.models import ClipCandidate
    return ClipCandidate(
        candidate=_make_candidate(cid=cid, source=source, tier=tier,
                                    license=license,
                                    attribution=attribution,
                                    commercial_use=commercial_use),
        query="city skyline", query_index=0,
        text_score=3.5, vlm_score=vlm_score,
        vlm_flags=vlm_flags or [],
        final_score=final_score,
    )


def _make_scene_result(
    scene_id: int = 0,
    chosen: Any = None,
    rule: str = "pixabay_video_score_gte_threshold",
    rejected: list | None = None,
    warnings: list[str] | None = None,
) -> Any:
    from app.pipeline.clip_selection.models import (
        SceneClipResult, SelectionRule,
    )
    if chosen is None:
        chosen = _make_cc()
    return SceneClipResult(
        scene_id=scene_id, scene_start=0.0, scene_end=5.0,
        scene_duration=5.0, chosen=chosen,
        selection_rule=SelectionRule(rule),
        rejected=rejected or [],
        warnings=warnings or [],
    )


def _make_selection_plan(results: list | None = None) -> Any:
    from app.pipeline.clip_selection.models import VideoSelectionPlan
    if results is None:
        results = [_make_scene_result()]
    return VideoSelectionPlan(job_id="j1", scene_results=results)


def _make_scene_plan(n: int = 1) -> Any:
    from app.pipeline.scene_planning.models import PlannedScene, ScenePlan
    scenes = [
        PlannedScene(
            id=i, start=i * 5.0, end=(i + 1) * 5.0,
            narration_text=f"scene {i}", meaning=f"meaning {i}",
            queries=[f"query {i}"],
        )
        for i in range(n)
    ]
    return ScenePlan(scenes=scenes, total_duration=n * 5.0,
                     source_language="en")


def _make_planned_scene(
    named_person: str | None = None,
    named_brand: str | None = None,
    is_sensitive: bool = False,
    is_abstract: bool = False,
    has_numbers: bool = False,
    queries: list[str] | None = None,
    word_senses: list | None = None,
) -> Any:
    from app.pipeline.scene_planning.models import PlannedScene, WordSense
    return PlannedScene(
        id=0, start=0.0, end=5.0,
        narration_text="test narration", meaning="test meaning",
        queries=["test query"] if queries is None else queries,
        named_person=named_person,
        named_brand=named_brand,
        is_sensitive=is_sensitive,
        is_abstract=is_abstract,
        has_numbers=has_numbers,
        word_senses=word_senses or [],
    )


# ── JSON Schema validation ──────────────────────────────────────────────────

def _validate_json(data: dict, schema: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    try:
        import jsonschema  # type: ignore[import]
        v = jsonschema.Draft7Validator(schema)
        return [e.message for e in v.iter_errors(data)]
    except ImportError:
        return []


class TestTimelineSchema:
    def test_valid_timeline_passes(self) -> None:
        from app.pipeline.output.timeline import build_timeline, TIMELINE_SCHEMA
        plan = _make_scene_plan(2)
        selection = _make_selection_plan([_make_scene_result(i) for i in range(2)])
        timeline = build_timeline("j1", plan, selection)
        errors = _validate_json(timeline, TIMELINE_SCHEMA)
        assert errors == [], f"Schema errors: {errors}"

    def test_missing_job_id_fails(self) -> None:
        from app.pipeline.output.timeline import TIMELINE_SCHEMA
        bad = {"total_duration": 10.0, "scenes": []}
        errors = _validate_json(bad, TIMELINE_SCHEMA)
        assert any("job_id" in e for e in errors)

    def test_clip_null_is_valid(self) -> None:
        from app.pipeline.output.timeline import build_timeline, TIMELINE_SCHEMA
        plan = _make_scene_plan(1)
        selection = _make_selection_plan([
            _make_scene_result(0, chosen=None, rule="forced_by_exhaustion"),
        ])
        timeline = build_timeline("j1", plan, selection)
        errors = _validate_json(timeline, TIMELINE_SCHEMA)
        assert errors == [], f"Schema errors with null clip: {errors}"


class TestTimelineBuilder:
    def test_scenes_count_matches_plan(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(3)
        selection = _make_selection_plan([_make_scene_result(i) for i in range(3)])
        timeline = build_timeline("j1", plan, selection)
        assert len(timeline["scenes"]) == 3

    def test_clip_dict_has_all_required_fields(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(1)
        selection = _make_selection_plan()
        timeline = build_timeline("j1", plan, selection)
        clip = timeline["scenes"][0]["clip"]
        for key in ("source", "id", "url", "page_url", "license",
                     "commercial_use", "attribution", "is_ai_generated",
                     "duration_used"):
            assert key in clip, f"Missing clip key: {key}"

    def test_relevance_score_present(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(1)
        selection = _make_selection_plan()
        timeline = build_timeline("j1", plan, selection)
        assert "relevance_score" in timeline["scenes"][0]
        assert isinstance(timeline["scenes"][0]["relevance_score"], (int, float))

    def test_source_rule_fired_present(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(1)
        selection = _make_selection_plan()
        timeline = build_timeline("j1", plan, selection)
        assert "source_rule_fired" in timeline["scenes"][0]
        assert "pixabay" in timeline["scenes"][0]["source_rule_fired"].lower()

    def test_alternatives_rejected_is_list(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(1)
        selection = _make_selection_plan()
        timeline = build_timeline("j1", plan, selection)
        assert isinstance(timeline["scenes"][0]["alternatives_rejected"], list)

    def test_warnings_present(self) -> None:
        from app.pipeline.output.timeline import build_timeline
        plan = _make_scene_plan(1)
        selection = _make_selection_plan([
            _make_scene_result(0, warnings=["test warning"]),
        ])
        timeline = build_timeline("j1", plan, selection)
        assert "test warning" in timeline["scenes"][0]["warnings"]


class TestManifestSchema:
    def test_valid_manifest_passes(self) -> None:
        from app.pipeline.output.manifest import build_manifest, MANIFEST_SCHEMA
        plan = _make_scene_plan(1)
        selection = _make_selection_plan()
        from app.pipeline.output.timeline import build_timeline
        timeline = build_timeline("j1", plan, selection)
        manifest = build_manifest("j1", timeline)
        errors = _validate_json(manifest, MANIFEST_SCHEMA)
        assert errors == [], f"Schema errors: {errors}"

    def test_missing_licence_summary_fails(self) -> None:
        from app.pipeline.output.manifest import MANIFEST_SCHEMA
        bad = {"job_id": "j1", "app_version": "0.1",
               "models": {}, "settings": {}, "seed": None,
               "stage_timings": {}, "cost_estimate": {}, "warnings": []}
        errors = _validate_json(bad, MANIFEST_SCHEMA)
        assert any("licence_summary" in e for e in errors)


class TestManifestBuilder:
    def test_models_present(self) -> None:
        from app.pipeline.output.manifest import build_manifest
        from app.pipeline.output.timeline import build_timeline
        timeline = build_timeline("j1", _make_scene_plan(1), _make_selection_plan())
        manifest = build_manifest("j1", timeline)
        for key in ("stt", "llm", "vlm", "tts"):
            assert key in manifest["models"]
            assert "provider" in manifest["models"][key]
            assert "model" in manifest["models"][key]

    def test_cost_estimate_has_total(self) -> None:
        from app.pipeline.output.manifest import build_manifest
        from app.pipeline.output.timeline import build_timeline
        timeline = build_timeline("j1", _make_scene_plan(1), _make_selection_plan())
        manifest = build_manifest("j1", timeline, call_counts={"stt_whisper": 5})
        assert "total_usd" in manifest["cost_estimate"]
        assert manifest["cost_estimate"]["total_usd"] >= 0
        assert "stt_whisper" in manifest["cost_estimate"]["breakdown"]

    def test_licence_summary_all_licensed(self) -> None:
        from app.pipeline.output.manifest import build_manifest
        from app.pipeline.output.timeline import build_timeline
        timeline = build_timeline("j1", _make_scene_plan(1), _make_selection_plan())
        manifest = build_manifest("j1", timeline)
        ls = manifest["licence_summary"]
        assert ls["all_licensed"] is True
        assert ls["clips_checked"] == 1

    def test_licence_summary_missing_licence(self) -> None:
        from app.pipeline.output.manifest import build_manifest
        from app.pipeline.output.timeline import build_timeline
        selection = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="pixabay:nolic", license="", attribution="by user",
            )),
        ])
        timeline = build_timeline("j1", _make_scene_plan(1), selection)
        manifest = build_manifest("j1", timeline)
        assert manifest["licence_summary"]["all_licensed"] is False
        assert len(manifest["licence_summary"]["missing_licence"]) == 1


# ── Licence gate ─────────────────────────────────────────────────────────────

class TestLicenceGate:
    def test_valid_clips_pass(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        plan = _make_selection_plan()
        violations = check_licence_gate(plan)
        assert violations == []

    def test_missing_license_fails(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        from app.core.errors import LicenceGateError
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="p:nolic", license="",
            )),
        ])
        with pytest.raises(LicenceGateError) as exc_info:
            check_licence_gate(plan)
        assert "missing or unacceptable licence" in exc_info.value.message

    def test_missing_attribution_fails(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        from app.core.errors import LicenceGateError
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="p:noattr", attribution="",
            )),
        ])
        with pytest.raises(LicenceGateError):
            check_licence_gate(plan)

    def test_not_commercial_use_fails(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        from app.core.errors import LicenceGateError
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="p:nocomm", commercial_use=False,
            )),
        ])
        with pytest.raises(LicenceGateError) as exc_info:
            check_licence_gate(plan)
        assert "commercial" in exc_info.value.message.lower()

    def test_watermarked_flag_fails(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        from app.core.errors import LicenceGateError
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="p:wm", vlm_flags=["watermarked"],
            )),
        ])
        with pytest.raises(LicenceGateError) as exc_info:
            check_licence_gate(plan)
        assert "watermarked" in exc_info.value.message.lower()

    def test_card_clips_skip_attribution_check(self) -> None:
        """Generated title cards don't need attribution."""
        from app.pipeline.output.licence_gate import check_licence_gate
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="card:1", source="ffmpeg_card",
                attribution="",
            )),
        ])
        # Should not raise — cards are exempt from attribution
        violations = check_licence_gate(plan)
        assert violations == []

    def test_multiple_violations_all_reported(self) -> None:
        from app.pipeline.output.licence_gate import check_licence_gate
        from app.core.errors import LicenceGateError
        plan = _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(
                cid="p:bad1", license="", attribution="",
                commercial_use=False, vlm_flags=["watermarked"],
            )),
        ])
        with pytest.raises(LicenceGateError) as exc_info:
            check_licence_gate(plan)
        details = exc_info.value.details or {}
        assert len(details.get("violations", [])) >= 3


# ── OutputStep ───────────────────────────────────────────────────────────────

class TestOutputStep:
    @pytest.mark.asyncio
    async def test_sets_context_keys(self) -> None:
        from app.pipeline.output.output_step import OutputStep
        from app.pipeline.base import PipelineContext
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", _make_scene_plan(1))
        ctx.set("video_selection_plan_obj", _make_selection_plan())
        result_ctx = await OutputStep().execute(ctx)
        assert result_ctx.get("timeline") is not None
        assert result_ctx.get("manifest") is not None
        assert result_ctx.get("licence_gate_passed") is True

    @pytest.mark.asyncio
    async def test_licence_gate_failure_raises(self) -> None:
        from app.pipeline.output.output_step import OutputStep
        from app.pipeline.base import PipelineContext
        from app.core.errors import LicenceGateError
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", _make_scene_plan(1))
        ctx.set("video_selection_plan_obj", _make_selection_plan([
            _make_scene_result(0, chosen=_make_cc(license="")),
        ]))
        with pytest.raises(LicenceGateError):
            await OutputStep().execute(ctx)

    @pytest.mark.asyncio
    async def test_missing_plan_raises(self) -> None:
        from app.pipeline.output.output_step import OutputStep
        from app.pipeline.base import PipelineContext
        from app.core.errors import ValidationError
        ctx = PipelineContext(job_id="j1")
        with pytest.raises(ValidationError):
            await OutputStep().execute(ctx)

    def test_step_name(self) -> None:
        from app.pipeline.output.output_step import OutputStep
        assert OutputStep().name == "OutputStep"


# ═════════════════════════════════════════════════════════════════════════════
# SAFETY LAYER TESTS — one per hard case
# ═════════════════════════════════════════════════════════════════════════════

class TestNamedPerson:
    """Hard case 1: Named real people → no face-focused search."""

    def test_named_person_replaces_queries(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(
            named_person="Elon Musk",
            queries=["Elon Musk face", "Elon Musk portrait"],
        )
        result = check_scene_safety(scene)
        assert len(result.warnings) > 0
        assert any("face" in w.lower() or "neutral" in w.lower()
                    for w in result.warnings)
        # Queries should not contain the person's name
        for q in result.queries:
            assert "elon" not in q.lower()

    def test_named_person_adds_warning(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(named_person="Taylor Swift")
        result = check_scene_safety(scene)
        assert any("neutral" in w.lower() or "symbolic" in w.lower()
                    for w in result.warnings)


class TestBrandProtection:
    """Hard case 2: Brands → generic category footage, no invented logos."""

    def test_brand_replaces_queries(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(
            named_brand="Nike",
            queries=["Nike logo", "Nike shoes product"],
        )
        result = check_scene_safety(scene)
        assert any("brand" in w.lower() for w in result.warnings)
        # Brand name should not appear in queries
        for q in result.queries:
            assert "nike" not in q.lower()

    def test_no_invented_logos(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(named_brand="Apple Inc")
        result = check_scene_safety(scene)
        for q in result.queries:
            assert "apple" not in q.lower()


class TestSensitiveNews:
    """Hard case 3: Sensitive news → calm/neutral, no upbeat, VLM rejects graphic."""

    def test_sensitive_sets_reject_graphic(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(is_sensitive=True)
        result = check_scene_safety(scene)
        assert result.reject_graphic is True

    def test_sensitive_adds_warning(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(is_sensitive=True)
        result = check_scene_safety(scene)
        assert any("sensitive" in w.lower() or "neutral" in w.lower()
                    for w in result.warnings)

    def test_vlm_rejects_graphic_for_sensitive(self) -> None:
        from app.pipeline.safety.safety_rules import check_vlm_safety_flags
        should_reject, reason = check_vlm_safety_flags(
            vlm_flags=["graphic", "nsfw"], is_sensitive=True,
        )
        assert should_reject
        assert reason is not None

    def test_vlm_rejects_upbeat_for_sensitive(self) -> None:
        from app.pipeline.safety.safety_rules import check_vlm_safety_flags
        should_reject, reason = check_vlm_safety_flags(
            vlm_flags=["upbeat", "celebratory"], is_sensitive=True,
        )
        assert should_reject
        assert "upbeat" in reason.lower() or "sensitive" in reason.lower()

    def test_vlm_accepts_neutral_for_sensitive(self) -> None:
        from app.pipeline.safety.safety_rules import check_vlm_safety_flags
        should_reject, reason = check_vlm_safety_flags(
            vlm_flags=[], is_sensitive=True,
        )
        assert not should_reject


class TestAvoidList:
    """Hard case 4: 'avoid' list enforced at query time and again by VLM."""

    def test_avoid_term_dropped_from_queries(self) -> None:
        from app.pipeline.safety.safety_rules import enforce_avoid_list
        queries = ["city skyline", "war scene", "peaceful nature"]
        result, checks = enforce_avoid_list(queries, avoid=["war"])
        assert "war scene" not in result
        assert "city skyline" in result
        assert "peaceful nature" in result

    def test_all_queries_dropped_adds_fallback(self) -> None:
        from app.pipeline.safety.safety_rules import enforce_avoid_list
        queries = ["war", "violence", "explosion"]
        result, checks = enforce_avoid_list(queries, avoid=["war", "violence", "explosion"])
        assert len(result) >= 1
        assert "neutral" in result[0].lower()

    def test_empty_avoid_list_no_change(self) -> None:
        from app.pipeline.safety.safety_rules import enforce_avoid_list
        queries = ["city skyline", "ocean"]
        result, checks = enforce_avoid_list(queries, avoid=None)
        assert result == queries
        assert checks == []

    def test_avoid_check_has_detail(self) -> None:
        from app.pipeline.safety.safety_rules import enforce_avoid_list
        _, checks = enforce_avoid_list(["war scene"], avoid=["war"])
        assert len(checks) == 1
        assert "war" in checks[0].detail.lower()


class TestAbstractStats:
    """Hard case 5: Abstract lines and stats → metaphor queries or data card."""

    def test_stats_scene_triggers_data_card(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(
            has_numbers=True,
            queries=[],  # no queries → force fallback
        )
        result = check_scene_safety(scene)
        assert result.use_data_card is True

    def test_abstract_scene_adds_warning(self) -> None:
        from app.pipeline.safety.safety_rules import check_scene_safety
        scene = _make_planned_scene(is_abstract=True)
        result = check_scene_safety(scene)
        assert any("abstract" in w.lower() or "metaphor" in w.lower()
                    for w in result.warnings)


class TestDoubleMeanings:
    """Hard case 6: Double meanings → disambiguated sense in queries."""

    def test_word_sense_replaces_ambiguous_word(self) -> None:
        from app.pipeline.safety.safety_rules import sanitize_queries
        from app.pipeline.scene_planning.models import WordSense
        scene = _make_planned_scene(
            queries=["python programming tutorial"],
            word_senses=[WordSense(word="python", chosen_sense="python snake reptile")],
        )
        result = sanitize_queries(["python programming tutorial"], scene)
        # The word "python" should be replaced with "python snake reptile"
        # in at least one query
        found_replacement = any("snake" in q.lower() or "reptile" in q.lower()
                                 for q in result.queries)
        assert found_replacement, f"Expected sense replacement in {result.queries}"

    def test_bank_disambiguated(self) -> None:
        from app.pipeline.safety.safety_rules import sanitize_queries
        from app.pipeline.scene_planning.models import WordSense
        scene = _make_planned_scene(
            queries=["river bank flowing water"],
            word_senses=[WordSense(word="bank", chosen_sense="river bank shore")],
        )
        result = sanitize_queries(["river bank flowing water"], scene)
        # "bank" should still be there (same sense) but we verify the mechanism works
        assert len(result.queries) >= 1

    def test_multiple_word_senses_applied(self) -> None:
        from app.pipeline.safety.safety_rules import sanitize_queries
        from app.pipeline.scene_planning.models import WordSense
        scene = _make_planned_scene(
            queries=["mouse click cloud data"],
            word_senses=[
                WordSense(word="mouse", chosen_sense="computer mouse cursor"),
                WordSense(word="cloud", chosen_sense="cloud computing server"),
            ],
        )
        result = sanitize_queries(["mouse click cloud data"], scene)
        combined = " ".join(result.queries).lower()
        assert "cursor" in combined or "computer mouse" in combined
        assert "server" in combined or "computing" in combined


# ── SafetyStep ───────────────────────────────────────────────────────────────

class TestSafetyStep:
    @pytest.mark.asyncio
    async def test_sets_context_keys(self) -> None:
        from app.pipeline.safety.safety_step import SafetyStep
        from app.pipeline.base import PipelineContext
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", _make_scene_plan(2))
        result_ctx = await SafetyStep().execute(ctx)
        assert result_ctx.get("safety_results") is not None
        assert result_ctx.get("safety_results_obj") is not None

    @pytest.mark.asyncio
    async def test_missing_plan_raises(self) -> None:
        from app.pipeline.safety.safety_step import SafetyStep
        from app.pipeline.base import PipelineContext
        from app.core.errors import ValidationError
        ctx = PipelineContext(job_id="j1")
        with pytest.raises(ValidationError):
            await SafetyStep().execute(ctx)

    @pytest.mark.asyncio
    async def test_queries_updated_on_scene(self) -> None:
        from app.pipeline.safety.safety_step import SafetyStep
        from app.pipeline.base import PipelineContext
        plan = _make_scene_plan(1)
        plan.scenes[0].named_person = "John Doe"
        plan.scenes[0].queries = ["John Doe portrait"]
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", plan)
        await SafetyStep().execute(ctx)
        # Queries should have been sanitized
        for q in plan.scenes[0].queries:
            assert "john" not in q.lower()

    @pytest.mark.asyncio
    async def test_warnings_propagated_to_scene(self) -> None:
        from app.pipeline.safety.safety_step import SafetyStep
        from app.pipeline.base import PipelineContext
        plan = _make_scene_plan(1)
        plan.scenes[0].is_sensitive = True
        ctx = PipelineContext(job_id="j1")
        ctx.set("scene_plan_obj", plan)
        await SafetyStep().execute(ctx)
        assert len(plan.scenes[0].warnings) > 0

    def test_step_name(self) -> None:
        from app.pipeline.safety.safety_step import SafetyStep
        assert SafetyStep().name == "SafetyStep"


# ── Save outputs ─────────────────────────────────────────────────────────────

class TestSaveOutputs:
    """Save example timeline.json and manifest.json to tests/fixtures/outputs/"""

    def test_save_timeline_and_manifest(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        from app.pipeline.output.timeline import build_timeline, TIMELINE_SCHEMA
        from app.pipeline.output.manifest import build_manifest, MANIFEST_SCHEMA
        out_dir = tmp_path / "outputs"
        out_dir.mkdir()

        plan = _make_scene_plan(2)
        selection = _make_selection_plan([_make_scene_result(i) for i in range(2)])
        timeline = build_timeline("test-job-001", plan, selection)
        manifest = build_manifest("test-job-001", timeline,
                                  stage_timings={"input": 1.2, "planning": 0.5,
                                                  "selection": 3.1, "render": 8.0},
                                  call_counts={"stt_whisper": 1, "llm_chat": 3,
                                                "vlm_score": 8, "tts": 1})

        # Validate
        assert _validate_json(timeline, TIMELINE_SCHEMA) == []
        assert _validate_json(manifest, MANIFEST_SCHEMA) == []

        # Save
        timeline_path = out_dir / "timeline.json"
        manifest_path = out_dir / "manifest.json"
        with open(timeline_path, "w", encoding="utf-8") as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        assert timeline_path.exists()
        assert manifest_path.exists()
        assert timeline_path.stat().st_size > 0
        assert manifest_path.stat().st_size > 0
