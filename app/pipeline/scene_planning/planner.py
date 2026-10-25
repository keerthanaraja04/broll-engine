"""
Scene planner — orchestrates LLM call → validate/repair → fallback.

Public API
----------
plan_scenes(transcript, style, avoid, visual_hints) -> ScenePlan

Flow
----
1. Build system + user prompts from the transcript and job metadata.
2. Call LLMAdapter.json_complete() with SCENE_PLAN_SCHEMA.
3. Validate and repair the result with validate_and_repair().
4. If validation still fails (after the LLM's own repair retries), and
   SCENE_PLAN_FALLBACK_ENABLED is True, fall back to build_rule_based_plan().
5. Propagate visual_hints from scenes[] input: if the transcript source is
   "scenes", attach each Sentence's visual_hint to the closest PlannedScene.
6. Return ScenePlan.

Hindi / Tamil handling
----------------------
The language code is passed to the LLM via the prompt.  The system prompt
instructs the model to write meaning in the source language but queries in
English.  No special code is required here.
"""

from __future__ import annotations

from typing import Any

from app.adapters.llm_adapter import LLMSchemaError, get_llm_adapter
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.input_stage.transcript import Sentence, Transcript
from app.pipeline.scene_planning.models import PlannedScene, ScenePlan
from app.pipeline.scene_planning.prompts import build_prompts
from app.pipeline.scene_planning.rule_based import build_rule_based_plan
from app.pipeline.scene_planning.validator import (
    ScenePlanValidationError,
    validate_and_repair,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentence_list_text(sentences: list[Sentence]) -> str:
    lines = [
        f"  [{s.id}] {s.start:.2f}s–{s.end:.2f}s  {s.text}"
        for s in sentences
    ]
    return "\n".join(lines)


def _collect_visual_hints(
    transcript: Transcript,
) -> list[tuple[float, float, str]] | None:
    """Return visual hint tuples from a scenes-source transcript, or None."""
    if transcript.source != "scenes":
        return None
    hints = [
        (s.start, s.end, s.visual_hint)
        for s in transcript.sentences
        if s.visual_hint
    ]
    return hints if hints else None  # type: ignore[return-value]


def _attach_visual_hints(
    plan: ScenePlan, transcript: Transcript
) -> None:
    """
    For scenes[] transcripts, copy the visual_hint of the sentence whose
    midpoint is closest to each PlannedScene's midpoint.
    """
    if transcript.source != "scenes":
        return
    hint_sentences = [s for s in transcript.sentences if s.visual_hint]
    if not hint_sentences:
        return

    for scene in plan.scenes:
        scene_mid = (scene.start + scene.end) / 2
        best = min(
            hint_sentences,
            key=lambda s: abs((s.start + s.end) / 2 - scene_mid),
        )
        if scene.visual_hint is None:
            scene.visual_hint = best.visual_hint
            # Prepend the visual_hint as queries[0] if not already there
            if best.visual_hint and best.visual_hint not in scene.queries:
                scene.queries = [best.visual_hint] + scene.queries[:2]


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------

async def plan_scenes(
    transcript: Transcript,
    style: str | None = None,
    avoid: list[str] | None = None,
) -> ScenePlan:
    """
    Plan scenes for a completed Transcript.

    Args:
        transcript: Output of the input stage.
        style:      Optional visual style hint (e.g. "cinematic", "documentary").
        avoid:      Optional list of themes/terms to exclude from queries.

    Returns:
        ScenePlan with at least one PlannedScene covering 0..total_duration.
    """
    s = get_settings()
    avoid = avoid or []
    total_duration = transcript.duration

    if total_duration <= 0:
        logger.warning("scene_planner: zero duration — building single stub scene")
        return _single_scene_plan(transcript, style, avoid)

    # ── Build prompts ──────────────────────────────────────────────────
    visual_hints = _collect_visual_hints(transcript)

    system_prompt, user_prompt = build_prompts(
        language=transcript.language,
        total_duration=total_duration,
        transcript_text=transcript.text,
        sentence_list=_sentence_list_text(transcript.sentences),
        style=style or "neutral",
        avoid=avoid,
        wsd_words=s.SCENE_WSD_WORDS,
        min_dur=s.SCENE_MIN_DURATION_SECONDS,
        max_dur=s.SCENE_MAX_DURATION_SECONDS,
        visual_hints=visual_hints,
    )

    # ── LLM call ───────────────────────────────────────────────────────
    from app.pipeline.scene_planning.models import SCENE_PLAN_SCHEMA
    llm = get_llm_adapter()

    llm_data: dict[str, Any] | None = None
    llm_error: Exception | None = None

    try:
        llm_data = await llm.json_complete(
            system=system_prompt,
            prompt=user_prompt,
            schema=SCENE_PLAN_SCHEMA,
        )
    except (LLMSchemaError, Exception) as exc:
        llm_error = exc
        logger.warning(
            "scene_planner_llm_failed",
            extra={"error": str(exc)},
        )

    # ── Validate / repair ──────────────────────────────────────────────
    plan_warnings: list[str] = []

    if llm_data is not None:
        try:
            repaired, repair_warnings = validate_and_repair(
                llm_data,
                total_duration=total_duration,
                min_scene_duration=s.SCENE_MIN_DURATION_SECONDS,
                max_scene_duration=s.SCENE_MAX_DURATION_SECONDS,
            )
            plan_warnings.extend(repair_warnings)
            plan = ScenePlan.from_llm_dict(
                repaired,
                total_duration=total_duration,
                source_language=transcript.language,
                source="llm",
            )
            plan.warnings.extend(plan_warnings)
            _attach_visual_hints(plan, transcript)
            logger.info(
                "scene_plan_ready",
                extra={
                    "source": "llm",
                    "scenes": plan.scene_count,
                    "duration": total_duration,
                    "warnings": len(plan.warnings),
                },
            )
            return plan
        except ScenePlanValidationError as exc:
            logger.warning(
                "scene_plan_validation_failed",
                extra={"error": str(exc)},
            )
            plan_warnings.append(f"LLM plan invalid: {exc.message}")

    # ── Fallback ───────────────────────────────────────────────────────
    if not s.SCENE_PLAN_FALLBACK_ENABLED:
        error_msg = str(llm_error) if llm_error else "validation failed"
        raise ScenePlanValidationError(
            f"Scene planning failed and fallback is disabled: {error_msg}",
        )

    logger.info("scene_plan_using_fallback")
    plan = build_rule_based_plan(transcript, style=style, avoid=avoid)
    plan.warnings = plan_warnings + plan.warnings
    _attach_visual_hints(plan, transcript)
    return plan


def _single_scene_plan(
    transcript: Transcript,
    style: str | None,
    avoid: list[str],
) -> ScenePlan:
    """Return a single-scene plan for zero-duration or degenerate transcripts."""
    scene = PlannedScene(
        id=0,
        start=0.0,
        end=max(transcript.duration, 0.001),
        narration_text=transcript.text,
        meaning=transcript.text[:80] or "scene",
        queries=[transcript.text[:60] or "scene"],
        source="rule_based",
    )
    return ScenePlan(
        scenes=[scene],
        total_duration=transcript.duration,
        source_language=transcript.language,
        warnings=["single_scene: zero or near-zero duration"],
    )
