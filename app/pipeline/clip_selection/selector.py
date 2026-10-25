"""
Main clip selector — orchestrates the full pipeline for one scene or
all scenes in a ScenePlan.

Per-scene pipeline
------------------
For each query variant (queries[0] → queries[1] → queries[2]):
  1. Call FootageRouter.resolve(query) to get a Candidate.
  2. Build ClipCandidate, assess duration fit, skip if duration_ok=False.
  3. After collecting enough candidates (CLIP_SELECTION_MIN_CANDIDATES),
     or after exhausting all 3 queries, proceed to scoring.

Scoring
-------
  Stage 1 (cheap):  text_scorer.score_text() for all candidates.
  Stage 2 (VLM):    vlm_scorer.score_vlm() for the top-N candidates.
  Blend:            ranker.blend_scores() → sorted by final_score DESC.

Deduplication
-------------
  deduplicate_candidates() removes clips already used in this video and
  clips whose thumbnail perceptual-hash is too close to an already-chosen one.

Selection rule
--------------
  pick_best(threshold=CLIP_THRESHOLD_PIXABAY_VIDEO) → if found, rule = PIXABAY_VIDEO.
  else pick_best(threshold=CLIP_THRESHOLD_KEN_BURNS)  → rule = KEN_BURNS.
  else fall back to FootageRouter for a title card     → rule = TITLE_CARD.
  Nothing found at all                                 → rule = FORCED (None chosen).

Warnings
--------
  Every fallback inside the scene adds a warning string to SceneClipResult.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.adapters.footage_router import FootageRouter, FootageResult, get_footage_router
from app.adapters.footage_types import FallbackTier, FootageKind
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.clip_selection.dedup import UsedClipRegistry, deduplicate_candidates
from app.pipeline.clip_selection.duration_fit import assess_duration_fit
from app.pipeline.clip_selection.models import (
    ClipCandidate,
    RejectedClip,
    SceneClipResult,
    SelectionRule,
    VideoSelectionPlan,
)
from app.pipeline.clip_selection.ranker import apply_selection_rule, blend_scores, pick_best
from app.pipeline.clip_selection.text_scorer import score_text
from app.pipeline.clip_selection.vlm_scorer import score_vlm
from app.pipeline.scene_planning.models import PlannedScene, ScenePlan

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Scene-level selection
# ---------------------------------------------------------------------------

async def select_clip_for_scene(
    scene: PlannedScene,
    router: FootageRouter,
    used_registry: UsedClipRegistry,
    job_id: str | None = None,
) -> SceneClipResult:
    """
    Run the full clip-selection pipeline for one *scene*.

    Returns a SceneClipResult — always populated (even when only a title
    card is available).
    """
    s = get_settings()
    scene_duration = scene.duration
    narration_text = scene.narration_text
    result_warnings: list[str] = []
    rejected: list[RejectedClip] = []
    raw_candidates: list[ClipCandidate] = []
    thumbnail_bytes_map: dict[str, bytes] = {}

    # ── Gather candidates across query variants ────────────────────────
    for q_idx, query in enumerate(scene.queries[:3]):
        for _attempt in range(s.CLIP_SELECTION_CANDIDATES_PER_QUERY):
            try:
                footage_result: FootageResult = await router.resolve(
                    query, job_id=job_id
                )
            except Exception as exc:
                result_warnings.append(
                    f"FootageRouter failed for query '{query}': {exc}"
                )
                break

            # Propagate any FootageRouter warnings
            for fw in footage_result.warnings:
                result_warnings.append(fw.to_str())

            cand = footage_result.candidate
            if cand is None:
                break

            cc = ClipCandidate(
                candidate=cand,
                query=query,
                query_index=q_idx,
            )

            # Duration fit
            assess_duration_fit(cc, scene_duration)
            if not cc.duration_ok:
                rejected.append(RejectedClip(
                    candidate_id=cand.id,
                    source=cand.source,
                    reason=(
                        f"too short (clip={cand.duration:.1f}s, "
                        f"scene={scene_duration:.1f}s) — "
                        f"cannot slow/loop enough"
                    ),
                    score=0.0,
                ))
                break  # don't retry the same query

            raw_candidates.append(cc)
            # One candidate per query variant is enough for now
            break

        if len(raw_candidates) >= s.CLIP_SELECTION_MIN_CANDIDATES:
            break

    # ── Dedup ─────────────────────────────────────────────────────────
    deduped = deduplicate_candidates(raw_candidates, used_registry, thumbnail_bytes_map)
    for cc in raw_candidates:
        if cc.is_duplicate:
            rejected.append(RejectedClip(
                candidate_id=cc.candidate.id,
                source=cc.candidate.source,
                reason="duplicate (already used or visually similar)",
                score=0.0,
            ))

    # ── Stage-1: text scoring ──────────────────────────────────────────
    for cc in deduped:
        score_text(cc, scene, cc.query)

    # Sort cheaply before VLM
    deduped.sort(key=lambda c: c.text_score, reverse=True)

    # ── Stage-2: VLM scoring (top-N only) ─────────────────────────────
    top_n = deduped[: s.CLIP_SELECTION_VLM_TOP_N]
    top_n, thumbnail_bytes_map = await score_vlm(
        top_n, narration_text, thumbnail_bytes_map
    )
    # The rest keep vlm_score=0; blend will fall back to text-only
    rest = deduped[s.CLIP_SELECTION_VLM_TOP_N:]

    # ── Blend & rank ──────────────────────────────────────────────────
    all_scored = blend_scores(top_n) + blend_scores(rest)
    all_scored.sort(key=lambda c: c.final_score, reverse=True)

    # ── Pick best by threshold ────────────────────────────────────────
    chosen: ClipCandidate | None = None
    rule: SelectionRule

    chosen = pick_best(all_scored, s.CLIP_THRESHOLD_PIXABAY_VIDEO)
    if chosen:
        rule = apply_selection_rule(chosen)
    else:
        result_warnings.append(
            f"No candidate scored >= {s.CLIP_THRESHOLD_PIXABAY_VIDEO} "
            f"(PIXABAY_VIDEO threshold) — trying KEN_BURNS threshold"
        )
        chosen = pick_best(all_scored, s.CLIP_THRESHOLD_KEN_BURNS)
        if chosen:
            rule = apply_selection_rule(chosen)
        else:
            result_warnings.append(
                f"No candidate scored >= {s.CLIP_THRESHOLD_KEN_BURNS} "
                f"(KEN_BURNS threshold) — falling back to title card"
            )
            # Tier-4 fallback: ask FootageRouter for a title card explicitly
            try:
                title_result = await router._try_title_card(
                    scene.primary_query, _make_dummy_footage_result(), job_id
                )
                if title_result is not None:
                    chosen = ClipCandidate(
                        candidate=title_result,
                        query=scene.primary_query,
                        query_index=2,
                        text_score=1.0,
                        final_score=1.0,
                        duration_ok=True,
                        final_score_reason="title_card_fallback",
                    )
                    rule = SelectionRule.TITLE_CARD
                else:
                    rule = SelectionRule.FORCED
            except Exception as exc:
                result_warnings.append(f"Title card generation failed: {exc}")
                rule = SelectionRule.FORCED

    # Record rejected candidates (those not chosen)
    if chosen:
        used_registry.mark_used(chosen.candidate.id)
    for cc in all_scored:
        if chosen is None or cc.candidate.id != chosen.candidate.id:
            rejected.append(RejectedClip(
                candidate_id=cc.candidate.id,
                source=cc.candidate.source,
                reason=f"outscored (final_score={cc.final_score:.2f})",
                score=cc.final_score,
            ))

    logger.info(
        "scene_clip_selected",
        extra={
            "scene_id": scene.id,
            "chosen_id": chosen.candidate.id if chosen else None,
            "rule": rule.value,
            "score": chosen.final_score if chosen else 0,
            "candidates_evaluated": len(all_scored),
            "warnings": len(result_warnings),
        },
    )

    return SceneClipResult(
        scene_id=scene.id,
        scene_start=scene.start,
        scene_end=scene.end,
        scene_duration=scene_duration,
        chosen=chosen,
        selection_rule=rule,
        rejected=rejected,
        warnings=result_warnings,
    )


def _make_dummy_footage_result() -> Any:
    """Create a FootageResult placeholder for title-card calls."""
    from app.adapters.footage_router import FootageResult
    return FootageResult(candidate=None, tier=FallbackTier.TITLE_CARD)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plan-level selection
# ---------------------------------------------------------------------------

async def select_clips_for_plan(
    plan: ScenePlan,
    job_id: str | None = None,
) -> VideoSelectionPlan:
    """
    Run clip selection for every scene in *plan* sequentially.

    Scenes are processed one at a time (not concurrently) to preserve the
    deduplication invariant (each clip used at most once per video).

    Args:
        plan:    The ScenePlan produced by the scene planning stage.
        job_id:  Optional job UUID for storage paths.

    Returns:
        VideoSelectionPlan with one SceneClipResult per scene.
    """
    router = get_footage_router()
    used_registry = UsedClipRegistry()
    plan_warnings: list[str] = list(plan.warnings)
    scene_results: list[SceneClipResult] = []

    for scene in plan.scenes:
        result = await select_clip_for_scene(
            scene, router, used_registry, job_id=job_id
        )
        scene_results.append(result)

    video_plan = VideoSelectionPlan(
        job_id=job_id or "",
        scene_results=scene_results,
        warnings=plan_warnings,
    )

    logger.info(
        "video_selection_plan_complete",
        extra={
            "job_id": job_id,
            "scenes": video_plan.scene_count,
            "chosen": video_plan.chosen_count,
        },
    )
    return video_plan
