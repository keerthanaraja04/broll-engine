"""
SafetyStep — pipeline step that enforces safety rules before clip selection.

Runs after ScenePlanningStep and before ClipSelectionStep.

Reads from PipelineContext:
    ctx.get("scene_plan_obj")  — ScenePlan
    ctx.payload.get("avoid")   — list of terms to avoid

Writes to PipelineContext:
    ctx.set("safety_results", [dict, ...])  — per-scene safety results
    ctx.set("safety_results_obj", [SafetyResult, ...])

The clip selection step should use safety_results_obj to sanitize queries
before passing them to FootageRouter.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.safety.safety_rules import check_scene_safety, SafetyResult

logger = get_logger(__name__)


class SafetyStep(Step):
    """Enforce safety rules on every scene's queries."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        plan = ctx.get("scene_plan_obj")
        if plan is None:
            raise ValidationError(
                "SafetyStep requires 'scene_plan_obj' — run ScenePlanningStep first"
            )

        avoid: list[str] = ctx.payload.get("avoid") or []

        results: list[SafetyResult] = []
        for scene in plan.scenes:
            result = check_scene_safety(scene, avoid=avoid)
            results.append(result)
            # Update the scene's queries with the sanitized versions
            scene.queries = result.queries
            # Add safety warnings to the scene's warnings
            scene.warnings.extend(result.warnings)

        ctx.set("safety_results", [r.__dict__ for r in results])
        ctx.set("safety_results_obj", results)

        total_warnings = sum(len(r.warnings) for r in results)
        cards_needed = sum(1 for r in results if r.use_data_card)

        logger.info(
            "safety_step_complete",
            extra={
                "scenes": len(results),
                "total_warnings": total_warnings,
                "data_cards_needed": cards_needed,
            },
        )
        return ctx
