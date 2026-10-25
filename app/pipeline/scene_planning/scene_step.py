"""
ScenePlanningStep — pipeline step wrapping the scene planner.

Reads from PipelineContext:
    ctx.get("transcript_obj")   — Transcript (required)
    ctx.payload.get("style")    — optional visual style
    ctx.payload.get("avoid")    — optional avoid list

Writes to PipelineContext:
    ctx.set("scene_plan", plan.to_dict())
    ctx.set("scene_plan_obj", plan)
    ctx.set("scene_count", plan.scene_count)
"""

from __future__ import annotations

from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.scene_planning.planner import plan_scenes

logger = get_logger(__name__)


class ScenePlanningStep(Step):
    """Convert the Transcript into a ScenePlan using LLM + fallback."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        transcript = ctx.get("transcript_obj")
        if transcript is None:
            raise ValidationError(
                "ScenePlanningStep requires 'transcript_obj' on context — "
                "run InputStageStep first"
            )

        style: str | None = ctx.payload.get("style")
        avoid: list[str] = ctx.payload.get("avoid") or []

        logger.info(
            "scene_planning_start",
            extra={
                "source": transcript.source,
                "language": transcript.language,
                "duration": transcript.duration,
                "sentences": len(transcript.sentences),
            },
        )

        try:
            plan = await plan_scenes(transcript, style=style, avoid=avoid)
        except (PipelineError, ValidationError):
            raise
        except Exception as exc:
            raise PipelineError(
                f"Scene planning failed: {exc}",
                details={"error": str(exc)},
            ) from exc

        ctx.set("scene_plan", plan.to_dict())
        ctx.set("scene_plan_obj", plan)
        ctx.set("scene_count", plan.scene_count)

        logger.info(
            "scene_planning_complete",
            extra={
                "scenes": plan.scene_count,
                "source": plan.scenes[0].source if plan.scenes else "none",
                "warnings": len(plan.warnings),
            },
        )
        return ctx
