"""
ClipSelectionStep — pipeline step wrapping select_clips_for_plan().

Reads from PipelineContext:
    ctx.get("scene_plan_obj")   — ScenePlan (required)

Writes to PipelineContext:
    ctx.set("video_selection_plan", plan.to_dict())
    ctx.set("video_selection_plan_obj", plan)
    ctx.set("scenes_chosen", plan.chosen_count)
"""

from __future__ import annotations

from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.clip_selection.selector import select_clips_for_plan

logger = get_logger(__name__)


class ClipSelectionStep(Step):
    """Select clips for every scene in the ScenePlan."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        scene_plan = ctx.get("scene_plan_obj")
        if scene_plan is None:
            raise ValidationError(
                "ClipSelectionStep requires 'scene_plan_obj' on context — "
                "run ScenePlanningStep first"
            )

        job_id: str | None = ctx.job_id or None

        logger.info(
            "clip_selection_start",
            extra={"scenes": scene_plan.scene_count, "job_id": job_id},
        )

        try:
            video_plan = await select_clips_for_plan(scene_plan, job_id=job_id)
        except (PipelineError, ValidationError):
            raise
        except Exception as exc:
            raise PipelineError(
                f"Clip selection failed: {exc}",
                details={"error": str(exc)},
            ) from exc

        ctx.set("video_selection_plan", video_plan.to_dict())
        ctx.set("video_selection_plan_obj", video_plan)
        ctx.set("scenes_chosen", video_plan.chosen_count)

        logger.info(
            "clip_selection_complete",
            extra={
                "scenes": video_plan.scene_count,
                "chosen": video_plan.chosen_count,
                "warnings": len(video_plan.warnings),
            },
        )
        return ctx
