"""
OutputStep — pipeline step that builds timeline.json, manifest.json,
and runs the licence gate.

Reads from PipelineContext:
    ctx.get("scene_plan_obj")            — ScenePlan
    ctx.get("video_selection_plan_obj")  — VideoSelectionPlan

Writes:
    ctx.set("timeline", dict)
    ctx.set("manifest", dict)
    ctx.set("licence_gate_passed", bool)
"""

from __future__ import annotations

from app.core.errors import LicenceGateError, PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.output.licence_gate import check_licence_gate
from app.pipeline.output.manifest import build_manifest
from app.pipeline.output.timeline import build_timeline

logger = get_logger(__name__)


class OutputStep(Step):
    """Generate timeline.json, manifest.json, and enforce the licence gate."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        plan = ctx.get("scene_plan_obj")
        selection = ctx.get("video_selection_plan_obj")

        if plan is None:
            raise ValidationError("OutputStep requires 'scene_plan_obj'")
        if selection is None:
            raise ValidationError("OutputStep requires 'video_selection_plan_obj'")

        # ── Licence gate ────────────────────────────────────────────────
        try:
            check_licence_gate(selection)
            ctx.set("licence_gate_passed", True)
        except LicenceGateError:
            # Re-raise — the job must fail
            raise

        # ── Build timeline ───────────────────────────────────────────────
        try:
            timeline = build_timeline(ctx.job_id, plan, selection)
        except Exception as exc:
            raise PipelineError(
                f"Failed to build timeline: {exc}",
                details={"error": str(exc)},
            ) from exc

        # ── Build manifest ───────────────────────────────────────────────
        all_warnings: list[str] = list(plan.warnings) + list(selection.warnings)
        try:
            manifest = build_manifest(
                job_id=ctx.job_id,
                timeline=timeline,
                warnings=all_warnings,
                aspect_ratios=ctx.payload.get("aspect_ratios"),
            )
        except Exception as exc:
            raise PipelineError(
                f"Failed to build manifest: {exc}",
                details={"error": str(exc)},
            ) from exc

        ctx.set("timeline", timeline)
        ctx.set("manifest", manifest)

        logger.info(
            "output_generated",
            extra={
                "job_id": ctx.job_id,
                "scenes": len(timeline.get("scenes", [])),
                "licence_gate": True,
            },
        )
        return ctx
