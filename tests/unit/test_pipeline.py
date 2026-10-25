"""
Unit tests for the pipeline runner and step abstractions.
"""

from __future__ import annotations

import pytest

from app.pipeline.base import Pipeline, PipelineContext, Step
from app.core.errors import PipelineError


class DoubleStep(Step):
    """Multiplies artifacts['value'] by 2."""

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        ctx.set("value", ctx.get("value", 0) * 2)
        return ctx


class AddOneStep(Step):
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        ctx.set("value", ctx.get("value", 0) + 1)
        return ctx


class ExplodingStep(Step):
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        raise RuntimeError("boom")


class TestPipelineContext:
    def test_set_get(self) -> None:
        ctx = PipelineContext(job_id="j1")
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self) -> None:
        ctx = PipelineContext(job_id="j1")
        assert ctx.get("missing", 42) == 42

    def test_payload_access(self) -> None:
        ctx = PipelineContext(job_id="j1", payload={"audio_object": "audio.mp3"})
        assert ctx.payload["audio_object"] == "audio.mp3"


class TestPipelineRunner:
    @pytest.mark.asyncio
    async def test_steps_run_in_order(self) -> None:
        ctx = PipelineContext(job_id="j1")
        ctx.set("value", 3)
        pipeline = Pipeline([AddOneStep(), DoubleStep()])  # (3+1)*2 = 8
        result = await pipeline.run(ctx)
        assert result.get("value") == 8

    @pytest.mark.asyncio
    async def test_empty_pipeline(self) -> None:
        ctx = PipelineContext(job_id="j1")
        ctx.set("value", 99)
        pipeline = Pipeline([])
        result = await pipeline.run(ctx)
        assert result.get("value") == 99

    @pytest.mark.asyncio
    async def test_step_failure_raises_pipeline_error(self) -> None:
        ctx = PipelineContext(job_id="j1")
        pipeline = Pipeline([AddOneStep(), ExplodingStep()])
        with pytest.raises(PipelineError) as exc_info:
            await pipeline.run(ctx)
        assert "ExplodingStep" in str(exc_info.value)

    def test_step_name_is_class_name(self) -> None:
        assert AddOneStep().name == "AddOneStep"
        assert DoubleStep().name == "DoubleStep"
