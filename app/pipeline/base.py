"""
Pipeline abstractions.

A Pipeline is a sequence of Steps. Each Step receives a PipelineContext,
performs its work, and returns a (possibly mutated) context to the next step.

     PipelineContext  ──►  Step1  ──►  Step2  ──►  StepN  ──►  result
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import PipelineError
from app.core.logging import get_logger, set_job_id

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Context — shared state passed through all steps
# ---------------------------------------------------------------------------
@dataclass
class PipelineContext:
    job_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def set(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)


# ---------------------------------------------------------------------------
# Step base class
# ---------------------------------------------------------------------------
class Step(ABC):
    """Abstract base for a single pipeline step."""

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Run this step and return the (mutated) context."""
        ...


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
class Pipeline:
    """
    Runs a list of Steps sequentially, sharing a PipelineContext.

    Usage::

        pipeline = Pipeline([StepA(), StepB(), StepC()])
        final_ctx = await pipeline.run(ctx)
    """

    def __init__(self, steps: list[Step]) -> None:
        self._steps = steps

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        set_job_id(ctx.job_id)
        logger.info("pipeline_start", extra={"job_id": ctx.job_id, "steps": len(self._steps)})

        for step in self._steps:
            t0 = time.perf_counter()
            logger.info("pipeline_step_start", extra={"step": step.name})
            try:
                ctx = await step.execute(ctx)
            except PipelineError:
                raise
            except Exception as exc:
                duration_ms = round((time.perf_counter() - t0) * 1000, 2)
                logger.error(
                    "pipeline_step_failed",
                    extra={"step": step.name, "duration_ms": duration_ms, "error": str(exc)},
                )
                raise PipelineError(
                    f"Step '{step.name}' failed: {exc}",
                    details={"step": step.name, "error": str(exc)},
                ) from exc

            duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info("pipeline_step_done", extra={"step": step.name, "duration_ms": duration_ms})

        logger.info("pipeline_complete", extra={"job_id": ctx.job_id})
        return ctx
