"""
Deterministic validator for LLM-produced scene plans.

validate_and_repair()
  1. Checks that the plan matches SCENE_PLAN_SCHEMA.
  2. Repairs minor structural issues without another LLM call:
       - sorts scenes by start time
       - closes tiny gaps / overlaps within REPAIR_TOLERANCE_SECONDS
       - extends the final scene to total_duration if it falls short
       - clamps first scene start to 0.0
       - fills missing queries from meaning/narration_text
  3. Returns (repaired_plan_dict, list_of_warnings).
  4. Raises ScenePlanValidationError if the plan is fatally broken and
     cannot be repaired.

Gap / overlap invariant
  After repair: scenes[i].end == scenes[i+1].start for all i,
  scenes[0].start == 0.0, scenes[-1].end == total_duration.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import BrollError
from app.core.logging import get_logger
from app.pipeline.scene_planning.models import SCENE_PLAN_SCHEMA

logger = get_logger(__name__)

# Gaps/overlaps within this tolerance are auto-repaired (seconds)
REPAIR_TOLERANCE_SECONDS = 0.5


class ScenePlanValidationError(BrollError):
    """LLM scene plan is fatally malformed and cannot be repaired."""
    code = "SCENE_PLAN_VALIDATION_ERROR"
    http_status = 422
    retryable = True


def _validate_schema(data: Any) -> list[str]:
    """Return JSON Schema validation errors (empty list = valid)."""
    try:
        import jsonschema  # type: ignore[import]
        v = jsonschema.Draft7Validator(SCENE_PLAN_SCHEMA)
        return [e.message for e in sorted(v.iter_errors(data), key=lambda e: list(e.path))]
    except ImportError:
        if not isinstance(data, dict) or "scenes" not in data:
            return ["Response must be a JSON object with a 'scenes' array"]
        return []


def _validate_timing_constraints(
    scenes: list[dict[str, Any]],
    total_duration: float,
    min_dur: float,
    max_dur: float,
) -> list[str]:
    """Check timing correctness beyond JSON schema."""
    errors: list[str] = []
    for i, sc in enumerate(scenes):
        start = float(sc.get("start", 0))
        end = float(sc.get("end", 0))
        dur = end - start
        if start >= end:
            errors.append(f"Scene {i}: start ({start}) >= end ({end})")
        if dur < min_dur - REPAIR_TOLERANCE_SECONDS:
            errors.append(f"Scene {i}: duration {dur:.2f}s < min {min_dur}s")
        if dur > max_dur + REPAIR_TOLERANCE_SECONDS:
            errors.append(f"Scene {i}: duration {dur:.2f}s > max {max_dur}s")
    return errors


def validate_and_repair(
    data: dict[str, Any],
    total_duration: float,
    min_scene_duration: float,
    max_scene_duration: float,
) -> tuple[dict[str, Any], list[str]]:
    """
    Validate and repair a raw LLM scene plan dict.

    Returns (repaired_dict, warnings).
    Raises ScenePlanValidationError on fatal errors.
    """
    warnings: list[str] = []

    # ── 1. JSON schema check ───────────────────────────────────────────
    schema_errors = _validate_schema(data)
    if schema_errors:
        raise ScenePlanValidationError(
            "Scene plan failed JSON schema validation",
            details={"errors": schema_errors[:5]},
        )

    scenes: list[dict[str, Any]] = [dict(sc) for sc in data["scenes"]]

    if not scenes:
        raise ScenePlanValidationError("Scene plan contains zero scenes")

    # ── 2. Sort by start time ──────────────────────────────────────────
    scenes.sort(key=lambda s: float(s.get("start", 0)))

    # ── 3. Clamp first start to 0.0 ───────────────────────────────────
    first_start = float(scenes[0]["start"])
    if abs(first_start) > REPAIR_TOLERANCE_SECONDS:
        raise ScenePlanValidationError(
            f"First scene starts at {first_start:.2f}s, not near 0.0"
        )
    if first_start != 0.0:
        warnings.append(f"First scene start clamped from {first_start} to 0.0")
        scenes[0]["start"] = 0.0

    # ── 4. Close gaps and overlaps ────────────────────────────────────
    for i in range(len(scenes) - 1):
        curr_end = float(scenes[i]["end"])
        next_start = float(scenes[i + 1]["start"])
        gap = next_start - curr_end
        if abs(gap) <= REPAIR_TOLERANCE_SECONDS and gap != 0.0:
            # Snap: set next_start to curr_end
            scenes[i + 1]["start"] = curr_end
            warnings.append(
                f"Gap/overlap of {gap:.3f}s between scenes {i} and {i+1} repaired"
            )
        elif gap > REPAIR_TOLERANCE_SECONDS:
            raise ScenePlanValidationError(
                f"Scene {i} ends at {curr_end:.2f}s but scene {i+1} "
                f"starts at {next_start:.2f}s — gap of {gap:.2f}s is too large",
                details={"scene_index": i, "gap_seconds": round(gap, 3)},
            )
        elif gap < -REPAIR_TOLERANCE_SECONDS:
            raise ScenePlanValidationError(
                f"Scene {i} ends at {curr_end:.2f}s but scene {i+1} "
                f"starts at {next_start:.2f}s — overlap of {-gap:.2f}s is too large",
                details={"scene_index": i, "overlap_seconds": round(-gap, 3)},
            )

    # ── 5. Extend / clamp final scene to total_duration ───────────────
    last_end = float(scenes[-1]["end"])
    shortfall = total_duration - last_end
    if abs(shortfall) <= REPAIR_TOLERANCE_SECONDS and last_end != total_duration:
        warnings.append(
            f"Final scene end extended from {last_end:.3f}s to {total_duration:.3f}s"
        )
        scenes[-1]["end"] = total_duration
    elif shortfall > REPAIR_TOLERANCE_SECONDS:
        raise ScenePlanValidationError(
            f"Final scene ends at {last_end:.2f}s but audio is {total_duration:.2f}s — "
            f"unplanned {shortfall:.2f}s at the end",
            details={"last_end": last_end, "total_duration": total_duration},
        )
    elif shortfall < -REPAIR_TOLERANCE_SECONDS:
        raise ScenePlanValidationError(
            f"Final scene ends at {last_end:.2f}s, past total_duration {total_duration:.2f}s",
            details={"last_end": last_end, "total_duration": total_duration},
        )

    # ── 6. Timing constraint check ────────────────────────────────────
    timing_errors = _validate_timing_constraints(
        scenes, total_duration, min_scene_duration, max_scene_duration
    )
    if timing_errors:
        raise ScenePlanValidationError(
            "Scene timing constraints violated",
            details={"errors": timing_errors[:5]},
        )

    # ── 7. Fill missing queries ────────────────────────────────────────
    for i, sc in enumerate(scenes):
        queries = [q for q in sc.get("queries", []) if q and str(q).strip()]
        if not queries:
            fallback = sc.get("meaning") or sc.get("narration_text", f"scene {i}")
            scenes[i]["queries"] = [str(fallback)[:80]]
            warnings.append(f"Scene {i}: missing queries filled from meaning/narration")

    return {"scenes": scenes}, warnings
