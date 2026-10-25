"""
timeline.json — per-scene record of what clip was chosen, why, and what
was rejected.

JSON Schema validates the output shape in tests.
"""

from __future__ import annotations

from typing import Any

from app.adapters.footage_types import Candidate
from app.core.logging import get_logger
from app.pipeline.clip_selection.models import (
    ClipCandidate,
    RejectedClip,
    SceneClipResult,
    SelectionRule,
)
from app.pipeline.scene_planning.models import PlannedScene, ScenePlan

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON Schema for timeline.json
# ---------------------------------------------------------------------------

TIMELINE_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["job_id", "total_duration", "scenes"],
    "additionalProperties": False,
    "properties": {
        "job_id": {"type": "string"},
        "total_duration": {"type": "number", "minimum": 0},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "index", "text", "start", "end", "clip",
                    "reason_chosen", "relevance_score", "vlm_score",
                    "source_rule_fired", "alternatives_rejected", "warnings",
                ],
                "additionalProperties": False,
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "text": {"type": "string"},
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "minimum": 0},
                    "clip": {
                        "type": ["object", "null"],
                        "required": [
                            "source", "id", "url", "page_url",
                            "license", "commercial_use", "attribution",
                            "is_ai_generated", "duration_used",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "source": {"type": "string"},
                            "id": {"type": "string"},
                            "url": {"type": "string"},
                            "page_url": {"type": "string"},
                            "license": {"type": "string"},
                            "commercial_use": {"type": "boolean"},
                            "attribution": {"type": "string"},
                            "is_ai_generated": {"type": "boolean"},
                            "duration_used": {"type": "number", "minimum": 0},
                        },
                    },
                    "reason_chosen": {"type": "string"},
                    "relevance_score": {"type": "number", "minimum": 0, "maximum": 5},
                    "vlm_score": {"type": "number", "minimum": 0, "maximum": 5},
                    "source_rule_fired": {"type": "string"},
                    "alternatives_rejected": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["candidate_id", "reason"],
                            "additionalProperties": False,
                            "properties": {
                                "candidate_id": {"type": "string"},
                                "reason": {"type": "string"},
                                "score": {"type": "number"},
                            },
                        },
                    },
                    "warnings": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _clip_to_timeline_dict(
    chosen: ClipCandidate | None,
    scene_duration: float,
) -> dict[str, Any] | None:
    """Convert the chosen ClipCandidate to the timeline clip dict."""
    if chosen is None:
        return None
    c: Candidate = chosen.candidate
    return {
        "source": c.source,
        "id": c.id,
        "url": c.url,
        "page_url": c.page_url,
        "license": c.license,
        "commercial_use": c.commercial_use,
        "attribution": c.attribution,
        "is_ai_generated": c.is_ai_generated,
        "duration_used": round(scene_duration, 3),
    }


def _rejected_to_timeline(rejected: list[RejectedClip]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": r.candidate_id,
            "reason": r.reason,
            "score": round(r.score, 3),
        }
        for r in rejected
    ]


def build_timeline(
    job_id: str,
    plan: ScenePlan,
    selection: Any,  # VideoSelectionPlan
) -> dict[str, Any]:
    """
    Build the timeline.json dict from a ScenePlan and VideoSelectionPlan.

    Args:
        job_id:     Job UUID string.
        plan:       ScenePlan from the scene planning stage.
        selection:  VideoSelectionPlan from the clip selection stage.

    Returns:
        Dict conforming to TIMELINE_SCHEMA.
    """
    scenes_out: list[dict[str, Any]] = []

    # Build a lookup: scene_id → SceneClipResult
    scene_results: dict[int, SceneClipResult] = {
        sr.scene_id: sr for sr in selection.scene_results
    }

    for scene in plan.scenes:
        sr = scene_results.get(scene.id)
        if sr is None:
            scenes_out.append({
                "index": scene.id,
                "text": scene.narration_text,
                "start": round(scene.start, 3),
                "end": round(scene.end, 3),
                "clip": None,
                "reason_chosen": "no clip selection result for this scene",
                "relevance_score": 0.0,
                "vlm_score": 0.0,
                "source_rule_fired": "FORCED",
                "alternatives_rejected": [],
                "warnings": ["scene missing from selection plan"],
            })
            continue

        chosen = sr.chosen
        clip_dict = _clip_to_timeline_dict(chosen, sr.scene_duration)

        reason = ""
        if chosen is not None:
            reason = chosen.final_score_reason or (
                f"text={chosen.text_score:.2f}, vlm={chosen.vlm_score:.2f}"
            )
        else:
            reason = "no candidate met any threshold"

        scenes_out.append({
            "index": scene.id,
            "text": scene.narration_text,
            "start": round(scene.start, 3),
            "end": round(scene.end, 3),
            "clip": clip_dict,
            "reason_chosen": reason,
            "relevance_score": round(sr.final_score, 3),
            "vlm_score": round(chosen.vlm_score, 3) if chosen else 0.0,
            "source_rule_fired": sr.selection_rule.value,
            "alternatives_rejected": _rejected_to_timeline(sr.rejected),
            "warnings": list(sr.warnings),
        })

    timeline = {
        "job_id": job_id,
        "total_duration": round(plan.total_duration, 3),
        "scenes": scenes_out,
    }

    logger.info(
        "timeline_built",
        extra={"job_id": job_id, "scenes": len(scenes_out)},
    )
    return timeline
