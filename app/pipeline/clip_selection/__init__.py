"""
Clip selection package.

For every PlannedScene this package finds, scores, deduplicates, and
selects the best available clip via a two-stage ranking pipeline.
"""

from app.pipeline.clip_selection.models import (
    ClipCandidate,
    RejectedClip,
    SceneClipResult,
    SelectionRule,
    VideoSelectionPlan,
)
from app.pipeline.clip_selection.selector import select_clips_for_plan
from app.pipeline.clip_selection.clip_selection_step import ClipSelectionStep

__all__ = [
    "ClipCandidate",
    "RejectedClip",
    "SceneClipResult",
    "SelectionRule",
    "VideoSelectionPlan",
    "select_clips_for_plan",
    "ClipSelectionStep",
]
