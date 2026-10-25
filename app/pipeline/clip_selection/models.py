"""
Data models for the clip selection stage.

ClipCandidate    — a scored candidate clip for one scene.
RejectedClip     — a candidate that was rejected, with the reason.
SceneClipResult  — the chosen clip + metadata for one scene.
SelectionRule    — which threshold rule fired for the chosen clip.
VideoSelectionPlan — the full clip-selection output for a job.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from app.adapters.footage_types import Candidate, FallbackTier


# ---------------------------------------------------------------------------
# Selection rule (measurable threshold that fired)
# ---------------------------------------------------------------------------

class SelectionRule(str, enum.Enum):
    """Which source-selection threshold rule decided the outcome."""
    PIXABAY_VIDEO = "pixabay_video_score_gte_threshold"    # tier 1, score >= 3.8
    KEN_BURNS     = "ken_burns_score_gte_threshold"        # tier 2, score >= 3.0
    TITLE_CARD    = "title_card_score_below_threshold"     # tier 4, score < 3.0
    FORCED        = "forced_by_exhaustion"                 # no scored candidates at all


# ---------------------------------------------------------------------------
# Per-candidate score record
# ---------------------------------------------------------------------------

@dataclass
class ClipCandidate:
    """
    One scored clip candidate for a scene.

    All scores are on a 1–5 scale before blending.
    ``final_score`` is the weighted combination used for ranking.
    """
    candidate: Candidate
    query: str                    # which query produced this candidate
    query_index: int              # 0 = most literal, 1 = thematic, 2 = abstract

    # Stage-1: cheap text / tag score
    text_score: float = 0.0       # 1–5
    text_score_reason: str = ""

    # Stage-2: VLM score (0 if VLM disabled / skipped)
    vlm_score: float = 0.0        # 1–5 (0 means not scored)
    vlm_reason: str = ""
    vlm_flags: list[str] = field(default_factory=list)

    # Blended
    final_score: float = 0.0      # weighted mix of text + vlm
    final_score_reason: str = ""

    # Duration fit
    needs_slow: bool = False       # True if clip must be slowed to fill scene
    needs_loop: int = 0            # >0: number of loops required
    duration_ok: bool = True       # False if clip is too short even with slow/loop

    # Deduplication
    phash: str | None = None       # perceptual hash hex string of thumbnail
    is_duplicate: bool = False     # flagged if hash distance ≤ CLIP_PHASH_MAX_DISTANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.id,
            "source": self.candidate.source,
            "kind": self.candidate.kind.value,
            "fallback_tier": self.candidate.fallback_tier.value,
            "query": self.query,
            "query_index": self.query_index,
            "text_score": round(self.text_score, 3),
            "vlm_score": round(self.vlm_score, 3),
            "final_score": round(self.final_score, 3),
            "final_score_reason": self.final_score_reason,
            "needs_slow": self.needs_slow,
            "needs_loop": self.needs_loop,
            "duration_ok": self.duration_ok,
            "phash": self.phash,
            "is_duplicate": self.is_duplicate,
        }


# ---------------------------------------------------------------------------
# Rejection record
# ---------------------------------------------------------------------------

@dataclass
class RejectedClip:
    """Records why a candidate was not chosen."""
    candidate_id: str
    source: str
    reason: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source": self.source,
            "reason": self.reason,
            "score": round(self.score, 3),
        }


# ---------------------------------------------------------------------------
# Per-scene result
# ---------------------------------------------------------------------------

@dataclass
class SceneClipResult:
    """
    The complete clip-selection outcome for one PlannedScene.

    ``chosen`` is None only in catastrophic failures (title card also failed).
    """
    scene_id: int
    scene_start: float
    scene_end: float
    scene_duration: float

    chosen: ClipCandidate | None
    selection_rule: SelectionRule
    rejected: list[RejectedClip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def chosen_id(self) -> str | None:
        return self.chosen.candidate.id if self.chosen else None

    @property
    def final_score(self) -> float:
        return self.chosen.final_score if self.chosen else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_start": self.scene_start,
            "scene_end": self.scene_end,
            "scene_duration": self.scene_duration,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "selection_rule": self.selection_rule.value,
            "rejected": [r.to_dict() for r in self.rejected],
            "warnings": self.warnings,
            "final_score": round(self.final_score, 3),
        }


# ---------------------------------------------------------------------------
# Full job plan
# ---------------------------------------------------------------------------

@dataclass
class VideoSelectionPlan:
    """Clip-selection results for every scene in a job."""
    job_id: str
    scene_results: list[SceneClipResult]
    warnings: list[str] = field(default_factory=list)

    @property
    def scene_count(self) -> int:
        return len(self.scene_results)

    @property
    def chosen_count(self) -> int:
        return sum(1 for r in self.scene_results if r.chosen is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "scene_count": self.scene_count,
            "chosen_count": self.chosen_count,
            "warnings": self.warnings,
            "scenes": [r.to_dict() for r in self.scene_results],
        }
