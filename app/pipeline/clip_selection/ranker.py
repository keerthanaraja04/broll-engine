"""
Ranker — blends text + VLM scores and applies source-selection rules.

blend_scores()
    Computes final_score = w_text * text_score + w_vlm * vlm_score.
    When vlm_score == 0 (VLM disabled or thumbnail missing), weight is
    shifted entirely onto text_score.

apply_selection_rule()
    Checks the final_score of the best candidate against the configured
    thresholds and returns the SelectionRule that fired.

    Threshold rules (measurable, from config):
        score >= CLIP_THRESHOLD_PIXABAY_VIDEO  → PIXABAY_VIDEO
        score >= CLIP_THRESHOLD_KEN_BURNS      → KEN_BURNS
        score  < CLIP_THRESHOLD_KEN_BURNS      → TITLE_CARD
"""

from __future__ import annotations

from app.adapters.footage_types import FallbackTier
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.clip_selection.models import ClipCandidate, SelectionRule

logger = get_logger(__name__)


def blend_scores(candidates: list[ClipCandidate]) -> list[ClipCandidate]:
    """
    Compute final_score for each candidate and sort by final_score DESC.

    Modifies each candidate in-place and returns the sorted list.
    """
    s = get_settings()
    w_text = s.CLIP_SELECTION_WEIGHT_TEXT
    w_vlm  = s.CLIP_SELECTION_WEIGHT_VLM

    for cc in candidates:
        if cc.vlm_score == 0.0:
            # VLM not available: use text score only, clamped to [1.0, 5.0]
            cc.final_score = round(max(1.0, min(5.0, cc.text_score)), 3)
            cc.final_score_reason = f"text_only={cc.text_score:.2f} (no VLM)"
        else:
            blended = w_text * cc.text_score + w_vlm * cc.vlm_score
            cc.final_score = round(max(1.0, min(5.0, blended)), 3)
            cc.final_score_reason = (
                f"text={cc.text_score:.2f}×{w_text} + "
                f"vlm={cc.vlm_score:.2f}×{w_vlm} = {cc.final_score:.2f}"
            )

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates


def apply_selection_rule(
    best_candidate: ClipCandidate | None,
) -> SelectionRule:
    """
    Return the SelectionRule that corresponds to the best candidate.

    Args:
        best_candidate: The highest-scored surviving candidate, or None.

    Returns:
        SelectionRule enum value indicating which threshold fired.
    """
    if best_candidate is None:
        return SelectionRule.FORCED

    s = get_settings()
    score = best_candidate.final_score
    tier  = best_candidate.candidate.fallback_tier

    # Card/ken-burns produced by FootageRouter always come from low tiers;
    # use the tier directly to assign the rule when the candidate is already
    # a generated asset.
    if tier == FallbackTier.TITLE_CARD:
        return SelectionRule.TITLE_CARD

    # Relevance score >= threshold (4.0): take video from Pixabay
    if score >= s.CLIP_THRESHOLD_PIXABAY_VIDEO and tier == FallbackTier.PIXABAY_VIDEO:
        return SelectionRule.PIXABAY_VIDEO

    # Relevance score >= KEN_BURNS threshold (3.0): generate / route to Ken Burns motion picture
    if score >= s.CLIP_THRESHOLD_KEN_BURNS and tier in (
        FallbackTier.PIXABAY_VIDEO, FallbackTier.KEN_BURNS, FallbackTier.CACHED_CLIP
    ):
        return SelectionRule.KEN_BURNS

    return SelectionRule.TITLE_CARD


def pick_best(
    candidates: list[ClipCandidate],
    threshold: float | None = None,
    score_threshold: float | None = None,
) -> ClipCandidate | None:
    """
    Return the first candidate with final_score >= threshold.
    Returns None if no candidate meets the threshold.
    """
    t = threshold if threshold is not None else (score_threshold if score_threshold is not None else 0.0)
    for cc in candidates:
        if cc.final_score >= t and cc.duration_ok and not cc.is_duplicate:
            return cc
    return None
