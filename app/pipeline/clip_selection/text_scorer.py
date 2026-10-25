"""
Stage-1 text scorer — cheap, no API calls.

Scores a ClipCandidate against a scene's narration text using:
  1. Tag overlap between candidate tags and query/narration tokens.
  2. A lightweight keyword-density heuristic.
  3. Source tier bonus (video > image > card).
  4. Sensitivity penalty: if the scene is_sensitive and the clip has
     known risky tags, the score is penalised.

Output: text_score in [1, 5], text_score_reason string.

For the real LLM batch scoring path (if we want it), replace or augment
this function; the interface is identical.
"""

from __future__ import annotations

import re
from typing import Any

from app.adapters.footage_types import FallbackTier, FootageKind
from app.pipeline.clip_selection.models import ClipCandidate
from app.pipeline.scene_planning.models import PlannedScene

# Tags that warrant a penalty when the scene is marked sensitive
_SENSITIVE_TAGS = frozenset([
    "war", "violence", "gun", "weapon", "blood", "death", "crime",
    "protest", "riot", "explosion", "fire", "crash", "accident",
])

# Common stopwords to exclude from keyword extraction
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can not in on at of to "
    "for with by from as and or but so yet i we you he she it they "
    "this that these those".split()
)


def _tokenise(text: str) -> set[str]:
    """Lower-cased word tokens, stopwords removed."""
    return {
        w.lower() for w in re.findall(r"[a-zA-Z]{2,}", text)
        if w.lower() not in _STOPWORDS
    }


def _tag_overlap_score(scene_tokens: set[str], candidate_tags: list[str]) -> float:
    """Fraction of scene tokens that appear in candidate tags, scaled to [0, 1]."""
    if not scene_tokens or not candidate_tags:
        return 0.0
    tag_tokens: set[str] = set()
    for tag in candidate_tags:
        tag_tokens.update(_tokenise(tag))
    overlap = len(scene_tokens & tag_tokens)
    return min(1.0, overlap / max(len(scene_tokens), 1))


def _tier_bonus(fallback_tier: FallbackTier) -> float:
    """Small score bonus based on footage source quality."""
    return {
        FallbackTier.PIXABAY_VIDEO: 0.5,
        FallbackTier.KEN_BURNS:     0.2,
        FallbackTier.CACHED_CLIP:   0.1,
        FallbackTier.TITLE_CARD:    -0.5,
    }.get(fallback_tier, 0.0)


def _sensitivity_penalty(scene: PlannedScene, candidate_tags: list[str]) -> float:
    if not scene.is_sensitive:
        return 0.0
    tag_tokens: set[str] = set()
    for tag in candidate_tags:
        tag_tokens.update(_tokenise(tag))
    if tag_tokens & _SENSITIVE_TAGS:
        return -1.0
    return 0.0


def score_text(
    cc: ClipCandidate,
    scene: PlannedScene,
    query_text: str,
) -> ClipCandidate:
    """
    Compute text_score and text_score_reason for *cc* in the context of *scene*.

    Modifies *cc* in-place and returns it.

    Args:
        cc:         The candidate to score.
        scene:      The planned scene being filled.
        query_text: The query string that retrieved this candidate.
    """
    # Build query + narration token set
    combined_text = f"{query_text} {scene.narration_text} {scene.meaning}"
    scene_tokens = _tokenise(combined_text)

    # Tag overlap [0, 1]
    overlap = _tag_overlap_score(scene_tokens, cc.candidate.tags)

    # Tier bonus
    tier_bonus = _tier_bonus(cc.candidate.fallback_tier)

    # Query recency bonus: earlier queries are more literal (more relevant)
    query_bonus = max(0.0, (2 - cc.query_index) * 0.15)

    # Sensitivity penalty
    sensitivity_penalty = _sensitivity_penalty(scene, cc.candidate.tags)

    # Raw score on [1, 5]
    raw = 1.0 + (overlap * 3.0) + tier_bonus + query_bonus + sensitivity_penalty
    score = max(1.0, min(5.0, raw))

    reasons: list[str] = []
    reasons.append(f"tag_overlap={overlap:.2f}")
    if tier_bonus != 0:
        reasons.append(f"tier_bonus={tier_bonus:+.1f}")
    if query_bonus:
        reasons.append(f"query_recency={query_bonus:+.2f}")
    if sensitivity_penalty:
        reasons.append(f"sensitivity_penalty={sensitivity_penalty:.1f}")

    cc.text_score = round(score, 3)
    cc.text_score_reason = "; ".join(reasons)
    return cc
