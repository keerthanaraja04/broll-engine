"""
Safety rules enforced at planning and clip selection time.

Rules
-----
1. Named real people   → no face-focused search queries; replace with
   neutral/symbolic visuals.  Add a warning.
2. Brands              → generic category footage, never invented logos.
   Strip brand names from queries.
3. Sensitive news      → calm, neutral, non-graphic footage.  No "upbeat"
   style.  VLM rejects clips with "graphic" or "nsfw" flags.
4. "avoid" list        → enforced at query time (queries containing avoid
   terms are replaced) and again by the VLM on the chosen clip (reject
   if VLM flags match avoid terms).
5. Abstract lines/stats→ metaphor queries or a clean data card.
6. Double meanings     → use the planner's disambiguated sense in queries.

All rules produce SafetyResult objects with actions taken and warnings.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger
from app.pipeline.scene_planning.models import PlannedScene

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SafetyAction(str, enum.Enum):
    QUERY_REPLACED = "query_replaced"       # query was rewritten
    QUERY_DROPPED = "query_dropped"         # query removed entirely
    WARNING_ADDED = "warning_added"         # warning to job
    CLIP_REJECTED = "clip_rejected"          # chosen clip rejected
    FALLBACK_CARD = "fallback_to_data_card"  # use a title/data card instead


@dataclass
class SafetyCheck:
    """Result of one safety rule check on one scene."""
    rule_name: str
    action: SafetyAction
    detail: str
    original_query: str | None = None
    replacement_query: str | None = None


@dataclass
class SafetyResult:
    """Aggregated safety result for one scene."""
    scene_id: int
    queries: list[str]                # sanitized queries to use
    warnings: list[str] = field(default_factory=list)
    checks: list[SafetyCheck] = field(default_factory=list)
    use_data_card: bool = False       # True if scene should use a card fallback
    reject_graphic: bool = False      # True if VLM should reject graphic clips


# ---------------------------------------------------------------------------
# Rule 1: Named real people
# ---------------------------------------------------------------------------

def _check_named_person(scene: PlannedScene) -> SafetyCheck | None:
    """
    If scene has named_person, replace face-focused queries with neutral/symbolic ones.
    """
    if not scene.named_person:
        return None

    # Replace any query that looks like a person-name search
    person_name = scene.named_person.lower()
    replacement = "abstract people crowd silhouette neutral"

    return SafetyCheck(
        rule_name="named_person",
        action=SafetyAction.QUERY_REPLACED,
        detail=(
            f"Scene mentions person '{scene.named_person}': "
            f"face-focused search disabled, using neutral/symbolic visuals"
        ),
        replacement_query=replacement,
    )


# ---------------------------------------------------------------------------
# Rule 2: Brands
# ---------------------------------------------------------------------------

_BRAND_SUFFIXES = re.compile(
    r"\b(inc|corp|corporation|company|co|ltd|llc|gmbh)\b",
    re.IGNORECASE,
)

def _check_brand(scene: PlannedScene) -> SafetyCheck | None:
    """
    If scene has named_brand, strip brand name from queries and use
    generic category footage instead.
    """
    if not scene.named_brand:
        return None

    return SafetyCheck(
        rule_name="brand_protection",
        action=SafetyAction.QUERY_REPLACED,
        detail=(
            f"Scene mentions brand '{scene.named_brand}': "
            f"using generic category footage, brand name removed from queries"
        ),
        replacement_query="generic product category neutral",
    )


# ---------------------------------------------------------------------------
# Rule 3: Sensitive news
# ---------------------------------------------------------------------------

_SENSITIVE_INDICATORS = frozenset([
    "death", "kill", "attack", "war", "crash", "disaster",
    "tragedy", "violence", "shooting", "bombing", "explosion",
    "casualt", "victim", "casualties",
])

_GRAPHIC_FLAGS = frozenset(["graphic", "nsfw", "gory", "violent", "disturbing"])

def _check_sensitive_news(scene: PlannedScene) -> SafetyCheck | None:
    """
    If scene is_sensitive, enforce calm/neutral queries and reject graphic clips.
    """
    if not scene.is_sensitive:
        return None

    return SafetyCheck(
        rule_name="sensitive_news",
        action=SafetyAction.WARNING_ADDED,
        detail=(
            "Sensitive content: using calm, neutral, non-graphic footage; "
            "upbeat style disabled; VLM will reject graphic clips"
        ),
    )


# ---------------------------------------------------------------------------
# Rule 4: Avoid list enforcement
# ---------------------------------------------------------------------------

def enforce_avoid_list(
    queries: list[str],
    avoid: list[str] | None,
) -> tuple[list[str], list[SafetyCheck]]:
    """
    Remove or replace any query containing a term from the *avoid* list.

    Returns (sanitized_queries, checks).
    """
    if not avoid:
        return queries, []

    avoid_lower = {a.strip().lower() for a in avoid if a.strip()}
    checks: list[SafetyCheck] = []
    sanitized: list[str] = []

    for q in queries:
        q_lower = q.lower()
        matched = [a for a in avoid_lower if a in q_lower]
        if matched:
            checks.append(SafetyCheck(
                rule_name="avoid_list",
                action=SafetyAction.QUERY_DROPPED,
                detail=f"Query '{q}' contains avoid term(s): {matched}",
                original_query=q,
            ))
            # Don't add the query — it's dropped
        else:
            sanitized.append(q)

    if not sanitized:
        # All queries were dropped — add a safe fallback
        sanitized.append("neutral abstract background")
        if not checks:
            checks.append(SafetyCheck(
                rule_name="avoid_list",
                action=SafetyAction.QUERY_REPLACED,
                detail="All queries contained avoid terms; using safe fallback",
                replacement_query="neutral abstract background",
            ))
        else:
            checks[-1].replacement_query = "neutral abstract background"

    return sanitized, checks


# ---------------------------------------------------------------------------
# Rule 5: Abstract / stats → metaphor or data card
# ---------------------------------------------------------------------------

def _check_abstract_stats(scene: PlannedScene) -> SafetyCheck | None:
    """
    If scene is abstract or has_numbers, prefer metaphor queries or a data card.
    """
    if scene.has_numbers and not scene.queries:
        return SafetyCheck(
            rule_name="stats_data_card",
            action=SafetyAction.FALLBACK_CARD,
            detail="Scene contains statistics: using a clean data card",
        )

    if scene.is_abstract:
        return SafetyCheck(
            rule_name="abstract_content",
            action=SafetyAction.WARNING_ADDED,
            detail="Abstract content: metaphor queries preferred over literal",
        )

    return None


# ---------------------------------------------------------------------------
# Rule 6: Double meanings → use disambiguated sense
# ---------------------------------------------------------------------------

def _apply_word_senses(queries: list[str], scene: PlannedScene) -> list[str]:
    """
    If the scene has word_senses, replace ambiguous words with their
    disambiguated sense in the queries.
    """
    if not scene.word_senses:
        return queries

    # Build replacement map: lowercased word → chosen_sense
    replacements: dict[str, str] = {}
    for ws in scene.word_senses:
        replacements[ws.word.lower()] = ws.chosen_sense

    updated: list[str] = []
    for q in queries:
        modified = q
        for word, sense in replacements.items():
            # Replace whole-word, case-insensitive
            modified = re.sub(
                r"\b" + re.escape(word) + r"\b",
                sense,
                modified,
                flags=re.IGNORECASE,
            )
        updated.append(modified)

    return updated


# ---------------------------------------------------------------------------
# VLM safety flags (Rule 4b — post-selection check)
# ---------------------------------------------------------------------------

def check_vlm_safety_flags(
    vlm_flags: list[str],
    is_sensitive: bool,
) -> tuple[bool, str | None]:
    """
    Check VLM flags on the chosen clip for safety violations.

    Returns (should_reject, reason).
    """
    flag_set = {f.lower() for f in vlm_flags}

    # Always reject graphic/NSFW
    if flag_set & _GRAPHIC_FLAGS:
        return True, f"VLM flagged clip as {list(flag_set & _GRAPHIC_FLAGS)}"

    # For sensitive scenes, reject upbeat/celebratory flags
    if is_sensitive:
        upbeat_flags = flag_set & {"upbeat", "celebratory", "party", "festive"}
        if upbeat_flags:
            return True, (
                f"Sensitive scene but clip flagged as upbeat: {list(upbeat_flags)}"
            )

    return False, None


# ---------------------------------------------------------------------------
# Full scene safety check
# ---------------------------------------------------------------------------

def sanitize_queries(
    queries: list[str],
    scene: PlannedScene,
    avoid: list[str] | None = None,
) -> SafetyResult:
    """
    Apply all safety rules to the scene's queries.

    Returns a SafetyResult with sanitized queries and warnings.
    """
    result = SafetyResult(scene_id=scene.id, queries=list(queries))
    checks: list[SafetyCheck] = []

    # Rule 6: Apply word senses first (disambiguate before other rules)
    disambiguated = _apply_word_senses(queries, scene)

    # Rule 1: Named person → replace queries
    person_check = _check_named_person(scene)
    if person_check:
        checks.append(person_check)
        new_q: list[str] = []
        person_words = [re.escape(w) for w in (scene.named_person or "").split() if len(w) > 1]
        pattern = re.compile(r"\b(" + "|".join(person_words) + r")\b", re.IGNORECASE) if person_words else None
        for q in disambiguated:
            cleaned = pattern.sub("", q).strip() if pattern else q
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned or len(cleaned) < 3 or (pattern and pattern.search(cleaned)):
                new_q.append(person_check.replacement_query or "abstract people crowd silhouette neutral")
            else:
                new_q.append(cleaned)
        if not new_q:
            new_q = [person_check.replacement_query or "abstract people crowd silhouette neutral"]
        disambiguated = new_q
        result.warnings.append(person_check.detail)

    # Rule 2: Brand → strip brand
    brand_check = _check_brand(scene)
    if brand_check:
        checks.append(brand_check)
        new_q = []
        brand_clean = _BRAND_SUFFIXES.sub("", scene.named_brand or "").strip()
        brand_words = [re.escape(w) for w in (brand_clean or scene.named_brand or "").split() if len(w) > 1]
        pattern = re.compile(r"\b(" + "|".join(brand_words) + r")\b", re.IGNORECASE) if brand_words else None
        for q in disambiguated:
            cleaned = pattern.sub("", q).strip() if pattern else q
            cleaned = _BRAND_SUFFIXES.sub("", cleaned).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if not cleaned or len(cleaned) < 3 or (pattern and pattern.search(cleaned)):
                new_q.append(brand_check.replacement_query or "generic product category neutral")
            else:
                new_q.append(cleaned)
        if not new_q:
            new_q = [brand_check.replacement_query or "generic product category neutral"]
        disambiguated = new_q
        result.warnings.append(brand_check.detail)

    # Rule 3: Sensitive news
    sensitive_check = _check_sensitive_news(scene)
    if sensitive_check:
        checks.append(sensitive_check)
        result.warnings.append(sensitive_check.detail)
        result.reject_graphic = True

    # Rule 4: Avoid list
    disambiguated, avoid_checks = enforce_avoid_list(disambiguated, avoid)
    checks.extend(avoid_checks)
    for ac in avoid_checks:
        if ac.action == SafetyAction.QUERY_DROPPED:
            result.warnings.append(ac.detail)

    # Rule 5: Abstract / stats
    stats_check = _check_abstract_stats(scene)
    if stats_check:
        checks.append(stats_check)
        if stats_check.action == SafetyAction.FALLBACK_CARD:
            result.use_data_card = True
        result.warnings.append(stats_check.detail)

    result.queries = disambiguated[:3]  # max 3 queries
    result.checks = checks

    logger.info(
        "safety_checked",
        extra={
            "scene_id": scene.id,
            "original_queries": len(queries),
            "sanitized_queries": len(result.queries),
            "warnings": len(result.warnings),
            "use_data_card": result.use_data_card,
        },
    )
    return result


def check_scene_safety(
    scene: PlannedScene,
    avoid: list[str] | None = None,
) -> SafetyResult:
    """
    Full safety check for one scene — convenience wrapper for sanitize_queries.
    """
    return sanitize_queries(scene.queries, scene, avoid=avoid)
