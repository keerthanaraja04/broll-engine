"""
Rule-based scene splitter — the fallback when LLM planning fails.

Algorithm
---------
1. Walk the Transcript sentences left to right, accumulating a window.
2. Flush the window into a new scene when:
     a. The accumulated duration exceeds SCENE_MAX_DURATION_SECONDS, OR
     b. The current sentence ends with terminal punctuation (.!?) AND
        the accumulated duration >= SCENE_MIN_DURATION_SECONDS.
3. If a sentence would create a scene shorter than SCENE_MIN_DURATION_SECONDS,
   merge it with the next one.
4. The final scene is always extended to total_duration.

Per-scene metadata
------------------
Since there is no LLM call, metadata is generated deterministically:
  - meaning:      first 80 characters of narration text
  - queries:      naive extraction — nouns and noun phrases (simple regex)
  - is_abstract:  False
  - has_numbers:  True if any digit in narration_text
  - named_person: None  (no NER without LLM)
  - named_brand:  None
  - is_sensitive: False
  - word_senses:  [] (no WSD without LLM)
"""

from __future__ import annotations

import re
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.input_stage.transcript import Sentence, Transcript
from app.pipeline.scene_planning.models import PlannedScene, ScenePlan

logger = get_logger(__name__)

# Simple noun-phrase extractor: capitalised tokens or runs of non-stop-words
_STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "could should may might shall can cannot in on at of to for with by from as "
    "and or but not so yet i we you he she it they this that these those".split()
)
_DIGIT_RE = re.compile(r"\d")


def _extract_queries(text: str, visual_hint: str | None = None) -> list[str]:
    """
    Return up to 3 search queries extracted from *text*.

    Strategy:
    1. If visual_hint is set, use it as the first query.
    2. Capitalised multi-word phrases (likely proper nouns / titles).
    3. Non-stop-word content words (top 3 by frequency).
    """
    queries: list[str] = []

    if visual_hint and visual_hint.strip():
        queries.append(visual_hint.strip())

    # Capitalised phrases (2+ consecutive cap-starting words)
    cap_phrases = re.findall(
        r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
    )
    for phrase in cap_phrases:
        if phrase not in queries:
            queries.append(phrase)
        if len(queries) >= 3:
            break

    # Non-stop content words sorted by length (longer = more specific)
    if len(queries) < 3:
        words = re.findall(r"[a-zA-Z]{3,}", text)
        content = sorted(
            {w.lower() for w in words if w.lower() not in _STOPWORDS},
            key=lambda w: -len(w),
        )
        for w in content:
            candidate = w.replace("_", " ")
            if candidate not in queries:
                queries.append(candidate)
            if len(queries) >= 3:
                break

    # Final fallback: first 60 chars of text
    if not queries:
        queries.append(text[:60].strip())

    return queries[:3]


def _sentence_groups_to_scenes(
    groups: list[list[Sentence]],
    total_duration: float,
    style: str | None = None,
) -> list[PlannedScene]:
    scenes: list[PlannedScene] = []
    for i, group in enumerate(groups):
        if not group:
            continue
        start = group[0].start
        end = group[-1].end
        # Extend final group to cover total_duration
        if i == len(groups) - 1:
            end = total_duration

        narration = " ".join(s.text for s in group)
        visual_hint = next(
            (s.visual_hint for s in group if s.visual_hint), None
        )
        queries = _extract_queries(narration, visual_hint=visual_hint)

        scenes.append(
            PlannedScene(
                id=i,
                start=start,
                end=end,
                narration_text=narration,
                meaning=narration[:80],
                queries=queries,
                is_abstract=False,
                has_numbers=bool(_DIGIT_RE.search(narration)),
                named_person=None,
                named_brand=None,
                is_sensitive=False,
                visual_hint=visual_hint,
                source="rule_based",
            )
        )
    return scenes


def build_rule_based_plan(
    transcript: Transcript,
    style: str | None = None,
    avoid: list[str] | None = None,
) -> ScenePlan:
    """
    Produce a ScenePlan without an LLM call.

    Args:
        transcript: The input transcript.
        style:      Optional visual style hint (stored in warnings for reference).
        avoid:      Optional list of terms to exclude from queries.

    Returns:
        ScenePlan(source="rule_based").
    """
    s = get_settings()
    min_dur = s.SCENE_MIN_DURATION_SECONDS
    max_dur = s.SCENE_MAX_DURATION_SECONDS
    total_duration = transcript.duration
    avoid_set = {a.lower() for a in (avoid or [])}

    sentences = transcript.sentences
    if not sentences:
        # Degenerate: single scene for the whole duration
        scenes = [
            PlannedScene(
                id=0,
                start=0.0,
                end=total_duration,
                narration_text=transcript.text,
                meaning=transcript.text[:80],
                queries=[transcript.text[:60]] if transcript.text else ["scene"],
                source="rule_based",
            )
        ]
        return ScenePlan(
            scenes=scenes,
            total_duration=total_duration,
            source_language=transcript.language,
            warnings=["rule_based: no sentences — single scene created"],
        )

    # ── Group sentences into scenes ────────────────────────────────────
    groups: list[list[Sentence]] = []
    current: list[Sentence] = []
    current_dur = 0.0

    for sent in sentences:
        sent_dur = sent.duration
        current.append(sent)
        current_dur += sent_dur

        # Should we flush?
        ends_sentence = sent.text.rstrip().endswith((".", "!", "?"))
        over_max = current_dur >= max_dur
        over_min_and_punct = current_dur >= min_dur and ends_sentence

        if over_max or over_min_and_punct:
            groups.append(list(current))
            current = []
            current_dur = 0.0

    if current:
        # Merge tiny trailing group with previous if possible
        if groups and current_dur < min_dur:
            groups[-1].extend(current)
        else:
            groups.append(current)

    # ── Filter avoid terms from queries ───────────────────────────────
    def _filter_queries(queries: list[str]) -> list[str]:
        if not avoid_set:
            return queries
        filtered = [
            q for q in queries
            if not any(a in q.lower() for a in avoid_set)
        ]
        return filtered if filtered else queries  # never leave queries empty

    scenes = _sentence_groups_to_scenes(groups, total_duration, style)
    for sc in scenes:
        sc.queries = _filter_queries(sc.queries)

    warnings = [
        "rule_based fallback used — LLM scene planning failed or was disabled",
    ]
    if style:
        warnings.append(f"style='{style}' noted but not applied in rule-based mode")

    logger.info(
        "rule_based_plan_built",
        extra={
            "scenes": len(scenes),
            "duration": total_duration,
            "language": transcript.language,
        },
    )
    return ScenePlan(
        scenes=scenes,
        total_duration=total_duration,
        source_language=transcript.language,
        warnings=warnings,
    )
