"""
Data models for the scene planning stage.

PlannedScene  — one scene with timing, queries, and semantic metadata.
ScenePlan     — the complete ordered list of scenes for a job.

These are the canonical objects consumed by:
  - FootageRouter (uses queries[0] → queries[1] → queries[2])
  - Subtitle burn-in (uses start/end/narration_text)
  - Analytics / reporting

JSON Schema
-----------
The LLM is asked to return a JSON object matching SCENE_PLAN_SCHEMA
(defined below).  The validator and rule-based fallback both produce
objects that can be deserialised by ScenePlan.from_llm_dict().
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# JSON schema sent to the LLM
# ---------------------------------------------------------------------------

SCENE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "start", "end", "narration_text",
        "meaning", "queries",
        "is_abstract", "has_numbers",
        "named_person", "named_brand", "is_sensitive",
    ],
    "additionalProperties": False,
    "properties": {
        # Timing (seconds, from the audio timeline)
        "start": {"type": "number", "minimum": 0},
        "end":   {"type": "number", "minimum": 0},
        # Narration text covered by this scene (verbatim excerpt)
        "narration_text": {"type": "string", "minLength": 1},
        # One-line meaning in the content language
        "meaning": {"type": "string", "minLength": 1, "maxLength": 200},
        # 3 English search queries, literal → abstract
        "queries": {
            "type": "array",
            "items": {"type": "string", "minLength": 2},
            "minItems": 1,
            "maxItems": 3,
        },
        # Semantic flags
        "is_abstract":    {"type": "boolean"},
        "has_numbers":    {"type": "boolean"},
        "named_person":   {"type": ["string", "null"]},
        "named_brand":    {"type": ["string", "null"]},
        "is_sensitive":   {"type": "boolean"},
        # Ambiguous word senses: {"word": "bank", "chosen_sense": "river bank"}
        "word_senses": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["word", "chosen_sense"],
                "properties": {
                    "word":         {"type": "string"},
                    "chosen_sense": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
}

SCENE_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["scenes"],
    "additionalProperties": False,
    "properties": {
        "scenes": {
            "type": "array",
            "minItems": 1,
            "items": SCENE_ITEM_SCHEMA,
        }
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WordSense:
    word: str
    chosen_sense: str


@dataclass
class PlannedScene:
    """
    A single planned scene: timing + queries + semantic metadata.

    Attributes
    ----------
    id              : 0-based scene index within the job.
    start / end     : seconds from audio start; guaranteed start < end.
    narration_text  : verbatim transcript excerpt this scene covers.
    meaning         : one-line paraphrase / description (content language).
    queries         : 1–3 English search queries, literal→metaphorical order.
                      queries[0] is always tried first by FootageRouter.
    is_abstract     : True when the meaning is conceptual (ideas, feelings).
    has_numbers     : True when narration contains statistics, dates, counts.
    named_person    : person name if mentioned, else None.
    named_brand     : brand/product name if mentioned, else None.
    is_sensitive    : True if narration touches violence, death, politics, etc.
    word_senses     : WSD resolutions for ambiguous words.
    visual_hint     : optional caller-supplied override (from scenes[] input).
    source          : "llm" | "rule_based" — which planner produced it.
    warnings        : non-fatal notes.
    """

    id: int
    start: float
    end: float
    narration_text: str
    meaning: str
    queries: list[str]

    is_abstract: bool = False
    has_numbers: bool = False
    named_person: str | None = None
    named_brand: str | None = None
    is_sensitive: bool = False
    word_senses: list[WordSense] = field(default_factory=list)

    # Provenance
    visual_hint: str | None = None
    source: str = "llm"
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def primary_query(self) -> str:
        """The first (most literal) search query, falling back to meaning."""
        return (self.queries[0] if self.queries else self.meaning) or self.narration_text[:60]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "narration_text": self.narration_text,
            "meaning": self.meaning,
            "queries": self.queries,
            "is_abstract": self.is_abstract,
            "has_numbers": self.has_numbers,
            "named_person": self.named_person,
            "named_brand": self.named_brand,
            "is_sensitive": self.is_sensitive,
            "word_senses": [
                {"word": ws.word, "chosen_sense": ws.chosen_sense}
                for ws in self.word_senses
            ],
            "visual_hint": self.visual_hint,
            "source": self.source,
            "duration": self.duration,
        }

    @classmethod
    def from_llm_item(
        cls,
        item: dict[str, Any],
        scene_id: int,
        source: str = "llm",
    ) -> "PlannedScene":
        word_senses = [
            WordSense(word=ws["word"], chosen_sense=ws["chosen_sense"])
            for ws in item.get("word_senses", [])
        ]
        # Ensure at least one query
        queries = [q for q in item.get("queries", []) if q and q.strip()]
        if not queries:
            queries = [item.get("meaning", item.get("narration_text", "scene"))[:80]]

        return cls(
            id=scene_id,
            start=float(item["start"]),
            end=float(item["end"]),
            narration_text=item["narration_text"],
            meaning=item.get("meaning", ""),
            queries=queries[:3],
            is_abstract=bool(item.get("is_abstract", False)),
            has_numbers=bool(item.get("has_numbers", False)),
            named_person=item.get("named_person") or None,
            named_brand=item.get("named_brand") or None,
            is_sensitive=bool(item.get("is_sensitive", False)),
            word_senses=word_senses,
            source=source,
        )


@dataclass
class ScenePlan:
    """The complete ordered plan for a job."""

    scenes: list[PlannedScene]
    total_duration: float
    source_language: str
    warnings: list[str] = field(default_factory=list)

    @property
    def scene_count(self) -> int:
        return len(self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration": self.total_duration,
            "source_language": self.source_language,
            "scene_count": self.scene_count,
            "warnings": self.warnings,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    @classmethod
    def from_llm_dict(
        cls,
        data: dict[str, Any],
        total_duration: float,
        source_language: str,
        source: str = "llm",
    ) -> "ScenePlan":
        scenes = [
            PlannedScene.from_llm_item(item, i, source=source)
            for i, item in enumerate(data["scenes"])
        ]
        return cls(
            scenes=scenes,
            total_duration=total_duration,
            source_language=source_language,
        )
