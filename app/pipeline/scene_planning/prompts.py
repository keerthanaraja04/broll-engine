"""
Prompt templates for scene planning.

Kept in a separate module so they can be tuned without touching business logic.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a professional video editor planning b-roll footage for a narrated video.
Your task: divide the narration into visual scenes and generate English stock-footage
search queries for each scene.

Rules
-----
1. Scene boundaries MUST align with sentence or phrase boundaries in the transcript.
2. Scenes MUST cover the ENTIRE audio: scene[0].start == 0.0,
   scene[-1].end == total_duration, and each scene[i].end == scene[i+1].start
   (no gaps, no overlaps).
3. Scene duration: between {min_dur:.1f}s and {max_dur:.1f}s.
4. Queries are in English regardless of narration language.
5. Provide 3 queries per scene ordered literal → metaphorical:
     query[0]: literal visual description  (e.g. "person walking on street")
     query[1]: broader thematic query      (e.g. "city commuter lifestyle")
     query[2]: abstract / conceptual query (e.g. "freedom independence movement")
6. Respect style="{style}" and exclude themes from avoid={avoid}.
7. For ambiguous words ({wsd_words}), choose the correct sense for this context
   and record it in word_senses.
8. For non-English narration, write meaning in the source language but
   queries MUST be in English.
9. Output ONLY a valid JSON object — no prose, no markdown fences.
"""

USER_PROMPT = """\
Narration language: {language}
Total audio duration: {total_duration:.3f} seconds
Style hint: {style}
Avoid: {avoid}

Transcript (with word timestamps):
{transcript_text}

Sentence list (id, start, end, text):
{sentence_list}

{scenes_hint}

Return a JSON object with this exact structure:
{{
  "scenes": [
    {{
      "start": <float>,
      "end": <float>,
      "narration_text": "<verbatim excerpt>",
      "meaning": "<one-line meaning in {language}>",
      "queries": ["<literal EN query>", "<thematic EN query>", "<abstract EN query>"],
      "is_abstract": <bool>,
      "has_numbers": <bool>,
      "named_person": <"name" or null>,
      "named_brand": <"name" or null>,
      "is_sensitive": <bool>,
      "word_senses": [{{"word": "<word>", "chosen_sense": "<sense>"}}]
    }},
    ...
  ]
}}
"""

SCENES_HINT_WITH_HINTS = """\
The caller supplied per-scene visual hints. For scenes that map to a hint,
use the hint as queries[0] (override your own literal query):
{hints}
"""

SCENES_HINT_EMPTY = ""


def build_prompts(
    language: str,
    total_duration: float,
    transcript_text: str,
    sentence_list: str,
    style: str,
    avoid: list[str],
    wsd_words: list[str],
    min_dur: float,
    max_dur: float,
    visual_hints: list[tuple[float, float, str]] | None = None,
) -> tuple[str, str]:
    """
    Return (system_prompt, user_prompt) ready to pass to LLMAdapter.json_complete().

    Args:
        visual_hints: list of (start, end, hint_text) from scenes[] input.
    """
    system = SYSTEM_PROMPT.format(
        min_dur=min_dur,
        max_dur=max_dur,
        style=style or "neutral",
        avoid=str(avoid) if avoid else "[]",
        wsd_words=", ".join(wsd_words),
    )

    if visual_hints:
        hints_text = "\n".join(
            f"  [{start:.2f}s–{end:.2f}s]: {hint}"
            for start, end, hint in visual_hints
        )
        scenes_hint = SCENES_HINT_WITH_HINTS.format(hints=hints_text)
    else:
        scenes_hint = SCENES_HINT_EMPTY

    # Build readable sentence list
    user = USER_PROMPT.format(
        language=language,
        total_duration=total_duration,
        style=style or "neutral",
        avoid=str(avoid) if avoid else "[]",
        transcript_text=transcript_text[:3000],  # cap to avoid token overflow
        sentence_list=sentence_list[:2000],
        scenes_hint=scenes_hint,
    )
    return system, user
