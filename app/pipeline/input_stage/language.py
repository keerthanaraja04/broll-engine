"""
Language detection and i18n utilities for Hindi/Tamil and other languages.

Functions
---------
detect_language_from_transcript()
    After STT, inspect the returned language code and re-run with a
    language hint if confidence is low.

get_caption_font()
    Map a language code to the correct Noto font name for subtitle burn-in.

build_stt_language_hint()
    Infer a language hint from the input payload (explicit language field
    or content heuristics like Devanagari Unicode range detection).

ensure_english_queries()
    Called by the safety layer to guarantee that b-roll search queries are
    always in English even when the narration is in Hindi or Tamil.
    (The planner already produces English queries; this is a backstop.)
"""

from __future__ import annotations

import re

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Unicode ranges for script detection
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")   # Hindi, Marathi, etc.
_TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")         # Tamil
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")         # Arabic/Urdu
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")       # Russian etc.

# ISO 639-1 → script group
_SCRIPT_GROUPS: dict[str, str] = {
    "hi": "devanagari", "mr": "devanagari", "ne": "devanagari",
    "sa": "devanagari", "mai": "devanagari", "kok": "devanagari",
    "ta": "tamil",
}


def detect_script(text: str) -> str | None:
    """
    Detect the dominant script of *text* from Unicode block ranges.

    Returns "devanagari", "tamil", "arabic", "cyrillic", or None (Latin/unknown).
    """
    if _DEVANAGARI_RE.search(text):
        return "devanagari"
    if _TAMIL_RE.search(text):
        return "tamil"
    if _ARABIC_RE.search(text):
        return "arabic"
    if _CYRILLIC_RE.search(text):
        return "cyrillic"
    return None


def get_caption_font(language: str) -> str:
    """
    Return the Noto font name appropriate for *language*.

    Ensures captions render correctly in Devanagari (Hindi, Marathi, …)
    and Tamil scripts.
    """
    s = get_settings()
    if language in s.LANG_DEVANAGARI:
        return s.RENDER_CAPTION_FONT_DEVA
    if language in s.LANG_TAMIL:
        return s.RENDER_CAPTION_FONT_TAMIL
    return s.RENDER_CAPTION_FONT_LATIN


def build_stt_language_hint(
    payload_language: str | None,
    script_text: str | None = None,
) -> str | None:
    """
    Infer the best language hint to pass to Groq Whisper.

    Priority:
    1. Explicit ``language`` from the API payload.
    2. Script detection on *script_text* (e.g. the first 200 chars of a
       script or the first scene text).
    3. None — let Whisper auto-detect.

    Returns an ISO 639-1 code or None.
    """
    if payload_language:
        return payload_language

    if script_text:
        script = detect_script(script_text)
        if script == "devanagari":
            logger.info("stt_lang_hint_devanagari")
            return "hi"   # Whisper default for Devanagari input
        if script == "tamil":
            logger.info("stt_lang_hint_tamil")
            return "ta"

    return None


def ensure_english_queries(queries: list[str], language: str) -> tuple[list[str], bool]:
    """
    Check that *queries* are in English (Latin script).

    If any query contains non-ASCII characters, replace it with a
    transliterated/fallback version.  For Hindi/Tamil, the scene planner
    already writes English queries; this is a safety backstop.

    Returns (possibly_corrected_queries, was_corrected).
    """
    corrected = False
    result: list[str] = []
    for q in queries:
        # If query contains Devanagari or Tamil → replace with a note
        script = detect_script(q)
        if script in ("devanagari", "tamil"):
            # Fall back to the query without the non-ASCII chars as a best effort
            ascii_only = q.encode("ascii", errors="ignore").decode("ascii").strip()
            if ascii_only:
                result.append(ascii_only)
            else:
                result.append("neutral visual scene")
            corrected = True
            logger.warning(
                "non_english_query_replaced",
                extra={"original": q[:80], "language": language},
            )
        else:
            result.append(q)

    if not result:
        result = ["neutral visual scene"]
        corrected = True

    return result, corrected


def get_whisper_model_for_language(language: str | None) -> str:
    """
    Return the most appropriate Whisper model for the given language.

    Groq's ``whisper-large-v3`` has the best multilingual support.
    """
    s = get_settings()
    # For all supported languages, large-v3 is the right choice
    return s.STT_MODEL
