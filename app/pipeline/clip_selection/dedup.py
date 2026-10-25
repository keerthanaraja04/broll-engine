"""
Clip deduplication utilities.

Two mechanisms are used:

1. Exact ID dedup — ``source:id`` strings; O(1) set lookup.
   Used to drop clips already selected earlier in the same video.

2. Perceptual hash dedup — thumbnail bytes → 64-bit dHash via the
   ``imagehash`` library.  Two thumbnails whose Hamming distance is
   ≤ CLIP_PHASH_MAX_DISTANCE are considered the same shot.

Both run entirely in-process with no network calls.  If ``imagehash``
is not installed, perceptual dedup is skipped (with a log warning).
"""

from __future__ import annotations

import hashlib
from typing import Iterator

from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.clip_selection.models import ClipCandidate

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Exact dedup
# ---------------------------------------------------------------------------

class UsedClipRegistry:
    """
    Job-scoped registry of already-selected clip IDs.

    Reset once per job; passed to every scene's selection call.
    """

    def __init__(self) -> None:
        self._ids: set[str] = set()

    def mark_used(self, clip_id: str) -> None:
        self._ids.add(clip_id)

    def is_used(self, clip_id: str) -> bool:
        return clip_id in self._ids

    def __len__(self) -> int:
        return len(self._ids)


# ---------------------------------------------------------------------------
# Perceptual hash
# ---------------------------------------------------------------------------

def _compute_phash(image_bytes: bytes) -> str | None:
    """
    Compute a perceptual dHash for *image_bytes*.

    Returns a hex-string or None if imagehash / PIL are unavailable.
    """
    try:
        import io
        import imagehash  # type: ignore[import]
        from PIL import Image  # type: ignore[import]

        img = Image.open(io.BytesIO(image_bytes))
        h = imagehash.dhash(img)
        return str(h)
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("phash_failed", extra={"error": str(exc)})
        return None


def _hamming_distance(h1: str, h2: str) -> int:
    """
    Hamming distance between two hex-encoded hash strings.
    Converts each hex char to 4 bits and counts differing bits.
    """
    if len(h1) != len(h2):
        return 999  # incompatible hashes
    try:
        a = int(h1, 16)
        b = int(h2, 16)
        return bin(a ^ b).count("1")
    except ValueError:
        return 999


# ---------------------------------------------------------------------------
# Main dedup function
# ---------------------------------------------------------------------------

def deduplicate_candidates(
    candidates: list[ClipCandidate],
    used_registry: UsedClipRegistry,
    thumbnail_bytes_map: dict[str, bytes],  # candidate_id → thumb bytes
) -> list[ClipCandidate]:
    """
    Filter and annotate *candidates* for duplicates.

    Steps
    -----
    1. Drop candidates whose ID is already in ``used_registry``.
    2. Compute perceptual hashes for thumbnails (if enabled and bytes available).
    3. Mark candidates as duplicate if their hash is within
       CLIP_PHASH_MAX_DISTANCE of any already-seen hash.
    4. Return the surviving candidates (non-duplicate, not in used_registry),
       with ``phash`` and ``is_duplicate`` fields populated.

    The caller decides whether to skip ``is_duplicate`` entries or use them
    as a lower-priority fallback.
    """
    s = get_settings()
    surviving: list[ClipCandidate] = []
    seen_hashes: list[str] = []

    for cc in candidates:
        cid = cc.candidate.id

        # 1. Exact ID check
        if used_registry.is_used(cid):
            cc.is_duplicate = True
            continue

        # 2. Perceptual hash
        if s.CLIP_PHASH_ENABLED:
            thumb_bytes = thumbnail_bytes_map.get(cid)
            if thumb_bytes:
                h = _compute_phash(thumb_bytes)
                if h is not None:
                    cc.phash = h
                    # 3. Compare against already-seen hashes
                    for seen in seen_hashes:
                        if _hamming_distance(h, seen) <= s.CLIP_PHASH_MAX_DISTANCE:
                            cc.is_duplicate = True
                            break
                    if not cc.is_duplicate:
                        seen_hashes.append(h)
            # If no thumb bytes, we can't phash — not a duplicate by default

        if not cc.is_duplicate:
            surviving.append(cc)

    return surviving
