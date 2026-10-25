"""
Stage-2 VLM scorer — expensive, Groq vision API call per thumbnail.

Takes the top-N ClipCandidates (after text scoring), fetches their
thumbnail bytes, and calls VLMAdapter.score() in one batched call.

The VLM score (1–5) is stored on each candidate.  If the adapter
returns flags such as ["nsfw", "text_overlay"], they are recorded.

Thumbnail fetching is done with httpx; failures are tolerated (the
candidate keeps vlm_score=0 and a warning is added).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.adapters.vlm_adapter import get_vlm_adapter
from app.core.config import get_settings
from app.core.logging import get_logger
from app.pipeline.clip_selection.models import ClipCandidate

logger = get_logger(__name__)


async def _fetch_thumb(url: str, timeout: float = 10.0) -> bytes | None:
    """Fetch thumbnail bytes. Returns None on any error."""
    if not url:
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content
    except Exception as exc:
        logger.debug("thumb_fetch_failed", extra={"url": url, "error": str(exc)})
        return None


async def score_vlm(
    candidates: list[ClipCandidate],
    narration_text: str,
    thumbnail_bytes_map: dict[str, bytes] | None = None,
) -> tuple[list[ClipCandidate], dict[str, bytes]]:
    """
    Score *candidates* with the VLM adapter.

    Args:
        candidates:           Candidates to score (mutated in-place).
        narration_text:       The scene narration used as the scoring text.
        thumbnail_bytes_map:  Pre-fetched bytes by candidate_id (avoids
                              re-fetching when called multiple times for
                              the same scene).  Updated with any newly
                              fetched bytes.

    Returns:
        (candidates, updated_thumbnail_bytes_map)
    """
    if thumbnail_bytes_map is None:
        thumbnail_bytes_map = {}

    s = get_settings()
    if not s.CLIP_SELECTION_VLM_ENABLED or not candidates:
        return candidates, thumbnail_bytes_map

    vlm = get_vlm_adapter()

    # Fetch any missing thumbnails in parallel
    async def _get_thumb(cc: ClipCandidate) -> bytes | None:
        cid = cc.candidate.id
        if cid in thumbnail_bytes_map:
            return thumbnail_bytes_map[cid]
        thumb_url = cc.candidate.thumb_url
        data = await _fetch_thumb(thumb_url)
        if data:
            thumbnail_bytes_map[cid] = data
        return data

    thumb_tasks = [_get_thumb(cc) for cc in candidates]
    thumb_results = await asyncio.gather(*thumb_tasks, return_exceptions=True)

    # Build parallel lists of (candidate, image_bytes) for scoring
    scorable: list[tuple[ClipCandidate, bytes]] = []
    for cc, result in zip(candidates, thumb_results):
        if isinstance(result, bytes) and result:
            scorable.append((cc, result))
        else:
            # No thumbnail available — vlm_score stays 0
            cc.vlm_score = 0.0
            cc.vlm_reason = "thumbnail unavailable"

    if not scorable:
        return candidates, thumbnail_bytes_map

    # Batch VLM call
    images = [img for _, img in scorable]
    try:
        vlm_scores = await vlm.score(images, narration_text)
    except Exception as exc:
        logger.warning("vlm_score_failed", extra={"error": str(exc)})
        for cc, _ in scorable:
            cc.vlm_score = 0.0
            cc.vlm_reason = f"VLM error: {exc}"
        return candidates, thumbnail_bytes_map

    for (cc, _), vs in zip(scorable, vlm_scores):
        cc.vlm_score = float(vs.score)
        cc.vlm_reason = vs.reason
        cc.vlm_flags = vs.flags

    return candidates, thumbnail_bytes_map
