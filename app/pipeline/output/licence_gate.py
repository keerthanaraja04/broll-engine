"""
Licence gate — fails the job if any chosen clip lacks licence/attribution
or is flagged watermarked.

Checks performed per chosen clip:
  1. license field is non-empty and not "Unknown"
  2. attribution field is non-empty
  3. commercial_use is True
  4. No "watermarked" flag in vlm_flags
  5. No "copyright" flag in vlm_flags

Raises LicenceGateError if any check fails.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import LicenceGateError
from app.core.logging import get_logger
from app.pipeline.clip_selection.models import VideoSelectionPlan

logger = get_logger(__name__)

_WATERMARK_FLAGS = frozenset(["watermarked", "copyright", "logo_overlay"])
_UNACCEPTABLE_LICENSES = frozenset(["unknown", "none", "unlicensed", "all rights reserved"])


def check_licence_gate(selection_plan: VideoSelectionPlan) -> list[str]:
    """
    Validate every chosen clip in *selection_plan* against the licence gate.

    Returns a list of violation strings (empty = pass).
    Raises LicenceGateError if any violations are found.
    """
    violations: list[str] = []

    for sr in selection_plan.scene_results:
        if sr.chosen is None:
            continue

        c = sr.chosen.candidate
        clip_id = c.id

        # 1. Licence present
        lic = (c.license or "").strip()
        if not lic or lic.lower() in _UNACCEPTABLE_LICENSES:
            violations.append(
                f"Clip {clip_id}: missing or unacceptable licence '{lic}'"
            )

        # 2. Attribution present
        attr = (c.attribution or "").strip()
        if not attr and c.source != "ffmpeg_card":
            violations.append(
                f"Clip {clip_id}: missing attribution"
            )

        # 3. Commercial use
        if not c.commercial_use:
            violations.append(
                f"Clip {clip_id}: not cleared for commercial use"
            )

        # 4. Watermark / copyright flags from VLM
        for flag in sr.chosen.vlm_flags:
            if flag.lower() in _WATERMARK_FLAGS:
                violations.append(
                    f"Clip {clip_id}: flagged as '{flag}' by VLM"
                )

    if violations:
        logger.error(
            "licence_gate_failed",
            extra={"violations": len(violations), "details": violations[:5]},
        )
        raise LicenceGateError(
            f"Licence gate failed: {len(violations)} violation(s): {'; '.join(violations)}",
            details={"violations": violations},
        )

    logger.info("licence_gate_passed", extra={
        "scenes": selection_plan.scene_count,
        "clips": selection_plan.chosen_count,
    })
    return violations
