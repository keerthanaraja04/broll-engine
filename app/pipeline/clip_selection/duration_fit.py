"""
Duration fit utilities.

A clip is "usable" for a scene if it can fill the scene duration either:
  - directly  (clip.duration >= scene.duration)
  - slowed    (clip.duration / min_slow_factor >= scene.duration)
  - looped    (clip.duration * max_loops >= scene.duration)

Clips that cannot fill the scene even with slowing or looping are flagged
``duration_ok = False`` and filtered out before scoring.
"""

from __future__ import annotations

from app.core.config import get_settings
from app.pipeline.clip_selection.models import ClipCandidate


def assess_duration_fit(
    cc: ClipCandidate,
    scene_duration: float,
) -> ClipCandidate:
    """
    Annotate *cc* with ``needs_slow``, ``needs_loop``, ``duration_ok``.

    Modifies *cc* in-place and returns it.
    """
    s = get_settings()
    clip_dur = cc.candidate.duration

    # Cards and static images (duration = 0 or very short) are special:
    # they are always generated to the exact scene length, so always fit.
    if clip_dur <= 0 or cc.candidate.source in ("ffmpeg_card", "ken_burns"):
        cc.needs_slow = False
        cc.needs_loop = 0
        cc.duration_ok = True
        return cc

    if clip_dur >= scene_duration:
        # Fits directly — no adjustment needed
        cc.needs_slow = False
        cc.needs_loop = 0
        cc.duration_ok = True
        return cc

    # Try slowing
    max_playback_dur_with_slow = clip_dur / s.CLIP_SLOW_MOTION_MIN_FACTOR
    if max_playback_dur_with_slow >= scene_duration:
        cc.needs_slow = True
        cc.needs_loop = 0
        cc.duration_ok = True
        return cc

    # Try looping
    loops_needed = int(scene_duration / clip_dur) + 1
    if loops_needed <= s.CLIP_MAX_LOOP_COUNT:
        cc.needs_slow = False
        cc.needs_loop = loops_needed
        cc.duration_ok = True
        return cc

    # Combine slow + loop
    loops_with_slow = int((scene_duration / max_playback_dur_with_slow)) + 1
    if loops_with_slow <= s.CLIP_MAX_LOOP_COUNT:
        cc.needs_slow = True
        cc.needs_loop = loops_with_slow
        cc.duration_ok = True
        return cc

    # Cannot fill scene
    cc.needs_slow = False
    cc.needs_loop = 0
    cc.duration_ok = False
    return cc
