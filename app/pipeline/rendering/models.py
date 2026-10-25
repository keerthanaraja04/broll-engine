"""Data models for the rendering stage."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class AspectRatio(str, enum.Enum):
    RATIO_16_9 = "16:9"
    RATIO_9_16 = "9:16"
    RATIO_1_1 = "1:1"


class ReframeMethod(str, enum.Enum):
    VLM_BBOX = "vlm_bbox"
    SALIENCY = "saliency"
    CENTER = "center"
    BLURRED_BG = "blurred_bg"


@dataclass
class RenderedSegment:
    """One normalised scene segment ready for concatenation."""
    scene_id: int
    local_path: str          # path to the normalised .mp4 on disk
    duration: float
    width: int
    height: int
    fps: int
    reframe_method: str = "center"
    brightness_adjusted: bool = False
    cached: bool = False     # True if loaded from scene cache
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id, "local_path": self.local_path,
            "duration": self.duration, "width": self.width,
            "height": self.height, "fps": self.fps,
            "reframe_method": self.reframe_method,
            "brightness_adjusted": self.brightness_adjusted,
            "cached": self.cached, "warnings": self.warnings,
        }


@dataclass
class RenderResult:
    """Output of the entire rendering stage for one job."""
    job_id: str
    aspect_ratio: str
    final_video_path: str | None = None
    srt_path: str | None = None
    final_video_object: str | None = None   # MinIO object key
    srt_object: str | None = None
    segments: list[RenderedSegment] = field(default_factory=list)
    post_check_passed: bool = False
    post_check_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id, "aspect_ratio": self.aspect_ratio,
            "final_video_path": self.final_video_path,
            "srt_path": self.srt_path,
            "final_video_object": self.final_video_object,
            "srt_object": self.srt_object,
            "segments": [s.to_dict() for s in self.segments],
            "post_check_passed": self.post_check_passed,
            "post_check_errors": self.post_check_errors,
            "warnings": self.warnings,
            "total_duration": self.total_duration,
        }
