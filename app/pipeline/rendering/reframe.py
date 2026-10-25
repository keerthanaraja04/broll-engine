"""
Smart reframing for 9:16 and 1:1 outputs.

Three strategies in priority order:
  1. VLM subject_bbox — if the clip has a VLM-provided bounding box, crop to it
  2. OpenCV saliency   — compute a saliency map to find the most visually
                         interesting region and crop to it
  3. Blurred background — scale + blur the source to fill the frame and
                           overlay the sharp (letterboxed) version on top

The chosen method is recorded on the RenderedSegment for auditability.
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def get_reframe_method(
    has_vlm_bbox: bool,
    aspect_ratio: str,
) -> str:
    """
    Decide which reframe method to use for *aspect_ratio*.

    For 16:9 we never reframe (just scale + pad).
    For 9:16 / 1:1 we prefer vlm_bbox if available, then saliency,
    then blurred_bg as the last resort.
    """
    s = get_settings()
    if aspect_ratio == "16:9":
        return "center"  # no reframing needed

    if has_vlm_bbox and s.RENDER_REFRAME_METHOD == "vlm_bbox":
        return "vlm_bbox"

    return s.RENDER_REFRAME_METHOD


def compute_crop_from_bbox(
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    bbox_x1: float = 0.0,
    bbox_y1: float = 0.0,
    bbox_x2: float = 1.0,
    bbox_y2: float = 1.0,
) -> tuple[float, float]:
    """
    Given a normalised VLM bounding box (x1,y1,x2,y2 in [0,1]),
    return the (crop_center_x, crop_center_y) normalised coordinates
    to use for cropping the source to the target aspect ratio.

    The crop center is the center of the bbox, clamped so the crop
    window stays within the source frame.
    """
    bbox_cx = (bbox_x1 + bbox_x2) / 2.0
    bbox_cy = (bbox_y1 + bbox_y2) / 2.0

    tgt_ratio = tgt_w / max(tgt_h, 1)
    src_ratio = src_w / max(src_h, 1)

    if src_ratio > tgt_ratio:
        # Source wider than target — horizontal crop
        crop_w_ratio = (src_h * tgt_ratio) / src_w
        # Clamp center so crop window stays inside [0, 1]
        half = crop_w_ratio / 2.0
        cx = max(half, min(1.0 - half, bbox_cx))
        cy = 0.5
    else:
        # Source taller than target — vertical crop
        crop_h_ratio = (src_w / tgt_ratio) / src_h
        half = crop_h_ratio / 2.0
        cx = 0.5
        cy = max(half, min(1.0 - half, bbox_cy))

    return cx, cy


def compute_saliency_crop(
    image_bytes: bytes,
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
) -> tuple[float, float] | None:
    """
    Compute the most salient region center using OpenCV's saliency detector.

    Returns (cx, cy) normalised to [0, 1], or None if OpenCV is unavailable
    or the saliency map is empty.
    """
    try:
        import cv2  # type: ignore[import]
        import numpy as np  # type: ignore[import]
        import io
    except ImportError:
        logger.warning("opencv_not_available_for_saliency")
        return None

    try:
        # Decode the image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None

        # Use StaticSaliencySpectralResidual
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(img)
        if not success or saliency_map is None:
            return None

        # Find the center of mass of the saliency map
        thresh = 0.5
        _, binary = cv2.threshold(
            (saliency_map * 255).astype("uint8"),
            int(thresh * 255), 255, cv2.THRESH_BINARY,
        )
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        # Largest contour center
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Normalise to [0, 1]
        h, w = img.shape[:2]
        return cx / max(w, 1), cy / max(h, 1)

    except Exception as exc:
        logger.warning("saliency_failed", extra={"error": str(exc)})
        return None


def resolve_crop_center(
    src_w: int,
    src_h: int,
    tgt_w: int,
    tgt_h: int,
    vlm_bbox: tuple[float, float, float, float] | None,
    thumb_bytes: bytes | None,
) -> tuple[float, float, str]:
    """
    Resolve the best crop center for reframing.

    Returns (cx, cy, method_name) where method_name is one of:
    "vlm_bbox", "saliency", "center".

    Priority: vlm_bbox > saliency > center.
    """
    s = get_settings()

    if vlm_bbox is not None:
        cx, cy = compute_crop_from_bbox(
            src_w, src_h, tgt_w, tgt_h, *vlm_bbox,
        )
        return cx, cy, "vlm_bbox"

    if s.RENDER_REFRAME_METHOD == "saliency" and thumb_bytes:
        result = compute_saliency_crop(
            thumb_bytes, src_w, src_h, tgt_w, tgt_h,
        )
        if result is not None:
            return result[0], result[1], "saliency"

    # Fallback: center crop
    return 0.5, 0.5, "center"
