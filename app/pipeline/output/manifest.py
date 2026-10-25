"""
manifest.json — job-level metadata: models, settings, timings, costs, licence summary.

JSON Schema validates the output shape in tests.
"""

from __future__ import annotations

import platform
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSON Schema for manifest.json
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "job_id", "app_version", "models", "settings",
        "seed", "stage_timings", "cost_estimate", "warnings",
        "licence_summary",
    ],
    "additionalProperties": False,
    "properties": {
        "job_id": {"type": "string"},
        "app_version": {"type": "string"},
        "models": {
            "type": "object",
            "required": ["stt", "llm", "vlm", "tts"],
            "additionalProperties": False,
            "properties": {
                "stt": {
                    "type": "object",
                    "required": ["provider", "model", "adapter"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "adapter": {"type": "string"},
                    },
                },
                "llm": {
                    "type": "object",
                    "required": ["provider", "model", "adapter"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "adapter": {"type": "string"},
                    },
                },
                "vlm": {
                    "type": "object",
                    "required": ["provider", "model", "adapter"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "adapter": {"type": "string"},
                    },
                },
                "tts": {
                    "type": "object",
                    "required": ["provider", "model", "adapter"],
                    "properties": {
                        "provider": {"type": "string"},
                        "model": {"type": "string"},
                        "adapter": {"type": "string"},
                    },
                },
            },
        },
        "settings": {
            "type": "object",
            "properties": {
                "aspect_ratios": {"type": "array", "items": {"type": "string"}},
                "target_fps": {"type": "integer"},
                "target_resolution": {"type": "string"},
                "scene_min_duration": {"type": "number"},
                "scene_max_duration": {"type": "number"},
                "clip_threshold_pixabay_video": {"type": "number"},
                "clip_threshold_ken_burns": {"type": "number"},
                "reframe_method": {"type": "string"},
                "brightness_match": {"type": "boolean"},
            },
        },
        "seed": {"type": ["string", "null"]},
        "stage_timings": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "cost_estimate": {
            "type": "object",
            "required": ["total_usd", "breakdown"],
            "properties": {
                "total_usd": {"type": "number", "minimum": 0},
                "breakdown": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "currency": {"type": "string"},
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
        "licence_summary": {
            "type": "object",
            "required": ["all_licensed", "all_attributed", "all_commercial_use",
                         "clips_checked"],
            "properties": {
                "all_licensed": {"type": "boolean"},
                "all_attributed": {"type": "boolean"},
                "all_commercial_use": {"type": "boolean"},
                "clips_checked": {"type": "integer", "minimum": 0},
                "licences_found": {"type": "array", "items": {"type": "string"}},
                "missing_licence": {"type": "array", "items": {"type": "string"}},
                "missing_attribution": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Cost estimation (rough, configurable)
# ---------------------------------------------------------------------------

# Approximate cost per call in USD (update as providers change pricing)
_COST_PER_CALL: dict[str, float] = {
    "stt_whisper": 0.004,      # Groq Whisper per minute of audio
    "llm_chat": 0.0002,        # Groq chat completion per 1K tokens
    "vlm_score": 0.001,        # Groq vision per image
    "tts": 0.005,              # Groq TTS per minute
    "pixabay_search": 0.0,    # free
    "download_minio": 0.0,    # self-hosted
    "ffmpeg_render": 0.0,     # self-hosted
}


def _estimate_costs(call_counts: dict[str, int] | None) -> dict[str, Any]:
    """Return cost estimate dict from call counts."""
    if call_counts is None:
        call_counts = {}
    breakdown: dict[str, float] = {}
    total = 0.0
    for key, count in call_counts.items():
        rate = _COST_PER_CALL.get(key, 0.0)
        cost = round(count * rate, 6)
        breakdown[key] = cost
        total += cost
    return {
        "total_usd": round(total, 6),
        "breakdown": breakdown,
        "currency": "USD",
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_manifest(
    job_id: str,
    timeline: dict[str, Any],
    stage_timings: dict[str, float] | None = None,
    call_counts: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    seed: str | None = None,
    aspect_ratios: list[str] | None = None,
) -> dict[str, Any]:
    """
    Build the manifest.json dict.

    Args:
        job_id:         Job UUID.
        timeline:       The built timeline dict (for licence summary).
        stage_timings:  Per-stage execution times in seconds.
        call_counts:    Per-API-call counts for cost estimation.
        warnings:       Accumulated warnings from all stages.
        seed:           Optional random seed used.
        aspect_ratios:  Aspect ratios requested for this job.

    Returns:
        Dict conforming to MANIFEST_SCHEMA.
    """
    s = get_settings()

    # Build licence summary from timeline
    licence_summary = _build_licence_summary(timeline)

    # Build models dict
    models = {
        "stt": {"provider": "groq", "model": s.STT_MODEL, "adapter": s.STT_ADAPTER},
        "llm": {"provider": "groq", "model": s.LLM_JSON_MODEL, "adapter": s.LLM_ADAPTER},
        "vlm": {"provider": "groq", "model": s.VLM_MODEL, "adapter": s.VLM_ADAPTER},
        "tts": {"provider": "groq", "model": s.TTS_MODEL, "adapter": s.TTS_ADAPTER},
    }

    # Build settings dict
    settings_dict = {
        "aspect_ratios": aspect_ratios or ["16:9"],
        "target_fps": s.RENDER_FPS,
        "target_resolution": f"{s.RENDER_WIDTH_16_9}x{s.RENDER_HEIGHT_16_9}",
        "scene_min_duration": s.SCENE_MIN_DURATION_SECONDS,
        "scene_max_duration": s.SCENE_MAX_DURATION_SECONDS,
        "clip_threshold_pixabay_video": s.CLIP_THRESHOLD_PIXABAY_VIDEO,
        "clip_threshold_ken_burns": s.CLIP_THRESHOLD_KEN_BURNS,
        "reframe_method": s.RENDER_REFRAME_METHOD,
        "brightness_match": s.RENDER_BRIGHTNESS_MATCH,
    }

    cost_estimate = _estimate_costs(call_counts)

    manifest = {
        "job_id": job_id,
        "app_version": s.APP_VERSION,
        "models": models,
        "settings": settings_dict,
        "seed": seed,
        "stage_timings": stage_timings or {},
        "cost_estimate": cost_estimate,
        "warnings": warnings or [],
        "licence_summary": licence_summary,
    }

    logger.info(
        "manifest_built",
        extra={"job_id": job_id, "cost_total": cost_estimate["total_usd"]},
    )
    return manifest


def _build_licence_summary(timeline: dict[str, Any]) -> dict[str, Any]:
    """Scan the timeline for licence/attribution/commercial_use status."""
    licences_found: set[str] = set()
    missing_licence: list[str] = []
    missing_attribution: list[str] = []
    all_commercial = True
    clips_checked = 0

    for scene in timeline.get("scenes", []):
        clip = scene.get("clip")
        if clip is None:
            continue
        clips_checked += 1
        clip_id = clip.get("id", "unknown")

        lic = clip.get("license", "")
        if not lic or lic.strip() == "":
            missing_licence.append(clip_id)
        else:
            licences_found.add(lic)

        attr = clip.get("attribution", "")
        if not attr or attr.strip() == "":
            missing_attribution.append(clip_id)

        if not clip.get("commercial_use", True):
            all_commercial = False

    return {
        "all_licensed": len(missing_licence) == 0,
        "all_attributed": len(missing_attribution) == 0,
        "all_commercial_use": all_commercial,
        "clips_checked": clips_checked,
        "licences_found": sorted(licences_found),
        "missing_licence": missing_licence,
        "missing_attribution": missing_attribution,
    }
