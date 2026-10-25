"""
RenderStep — the pipeline step that renders the final video.

Reads from PipelineContext:
    ctx.get("video_selection_plan_obj")  — VideoSelectionPlan (required)
    ctx.get("transcript_obj")            — Transcript (for captions)
    ctx.get("input_duration")            — total audio duration
    ctx.payload["aspect_ratios"]          — list of aspect ratio strings
    ctx.payload["captions"]               — {"burn_in": bool, "srt": bool}
    ctx.payload["voice"]                  — TTS voice (if text path)

Writes to PipelineContext:
    ctx.set("render_results", [dict, ...])   — one per aspect ratio
    ctx.set("render_result_objs", [RenderResult, ...])
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any

from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.rendering.assemble import (
    apply_crossfades,
    burn_in_captions,
    mix_narration_audio,
)
from app.pipeline.rendering.captions import build_subtitle_filter, generate_srt
from app.pipeline.rendering.ff_helpers import get_duration, target_dimensions
from app.pipeline.rendering.ken_burns_render import render_ken_burns
from app.pipeline.rendering.models import (
    AspectRatio,
    RenderResult,
    RenderedSegment,
)
from app.pipeline.rendering.normalise import normalise_clip
from app.pipeline.rendering.post_check import run_post_checks
from app.pipeline.rendering.reframe import get_reframe_method, resolve_crop_center
from app.pipeline.rendering.scene_cache import get_cached, put_cached
from app.pipeline.rendering.upload import upload_render_outputs

logger = get_logger(__name__)


class RenderStep(Step):
    """
    Render the final video for each requested aspect ratio.
    """

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        plan = ctx.get("video_selection_plan_obj")
        if plan is None:
            raise ValidationError(
                "RenderStep requires 'video_selection_plan_obj' — "
                "run ClipSelectionStep first"
            )

        transcript = ctx.get("transcript_obj")
        aspect_ratios: list[str] = ctx.payload.get("aspect_ratios", ["16:9"])
        captions_config: dict = ctx.payload.get("captions") or {}
        burn_in = captions_config.get("burn_in", False)
        srt_enabled = captions_config.get("srt", False)

        # Prepare narration audio path
        narration_path = self._get_narration_path(ctx)

        results: list[RenderResult] = []
        for ratio in aspect_ratios:
            result = await self._render_one_ratio(
                ctx.job_id, plan, ratio, transcript,
                narration_path, burn_in, srt_enabled,
            )
            results.append(result)

        # Upload all outputs
        uploaded: list[RenderResult] = []
        for r in results:
            r = await upload_render_outputs(r)
            uploaded.append(r)

        ctx.set("render_results", [r.to_dict() for r in uploaded])
        ctx.set("render_result_objs", uploaded)

        logger.info(
            "render_complete",
            extra={
                "job_id": ctx.job_id,
                "ratios": aspect_ratios,
                "results": len(uploaded),
            },
        )
        return ctx

    # ------------------------------------------------------------------
    # Per-ratio rendering
    # ------------------------------------------------------------------

    async def _render_one_ratio(
        self,
        job_id: str,
        plan: Any,
        aspect_ratio: str,
        transcript: Any,
        narration_path: str | None,
        burn_in: bool,
        srt_enabled: bool,
    ) -> RenderResult:
        s = get_settings()
        tmp_dir = s.RENDER_SCENE_CACHE_DIR or tempfile.gettempdir()
        tgt_w, tgt_h = target_dimensions(aspect_ratio)

        result = RenderResult(
            job_id=job_id,
            aspect_ratio=aspect_ratio,
        )

        # ── 1. Normalise each scene segment ───────────────────────────
        segments: list[RenderedSegment] = []
        for scene_result in plan.scene_results:
            if scene_result.chosen is None:
                result.warnings.append(
                    f"Scene {scene_result.scene_id}: no clip chosen — skipping"
                )
                continue

            segment = await self._normalise_scene(
                job_id, scene_result, aspect_ratio, tmp_dir,
            )
            segments.append(segment)

        if not segments:
            raise PipelineError(
                f"No renderable segments for aspect ratio {aspect_ratio}"
            )

        result.segments = segments
        result.total_duration = sum(seg.duration for seg in segments)

        # ── 2. Concatenate with crossfades ────────────────────────────
        try:
            video_path = await apply_crossfades(segments, output_dir=tmp_dir)
        except Exception as exc:
            # Fallback: simple concat without crossfades
            logger.warning("crossfade_failed_using_concat", extra={"error": str(exc)})
            from app.pipeline.rendering.assemble import concatenate_segments
            video_path = await concatenate_segments(segments, output_dir=tmp_dir)

        # ── 3. Mix narration audio ─────────────────────────────────────
        video_with_audio = await mix_narration_audio(
            video_path, narration_path, result.total_duration,
            output_dir=tmp_dir,
        )

        # ── 4. Generate SRT ─────────────────────────────────────────────
        if srt_enabled and transcript:
            words = transcript.words
            result.srt_path = generate_srt(words, output_path=os.path.join(
                tmp_dir, f"{job_id}_{aspect_ratio.replace(':', 'x')}.srt"
            ))

        # ── 5. Burn-in captions ─────────────────────────────────────────
        final_path = video_with_audio
        if burn_in and result.srt_path:
            sub_filter = build_subtitle_filter(
                result.srt_path,
                language=transcript.language if transcript else "en",
                video_width=tgt_w,
                video_height=tgt_h,
            )
            final_path = await burn_in_captions(
                video_with_audio, sub_filter, output_dir=tmp_dir,
            )

        result.final_video_path = final_path

        # ── 6. Post-render checks ──────────────────────────────────────
        post_result = await run_post_checks(final_path, tgt_w, tgt_h, s.RENDER_FPS)
        result.post_check_passed = post_result.passed
        result.post_check_errors = post_result.errors
        if not post_result.passed:
            result.warnings.append(
                f"Post-render checks failed: {'; '.join(post_result.errors[:3])}"
            )

        return result

    # ------------------------------------------------------------------
    # Normalise one scene
    # ------------------------------------------------------------------

    async def _normalise_scene(
        self,
        job_id: str,
        scene_result: Any,
        aspect_ratio: str,
        tmp_dir: str,
    ) -> RenderedSegment:
        s = get_settings()
        chosen = scene_result.chosen
        candidate = chosen.candidate
        scene_duration = scene_result.scene_duration

        # Check scene cache first
        cached_path = get_cached(
            job_id, scene_result.scene_id, candidate.id, aspect_ratio,
        )
        if cached_path:
            return RenderedSegment(
                scene_id=scene_result.scene_id,
                local_path=cached_path,
                duration=scene_duration,
                width=target_dimensions(aspect_ratio)[0],
                height=target_dimensions(aspect_ratio)[1],
                fps=s.RENDER_FPS,
                cached=True,
            )

        # Download clip from MinIO to local temp
        local_clip_path = await self._download_clip(candidate, tmp_dir)

        # Determine clip duration
        clip_duration = candidate.duration or await get_duration(local_clip_path)

        # Determine reframe method
        reframe = get_reframe_method(
            has_vlm_bbox=(chosen.vlm_score > 0 and bool(chosen.vlm_flags)),
            aspect_ratio=aspect_ratio,
        )

        # Compute crop center for non-16:9
        crop_x, crop_y, reframe_used = None, None, "center"
        if aspect_ratio != "16:9":
            # If VLM gave us a bbox, use it; else saliency; else center
            crop_x, crop_y, reframe_used = resolve_crop_center(
                src_w=candidate.width or 1920,
                src_h=candidate.height or 1080,
                tgt_w=target_dimensions(aspect_ratio)[0],
                tgt_h=target_dimensions(aspect_ratio)[1],
                vlm_bbox=None,  # TODO: extract from VLM score result
                thumb_bytes=None,
            )

        # Handle different clip types
        if candidate.source == "ken_burns" or candidate.kind.value == "image":
            # Ken Burns for image clips
            norm_path = await render_ken_burns(
                local_clip_path, scene_duration, aspect_ratio, tmp_dir,
            )
            reframe_used = "ken_burns"
        elif candidate.source == "ffmpeg_card" or candidate.kind.value == "card":
            # Title cards are already at the right format — just trim if needed
            norm_path = await normalise_clip(
                local_clip_path, scene_duration, aspect_ratio,
                clip_duration, candidate.width, candidate.height,
                output_dir=tmp_dir,
            )
        else:
            # Regular video clip
            norm_path = await normalise_clip(
                local_clip_path, scene_duration, aspect_ratio,
                clip_duration, candidate.width, candidate.height,
                needs_slow=chosen.needs_slow,
                needs_loop=chosen.needs_loop,
                reframe_method=reframe_used,
                crop_x=crop_x,
                crop_y=crop_y,
                output_dir=tmp_dir,
            )

        # Cache the normalised segment
        cached = put_cached(
            norm_path, job_id, scene_result.scene_id, candidate.id, aspect_ratio,
        )

        return RenderedSegment(
            scene_id=scene_result.scene_id,
            local_path=cached,
            duration=scene_duration,
            width=target_dimensions(aspect_ratio)[0],
            height=target_dimensions(aspect_ratio)[1],
            fps=s.RENDER_FPS,
            reframe_method=reframe_used,
            brightness_adjusted=s.RENDER_BRIGHTNESS_MATCH,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _download_clip(self, candidate: Any, tmp_dir: str) -> str:
        """Download a clip from MinIO to a local temp path."""
        s = get_settings()
        if candidate.storage_object and candidate.storage_bucket:
            storage = get_storage_client()
            data = await storage.download_bytes(
                candidate.storage_bucket, candidate.storage_object,
            )
            ext = ".mp4"
            if candidate.kind.value == "image":
                ext = ".jpg"
            local_path = os.path.join(
                tmp_dir, f"clip_{candidate.id.replace(':', '_')}{ext}",
            )
            with open(local_path, "wb") as f:
                f.write(data)
            return local_path

        # If no storage object, assume local path
        if candidate.url and os.path.isfile(candidate.url):
            return candidate.url

        raise PipelineError(
            f"Cannot download clip {candidate.id}: no storage_object and not a local path",
        )

    def _get_narration_path(self, ctx: PipelineContext) -> str | None:
        """
        Get the path to the narration audio file for mixing.

        For audio-path jobs: the converted 16 kHz WAV.
        For text-path jobs: the TTS audio needs to be written to a temp file.
        """
        transcript = ctx.get("transcript_obj")
        if transcript is None:
            return None

        if transcript.audio_path:
            return transcript.audio_path

        if transcript.tts_audio_bytes:
            import tempfile
            fd, path = tempfile.mkstemp(suffix=".wav")
            with os.fdopen(fd, "wb") as f:
                f.write(transcript.tts_audio_bytes)
            return path

        return None
