"""
InputStageStep — the first step in every pipeline.

Reads from PipelineContext.payload and dispatches to the correct processor:

    payload["input_type"] == "audio_upload" or "audio_url"
        → audio_processor.process_audio()
        Required keys: audio_object | audio_url
        Optional keys: language

    payload["input_type"] == "script"
        → text_processor.process_script()
        Required keys: script
        Optional keys: voice, language

    payload["input_type"] == "scenes"
        → text_processor.process_scenes()
        Required keys: scenes
        Optional keys: voice, language

Output
------
ctx.set("transcript", transcript.to_dict())
ctx.set("transcript_obj", transcript)       # in-memory object for next steps
ctx.set("input_language", transcript.language)
ctx.set("input_duration", transcript.duration)
ctx.set("input_warnings", transcript.warnings)
"""

from __future__ import annotations

from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.errors import PipelineError, ValidationError
from app.core.logging import get_logger
from app.pipeline.base import PipelineContext, Step
from app.pipeline.input_stage.audio_processor import process_audio
from app.pipeline.input_stage.text_processor import process_scenes, process_script
from app.pipeline.input_stage.transcript import Transcript

logger = get_logger(__name__)


class InputStageStep(Step):
    """
    Entry point for every job.  Validates the input type, runs the
    appropriate processor, and stores the Transcript on the context.

    Raises PipelineError (wrapping any domain error) on failure.
    """

    async def execute(self, ctx: PipelineContext) -> PipelineContext:
        payload = ctx.payload
        input_type: str = payload.get("input_type", "").lower()

        logger.info("input_stage_start", extra={"input_type": input_type})

        try:
            transcript = await self._dispatch(input_type, payload)
        except (PipelineError, ValidationError):
            raise
        except Exception as exc:
            # Wrap unexpected exceptions with a PipelineError so the outer
            # runner always sees a typed error with a code.
            raise PipelineError(
                f"Input stage failed: {exc}",
                details={"input_type": input_type, "error": str(exc)},
            ) from exc

        # Publish to context
        ctx.set("transcript", transcript.to_dict())
        ctx.set("transcript_obj", transcript)
        ctx.set("input_language", transcript.language)
        ctx.set("input_duration", transcript.duration)
        ctx.set("input_warnings", transcript.warnings)

        logger.info(
            "input_stage_complete",
            extra={
                "source": transcript.source,
                "language": transcript.language,
                "duration": transcript.duration,
                "word_count": transcript.word_count,
                "warnings": len(transcript.warnings),
            },
        )
        return ctx

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self, input_type: str, payload: dict
    ) -> Transcript:
        s = get_settings()

        if input_type in ("audio_upload", "audio_url"):
            return await self._handle_audio(input_type, payload)

        if input_type == "script":
            script = payload.get("script", "")
            if not isinstance(script, str):
                raise ValidationError(
                    "payload 'script' must be a string",
                    details={"got": type(script).__name__},
                )
            return await process_script(
                script=script,
                voice=payload.get("voice"),
                language=payload.get("language"),
            )

        if input_type == "scenes":
            scenes = payload.get("scenes")
            if not isinstance(scenes, list) or not scenes:
                raise ValidationError(
                    "payload 'scenes' must be a non-empty list",
                    details={"got": type(scenes).__name__ if scenes is not None else "None"},
                )
            return await process_scenes(
                scenes=scenes,
                voice=payload.get("voice"),
                language=payload.get("language"),
            )

        raise ValidationError(
            f"Unknown input_type '{input_type}'. "
            "Expected: audio_upload | audio_url | script | scenes",
            details={"input_type": input_type},
        )

    async def _handle_audio(self, input_type: str, payload: dict) -> Transcript:
        import tempfile
        import os

        s = get_settings()
        tmp_dir = s.AUDIO_TEMP_DIR or tempfile.gettempdir()

        if input_type == "audio_upload":
            # Object stored in MinIO — download to temp file first
            audio_object = payload.get("audio_object")
            if not audio_object:
                raise ValidationError(
                    "payload 'audio_object' is required for audio_upload",
                )
            audio_bucket = payload.get("audio_bucket", s.STORAGE_BUCKET_RAW)

            storage = get_storage_client()
            audio_bytes = await storage.download_bytes(audio_bucket, audio_object)

            # Determine extension from object key
            ext = audio_object.rsplit(".", 1)[-1].lower() if "." in audio_object else "bin"
            tmp_path = os.path.join(tmp_dir, f"upload_{hash(audio_object) & 0xFFFFFF}.{ext}")
            with open(tmp_path, "wb") as f:
                f.write(audio_bytes)
            audio_path = tmp_path

        else:  # audio_url
            audio_url = payload.get("audio_url")
            if not audio_url:
                raise ValidationError(
                    "payload 'audio_url' is required for audio_url input",
                )
            audio_path = await self._download_url(str(audio_url), tmp_dir)

        return await process_audio(
            audio_path=audio_path,
            language=payload.get("language"),
            output_dir=tmp_dir,
        )

    @staticmethod
    async def _download_url(url: str, tmp_dir: str) -> str:
        """Download a URL to a temp file and return the local path."""
        import os
        import httpx
        from app.core.errors import MediaProcessingError

        ext = url.split("?")[0].rsplit(".", 1)[-1].lower() if "." in url else "bin"
        ext = ext if len(ext) <= 5 else "bin"
        tmp_path = os.path.join(tmp_dir, f"url_{abs(hash(url)) & 0xFFFFFF}.{ext}")

        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(tmp_path, "wb") as f:
                        async for chunk in response.aiter_bytes(1_048_576):
                            f.write(chunk)
        except Exception as exc:
            raise MediaProcessingError(
                f"Failed to download audio from URL: {exc}",
                details={"url": url},
            ) from exc

        return tmp_path
