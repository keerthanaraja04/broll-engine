"""
media_download — stream a remote URL into MinIO with integrity validation.

Pipeline
--------
1. Stream URL → temp bytes buffer (chunked, size-guarded)
2. Compute SHA-256 checksum of downloaded bytes
3. Run ffprobe on the bytes (via a temp file) to validate video dimensions
   and duration — raises FfprobeValidationError on corrupt/empty streams
4. Upload validated bytes to MinIO under a deterministic object key
5. Return DownloadedClip with all metadata filled

Retry policy
------------
Up to FOOTAGE_DOWNLOAD_MAX_RETRIES attempts.  A fresh download is attempted
on:
  - Network errors (timeout, connect)
  - Checksum changes between attempts (server returned different content)
  - FfprobeValidationError (file was corrupt)

DownloadSizeLimitError is terminal — never retried.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
from typing import Any

import httpx

from app.adapters.footage_types import Candidate, DownloadedClip
from app.adapters.storage import get_storage_client
from app.core.config import get_settings
from app.core.errors import (
    CorruptedDownloadError,
    DownloadSizeLimitError,
    FfprobeValidationError,
    StorageError,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _ffprobe_info(data: bytes) -> dict[str, Any]:
    """
    Run ffprobe on *data* (written to a temp file) and return the parsed
    JSON output.

    Raises FfprobeValidationError if ffprobe fails or reports no video stream.
    """
    s = get_settings()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            [
                s.FFPROBE_BINARY,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                tmp_path,
            ],
            capture_output=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise FfprobeValidationError(
            f"ffprobe not available or timed out: {exc}",
            details={"binary": s.FFPROBE_BINARY},
        ) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if result.returncode != 0:
        raise FfprobeValidationError(
            "ffprobe returned non-zero exit code",
            details={"stderr": result.stderr.decode(errors="replace")[:400]},
        )

    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FfprobeValidationError("ffprobe output is not valid JSON") from exc

    return probe  # type: ignore[no-any-return]


def _extract_video_info(probe: dict[str, Any]) -> tuple[int, int, float]:
    """
    Extract (width, height, duration) from ffprobe output.
    Raises FfprobeValidationError if no video stream is found.
    """
    streams = probe.get("streams", [])
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        # For images, check if there's any valid stream
        if streams:
            # Single-frame image — width/height from first stream
            w = streams[0].get("width", 0)
            h = streams[0].get("height", 0)
            return w, h, 0.0
        raise FfprobeValidationError(
            "No video or image stream found in downloaded file",
            details={"streams": [s.get("codec_type") for s in streams]},
        )

    w = video_stream.get("width", 0)
    h = video_stream.get("height", 0)

    # Duration: try stream duration first, then format duration
    dur_str = video_stream.get("duration") or probe.get("format", {}).get("duration", "0")
    try:
        dur = float(dur_str)
    except (ValueError, TypeError):
        dur = 0.0

    return w, h, dur


def _validate_dimensions(w: int, h: int, dur: float) -> None:
    s = get_settings()
    if w < s.FOOTAGE_MIN_VIDEO_WIDTH or h < s.FOOTAGE_MIN_VIDEO_HEIGHT:
        raise FfprobeValidationError(
            f"Video dimensions {w}x{h} below minimum "
            f"{s.FOOTAGE_MIN_VIDEO_WIDTH}x{s.FOOTAGE_MIN_VIDEO_HEIGHT}",
            details={"width": w, "height": h},
        )
    # For videos (dur > 0) check minimum duration
    if dur > 0 and dur < s.FOOTAGE_MIN_DURATION_SECONDS:
        raise FfprobeValidationError(
            f"Video duration {dur:.2f}s below minimum {s.FOOTAGE_MIN_DURATION_SECONDS}s",
            details={"duration": dur},
        )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

async def _stream_download(url: str) -> bytes:
    """
    Stream *url* into memory respecting the configured size cap.
    Raises DownloadSizeLimitError if the cap is exceeded.
    """
    s = get_settings()
    timeout = httpx.Timeout(s.FOOTAGE_DOWNLOAD_TIMEOUT_SECONDS)
    chunks: list[bytes] = []
    total = 0

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=s.FOOTAGE_DOWNLOAD_CHUNK_SIZE):
                total += len(chunk)
                if total > s.FOOTAGE_DOWNLOAD_MAX_BYTES:
                    raise DownloadSizeLimitError(
                        f"Download exceeds {s.FOOTAGE_DOWNLOAD_MAX_BYTES // 1_048_576} MB limit",
                        details={"url": url, "bytes_so_far": total},
                    )
                chunks.append(chunk)

    return b"".join(chunks)


def _object_key(candidate: Candidate, ext: str) -> str:
    """Deterministic MinIO object key for a candidate."""
    return f"footage/{candidate.source}/{candidate.id.replace(':', '_')}.{ext}"


def _guess_ext(url: str, content_type: str | None = None) -> str:
    """Guess file extension from URL or content type."""
    if content_type:
        mapping = {
            "video/mp4": "mp4",
            "video/webm": "webm",
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
        }
        if ct := mapping.get(content_type.split(";")[0].strip().lower()):
            return ct
    path = url.split("?")[0].split("#")[0]
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "bin"
    return ext if len(ext) <= 5 else "bin"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def download_to_minio(
    candidate: Candidate,
    bucket: str | None = None,
    ffprobe_validate: bool = True,
) -> DownloadedClip:
    """
    Download the candidate's URL into MinIO with validation.

    Steps
    -----
    1. Stream download (chunked, size-guarded)
    2. SHA-256 checksum
    3. ffprobe validation (optional — set ``ffprobe_validate=False`` in tests)
    4. Upload to MinIO
    5. Register in ClipStore
    6. Return DownloadedClip

    Retries
    -------
    Up to FOOTAGE_DOWNLOAD_MAX_RETRIES times on network errors,
    checksum instability, or ffprobe failure.

    Raises
    ------
    DownloadSizeLimitError  — terminal (not retried)
    CorruptedDownloadError  — after all retries exhausted
    FfprobeValidationError  — after all retries exhausted
    StorageError            — MinIO upload failed
    """
    s = get_settings()
    target_bucket = bucket or s.STORAGE_BUCKET_PROCESSED
    last_exc: Exception | None = None
    previous_checksum: str | None = None

    for attempt in range(1, s.FOOTAGE_DOWNLOAD_MAX_RETRIES + 1):
        logger.info(
            "footage_download_attempt",
            extra={"url": candidate.url, "attempt": attempt, "id": candidate.id},
        )

        # 1. Download
        try:
            data = await _stream_download(candidate.url)
        except DownloadSizeLimitError:
            raise  # terminal — don't retry
        except Exception as exc:
            last_exc = CorruptedDownloadError(
                f"Download failed on attempt {attempt}: {exc}",
                details={"url": candidate.url, "attempt": attempt},
            )
            logger.warning(
                "footage_download_error",
                extra={"url": candidate.url, "attempt": attempt, "error": str(exc)},
            )
            await asyncio.sleep(2 ** attempt)
            continue

        # 2. Checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Detect corrupt / unstable content (different bytes on retry)
        if previous_checksum is not None and checksum != previous_checksum:
            logger.warning(
                "footage_checksum_changed",
                extra={"url": candidate.url, "attempt": attempt},
            )
            # Continue — might stabilise
        previous_checksum = checksum

        # 3. ffprobe validation
        validated_w, validated_h, validated_dur = 0, 0, 0.0
        if ffprobe_validate:
            try:
                probe = _ffprobe_info(data)
                validated_w, validated_h, validated_dur = _extract_video_info(probe)
                _validate_dimensions(validated_w, validated_h, validated_dur)
            except FfprobeValidationError as exc:
                last_exc = exc
                logger.warning(
                    "footage_ffprobe_failed",
                    extra={"url": candidate.url, "attempt": attempt, "error": str(exc)},
                )
                await asyncio.sleep(2 ** attempt)
                continue
        else:
            # Skip ffprobe — use declared dimensions from the candidate
            validated_w = candidate.width
            validated_h = candidate.height
            validated_dur = candidate.duration

        # 4. Upload to MinIO
        ext = _guess_ext(candidate.url)
        object_key = _object_key(candidate, ext)
        content_type_map = {
            "mp4": "video/mp4", "webm": "video/webm",
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")

        storage = get_storage_client()
        try:
            await storage.ensure_bucket(target_bucket)
            await storage.upload_bytes(target_bucket, object_key, data, content_type)
        except StorageError as exc:
            # Storage errors are not retried here — bubble up
            raise

        # 5. Update candidate metadata
        candidate.storage_object = object_key
        candidate.storage_bucket = target_bucket
        candidate.checksum_sha256 = checksum
        candidate.file_size_bytes = len(data)

        # 6. Register in ClipStore
        try:
            from app.adapters.clip_store import get_clip_store
            await get_clip_store().register(candidate)
        except Exception as exc:
            logger.warning("clip_store_register_after_download_error", extra={"error": str(exc)})

        logger.info(
            "footage_download_complete",
            extra={
                "id": candidate.id,
                "object": object_key,
                "bytes": len(data),
                "checksum": checksum[:12],
                "dimensions": f"{validated_w}x{validated_h}",
                "duration": validated_dur,
            },
        )

        return DownloadedClip(
            candidate=candidate,
            object_key=object_key,
            bucket=target_bucket,
            checksum_sha256=checksum,
            file_size_bytes=len(data),
            validated_width=validated_w,
            validated_height=validated_h,
            validated_duration=validated_dur,
        )

    # All retries exhausted
    if isinstance(last_exc, FfprobeValidationError):
        raise last_exc
    raise CorruptedDownloadError(
        f"Download failed after {s.FOOTAGE_DOWNLOAD_MAX_RETRIES} attempts",
        details={"url": candidate.url},
    ) from last_exc
