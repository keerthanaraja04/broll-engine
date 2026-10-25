"""
Unit tests for media_download.download_to_minio().

Tests: successful download+upload, ffprobe validation, size limit,
corrupted download retry, checksum, MinIO upload, ClipStore registration.
All I/O mocked — no real network, no ffprobe, no MinIO.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")

from tests.fixtures.conftest_footage import (
    FAKE_MP4_BYTES, make_video_candidate,
)


class TestSuccessfulDownload:
    @pytest.mark.asyncio
    async def test_download_returns_downloaded_clip(
        self, mock_storage: Any, mock_ffprobe_ok: Any,
        mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)):
            result = await download_to_minio(
                c, ffprobe_validate=True)
        assert result.object_key.startswith("footage/pixabay/")
        assert result.checksum_sha256 == hashlib.sha256(
            FAKE_MP4_BYTES).hexdigest()
        assert result.file_size_bytes == len(FAKE_MP4_BYTES)

    @pytest.mark.asyncio
    async def test_download_without_ffprobe(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)):
            result = await download_to_minio(
                c, ffprobe_validate=False)
        assert result.validated_width == c.width
        assert result.validated_height == c.height

    @pytest.mark.asyncio
    async def test_candidate_metadata_updated(
        self, mock_storage: Any, mock_ffprobe_ok: Any,
        mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)):
            await download_to_minio(c, ffprobe_validate=True)
        assert c.storage_object is not None
        assert c.checksum_sha256 is not None
        assert c.file_size_bytes > 0


class TestSizeLimit:
    @pytest.mark.asyncio
    async def test_size_limit_is_terminal(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        from app.core.errors import DownloadSizeLimitError
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(side_effect=DownloadSizeLimitError(
                       "too big"))):
            with pytest.raises(DownloadSizeLimitError):
                await download_to_minio(c, ffprobe_validate=False)
        assert c.storage_object is None

    def test_download_size_limit_error_code(self) -> None:
        from app.core.errors import DownloadSizeLimitError
        err = DownloadSizeLimitError("big")
        assert err.code == "DOWNLOAD_SIZE_LIMIT"
        assert err.retryable is False


class TestFfprobeValidation:
    @pytest.mark.asyncio
    async def test_ffprobe_fail_triggers_retry_then_success(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        from app.core.errors import FfprobeValidationError
        c = make_video_candidate()
        call_count = 0

        good_probe = {"streams": [{"codec_type": "video", "width": 1920,
                                    "height": 1080, "duration": "10.0"}],
                       "format": {"duration": "10.0"}}

        async def _fake_stream(*args: Any, **kwargs: Any) -> bytes:
            return FAKE_MP4_BYTES

        def _flaky_ffprobe(data: bytes) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise FfprobeValidationError("corrupt")
            return good_probe

        with patch("app.adapters.media_download._stream_download",
                   _fake_stream), \
             patch("app.adapters.media_download._ffprobe_info",
                   side_effect=_flaky_ffprobe), \
             patch("app.adapters.media_download.asyncio.sleep",
                   AsyncMock()):
            result = await download_to_minio(c, ffprobe_validate=True)
        assert call_count == 2
        assert result.object_key is not None

    @pytest.mark.asyncio
    async def test_ffprobe_fail_all_retries_raises(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        from app.core.errors import FfprobeValidationError
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)), \
             patch("app.adapters.media_download._ffprobe_info",
                   side_effect=FfprobeValidationError("corrupt")), \
             patch("app.adapters.media_download.asyncio.sleep",
                   AsyncMock()):
            with pytest.raises(FfprobeValidationError):
                await download_to_minio(c, ffprobe_validate=True)


class TestCorruptedDownload:
    @pytest.mark.asyncio
    async def test_network_error_triggers_retry(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        call_count = 0

        async def _flaky_stream(*args: Any, **kwargs: Any) -> bytes:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("network down")
            return FAKE_MP4_BYTES

        with patch("app.adapters.media_download._stream_download",
                   _flaky_stream), \
             patch("app.adapters.media_download.asyncio.sleep",
                   AsyncMock()):
            result = await download_to_minio(
                c, ffprobe_validate=False)
        assert call_count == 2
        assert result is not None

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_raises_corrupted(
        self, mock_storage: Any, mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        from app.core.errors import CorruptedDownloadError
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(side_effect=ConnectionError("down")), \
             patch("app.adapters.media_download.asyncio.sleep",
                   AsyncMock())):
            with pytest.raises(CorruptedDownloadError):
                await download_to_minio(c, ffprobe_validate=False)

    def test_corrupted_download_error_code(self) -> None:
        from app.core.errors import CorruptedDownloadError
        err = CorruptedDownloadError("bad")
        assert err.code == "CORRUPTED_DOWNLOAD"
        assert err.retryable is True


class TestUploadToMinio:
    @pytest.mark.asyncio
    async def test_upload_called_with_correct_args(
        self, mock_storage: Any, mock_ffprobe_ok: Any,
        mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)):
            await download_to_minio(c, ffprobe_validate=True)
        mock_storage.upload_bytes.assert_called_once()
        args = mock_storage.upload_bytes.call_args
        assert args[0][2] == FAKE_MP4_BYTES

    @pytest.mark.asyncio
    async def test_bucket_ensured(
        self, mock_storage: Any, mock_ffprobe_ok: Any,
        mock_clip_store_empty: Any
    ) -> None:
        from app.adapters.media_download import download_to_minio
        c = make_video_candidate()
        with patch("app.adapters.media_download._stream_download",
                   AsyncMock(return_value=FAKE_MP4_BYTES)):
            await download_to_minio(c, ffprobe_validate=True)
        mock_storage.ensure_bucket.assert_called_once()


class TestObjectKey:
    def test_object_key_format(self) -> None:
        from app.adapters.media_download import _object_key
        c = make_video_candidate(cid="pixabay:12345")
        key = _object_key(c, "mp4")
        assert "pixabay_12345" in key
        assert key.endswith(".mp4")

    def test_guess_ext_from_url(self) -> None:
        from app.adapters.media_download import _guess_ext
        assert _guess_ext("https://example.com/clip.mp4") == "mp4"
        assert _guess_ext("https://example.com/clip.wav") == "wav"
        assert _guess_ext("https://example.com/clip") == "bin"

    def test_guess_ext_from_content_type(self) -> None:
        from app.adapters.media_download import _guess_ext
        assert _guess_ext("https://x.com/a", "video/mp4") == "mp4"
        assert _guess_ext("https://x.com/a", "image/jpeg") == "jpg"
