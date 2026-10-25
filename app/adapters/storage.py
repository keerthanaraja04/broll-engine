"""
Object-storage adapter (MinIO / S3-compatible).

Wraps the minio-py SDK with async-friendly helpers using run_in_executor.
"""

from __future__ import annotations

import asyncio
import io
from functools import partial
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import get_settings
from app.core.errors import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)


class StorageClient:
    """Thin async wrapper around the synchronous Minio client."""

    def __init__(self) -> None:
        s = get_settings()
        self._client = Minio(
            s.STORAGE_ENDPOINT,
            access_key=s.STORAGE_ACCESS_KEY,
            secret_key=s.STORAGE_SECRET_KEY,
            secure=s.STORAGE_USE_SSL,
        )
        self._settings = s

    # ------------------------------------------------------------------
    # Bucket management
    # ------------------------------------------------------------------
    async def ensure_bucket(self, bucket: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            exists = await loop.run_in_executor(None, self._client.bucket_exists, bucket)
            if not exists:
                await loop.run_in_executor(None, self._client.make_bucket, bucket)
                logger.info("storage_bucket_created", extra={"bucket": bucket})
        except S3Error as exc:
            raise StorageError(f"Failed to ensure bucket '{bucket}'", details={"error": str(exc)}) from exc

    async def ensure_all_buckets(self) -> None:
        s = self._settings
        for bucket in (s.STORAGE_BUCKET_RAW, s.STORAGE_BUCKET_PROCESSED, s.STORAGE_BUCKET_FINAL):
            await self.ensure_bucket(bucket)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------
    async def upload_bytes(
        self,
        bucket: str,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload raw bytes; returns the object name."""
        loop = asyncio.get_running_loop()
        stream = io.BytesIO(data)
        fn = partial(
            self._client.put_object,
            bucket,
            object_name,
            stream,
            length=len(data),
            content_type=content_type,
        )
        try:
            await loop.run_in_executor(None, fn)
        except S3Error as exc:
            raise StorageError(f"Upload failed for '{object_name}'", details={"error": str(exc)}) from exc
        logger.debug("storage_upload_ok", extra={"bucket": bucket, "object": object_name, "bytes": len(data)})
        return object_name

    async def upload_file(
        self,
        bucket: str,
        object_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload from a local file path."""
        loop = asyncio.get_running_loop()
        fn = partial(self._client.fput_object, bucket, object_name, file_path, content_type)
        try:
            await loop.run_in_executor(None, fn)
        except S3Error as exc:
            raise StorageError(f"File upload failed for '{file_path}'", details={"error": str(exc)}) from exc
        return object_name

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    async def download_bytes(self, bucket: str, object_name: str) -> bytes:
        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, partial(self._client.get_object, bucket, object_name))
            data = response.read()
            response.close()
            response.release_conn()
            return data  # type: ignore[return-value]
        except S3Error as exc:
            raise StorageError(f"Download failed for '{object_name}'", details={"error": str(exc)}) from exc

    # ------------------------------------------------------------------
    # Presigned URLs
    # ------------------------------------------------------------------
    async def presigned_get_url(self, bucket: str, object_name: str) -> str:
        from datetime import timedelta

        loop = asyncio.get_running_loop()
        expiry = timedelta(seconds=self._settings.STORAGE_PRESIGNED_URL_EXPIRY_SECONDS)
        fn = partial(self._client.presigned_get_object, bucket, object_name, expires=expiry)
        try:
            url: str = await loop.run_in_executor(None, fn)
            return url
        except S3Error as exc:
            raise StorageError("Presigned URL generation failed", details={"error": str(exc)}) from exc

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------
    async def delete_object(self, bucket: str, object_name: str) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, partial(self._client.remove_object, bucket, object_name))
        except S3Error as exc:
            raise StorageError(f"Delete failed for '{object_name}'", details={"error": str(exc)}) from exc


# Singleton
_storage: StorageClient | None = None


def get_storage_client() -> StorageClient:
    global _storage
    if _storage is None:
        _storage = StorageClient()
    return _storage
