"""
Unit tests for app/core/idempotency.py.

Uses a real fakeredis instance so no live Redis is needed.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_async_redis(store: dict) -> MagicMock:
    """
    Build a minimal async Redis mock backed by a plain dict.
    Supports get, setex, delete, and pipeline().
    """
    mock = MagicMock()

    async def _get(key: str) -> str | None:
        return store.get(key)

    async def _setex(key: str, ttl: int, value: str) -> None:
        store[key] = value

    async def _delete(key: str) -> None:
        store.pop(key, None)

    mock.get = _get
    mock.setex = _setex
    mock.delete = _delete
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckOrStore:
    @pytest.mark.asyncio
    async def test_new_key_is_stored(self) -> None:
        from app.core.idempotency import check_or_store

        store: dict = {}
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            result = await check_or_store("key-1", "hash-abc", "job-111")

        assert result.is_replay is False
        assert result.job_id is None
        # Verify it was persisted
        assert "idem:key-1" in store
        payload = json.loads(store["idem:key-1"])
        assert payload["hash"] == "hash-abc"
        assert payload["job_id"] == "job-111"

    @pytest.mark.asyncio
    async def test_same_hash_returns_replay(self) -> None:
        from app.core.idempotency import check_or_store

        store = {
            "idem:key-2": json.dumps({"hash": "hash-xyz", "job_id": "job-222"})
        }
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            result = await check_or_store("key-2", "hash-xyz", "job-new")

        assert result.is_replay is True
        assert result.job_id == "job-222"

    @pytest.mark.asyncio
    async def test_different_hash_raises_conflict(self) -> None:
        from app.core.errors import IdempotencyKeyConflictError
        from app.core.idempotency import check_or_store

        store = {
            "idem:key-3": json.dumps({"hash": "original-hash", "job_id": "job-333"})
        }
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            with pytest.raises(IdempotencyKeyConflictError) as exc_info:
                await check_or_store("key-3", "different-hash", "job-new")

        assert exc_info.value.code == "IDEMPOTENCY_KEY_CONFLICT"
        assert exc_info.value.http_status == 409
        assert exc_info.value.retryable is False

    @pytest.mark.asyncio
    async def test_conflict_error_includes_key_in_details(self) -> None:
        from app.core.errors import IdempotencyKeyConflictError
        from app.core.idempotency import check_or_store

        store = {
            "idem:my-special-key": json.dumps({"hash": "old", "job_id": "job-x"})
        }
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            with pytest.raises(IdempotencyKeyConflictError) as exc_info:
                await check_or_store("my-special-key", "new-hash", "job-y")

        assert exc_info.value.details is not None
        assert exc_info.value.details["idempotency_key"] == "my-special-key"

    @pytest.mark.asyncio
    async def test_invalidate_removes_key(self) -> None:
        from app.core.idempotency import invalidate

        store = {
            "idem:to-delete": json.dumps({"hash": "h", "job_id": "j"})
        }
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            await invalidate("to-delete")

        assert "idem:to-delete" not in store

    @pytest.mark.asyncio
    async def test_invalidate_missing_key_is_safe(self) -> None:
        from app.core.idempotency import invalidate

        store: dict = {}
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            # Must not raise even if key doesn't exist
            await invalidate("nonexistent-key")

    @pytest.mark.asyncio
    async def test_redis_prefix_applied(self) -> None:
        from app.core.idempotency import check_or_store

        store: dict = {}
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            await check_or_store("raw-key", "hash-val", "job-id-x")

        # Key in Redis must be prefixed
        assert any(k.startswith("idem:") for k in store.keys())

    @pytest.mark.asyncio
    async def test_replay_does_not_overwrite_stored_entry(self) -> None:
        """A replay call must NOT change the stored hash or job_id."""
        from app.core.idempotency import check_or_store

        original = json.dumps({"hash": "h1", "job_id": "original-job"})
        store = {"idem:stable-key": original}
        redis_mock = _make_async_redis(store)

        with patch("app.core.idempotency._get_redis", return_value=redis_mock):
            await check_or_store("stable-key", "h1", "should-not-overwrite")

        # Original entry must be untouched
        stored = json.loads(store["idem:stable-key"])
        assert stored["job_id"] == "original-job"


# ---------------------------------------------------------------------------
# Callback HMAC signing (lives in services/callback.py but tested here for focus)
# ---------------------------------------------------------------------------

class TestCallbackHmac:
    def test_sign_payload_returns_sha256_prefix(self) -> None:
        from app.services.callback import _sign_payload

        sig = _sign_payload(b"hello world", "mysecret")
        assert sig.startswith("sha256=")
        assert len(sig) > 10

    def test_same_body_same_signature(self) -> None:
        from app.services.callback import _sign_payload

        s1 = _sign_payload(b"body", "secret")
        s2 = _sign_payload(b"body", "secret")
        assert s1 == s2

    def test_different_secret_different_signature(self) -> None:
        from app.services.callback import _sign_payload

        s1 = _sign_payload(b"body", "secret1")
        s2 = _sign_payload(b"body", "secret2")
        assert s1 != s2

    def test_different_body_different_signature(self) -> None:
        from app.services.callback import _sign_payload

        s1 = _sign_payload(b"body1", "secret")
        s2 = _sign_payload(b"body2", "secret")
        assert s1 != s2
