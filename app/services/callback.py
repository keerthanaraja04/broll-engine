"""
Webhook callback service.

When a job finishes (success or failure) and the caller supplied a
``callback_url``, this module fires a POST request to that URL carrying
the job payload.

Security
--------
If ``CALLBACK_HMAC_SECRET`` is set, the request body is signed with
HMAC-SHA256 and the signature is sent in the
``X-IncuBrix-Signature: sha256=<hex>`` header so the receiver can verify
authenticity.

Retry policy
------------
Up to ``CALLBACK_MAX_RETRIES`` attempts with ``CALLBACK_RETRY_DELAY_SECONDS``
between each.  Failures are logged as warnings (never raise — the job itself
already succeeded/failed independently).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------
def _sign_payload(body: bytes, secret: str) -> str:
    """Return ``sha256=<hex>`` HMAC signature for *body* using *secret*."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------
def _build_payload(job_id: str, status: str, result: dict[str, Any] | None, error: dict[str, Any] | None) -> bytes:
    data: dict[str, Any] = {
        "event": "job.completed" if status == "succeeded" else "job.failed",
        "job_id": job_id,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if result:
        data["result"] = result
    if error:
        data["error"] = error
    return json.dumps(data, default=str).encode()


# ---------------------------------------------------------------------------
# Fire-and-forget dispatcher
# ---------------------------------------------------------------------------
async def _send_once(
    url: str,
    body: bytes,
    headers: dict[str, str],
    attempt: int,
) -> bool:
    """Send a single POST. Returns True on success (2xx)."""
    s = get_settings()
    try:
        async with httpx.AsyncClient(timeout=s.CALLBACK_TIMEOUT_SECONDS) as client:
            response = await client.post(url, content=body, headers=headers)
        if response.is_success:
            logger.info(
                "callback_delivered",
                extra={"url": url, "attempt": attempt, "status": response.status_code},
            )
            return True
        logger.warning(
            "callback_non_2xx",
            extra={"url": url, "attempt": attempt, "status": response.status_code},
        )
    except Exception as exc:
        logger.warning(
            "callback_request_error",
            extra={"url": url, "attempt": attempt, "error": str(exc)},
        )
    return False


async def _dispatch(
    url: str,
    job_id: str,
    status: str,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
) -> None:
    """Internal coroutine — retries up to CALLBACK_MAX_RETRIES times."""
    s = get_settings()
    body = _build_payload(job_id, status, result, error)

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": f"{s.APP_NAME}/{s.APP_VERSION}",
        "X-Delivery-Id": str(uuid.uuid4()),
    }

    if s.CALLBACK_HMAC_SECRET:
        headers[s.CALLBACK_SIGNATURE_HEADER] = _sign_payload(body, s.CALLBACK_HMAC_SECRET)

    for attempt in range(1, s.CALLBACK_MAX_RETRIES + 1):
        success = await _send_once(url, body, headers, attempt)
        if success:
            return
        if attempt < s.CALLBACK_MAX_RETRIES:
            await asyncio.sleep(s.CALLBACK_RETRY_DELAY_SECONDS * attempt)

    logger.error(
        "callback_all_attempts_failed",
        extra={"url": url, "job_id": job_id, "attempts": s.CALLBACK_MAX_RETRIES},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fire_callback(
    url: str,
    job_id: str,
    status: str,
    result: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> None:
    """
    Schedule a fire-and-forget callback delivery.

    Safe to call from both sync and async contexts.
    Uses ``asyncio.create_task`` when called from inside a running event loop
    (e.g. from a FastAPI route or ARQ task).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            _dispatch(url, job_id, status, result, error),
            name=f"callback-{job_id}",
        )
    except RuntimeError:
        # No running loop (e.g. called from a sync context / test)
        asyncio.run(_dispatch(url, job_id, status, result, error))
