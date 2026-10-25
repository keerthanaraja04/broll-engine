"""
VLM (Vision-Language Model) adapter.

Real  : GroqVLMAdapter  — Groq vision model, 1 image/request, batched
Mock  : MockVLMAdapter  — deterministic scores based on input hash

Factory: get_vlm_adapter() reads VLM_ADAPTER from settings ("groq" | "mock").
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
from typing import Any

import httpx

from app.adapters._cache import AdapterCache, make_cache_key_bytes
from app.adapters._ratelimiter import get_bucket
from app.adapters._retry import parse_retry_after, with_retry
from app.adapters.base import BoundingBox, VLMAdapter, VLMScore
from app.core.config import get_settings
from app.core.errors import GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)

_MAX_B64_BYTES = 4_000_000

_SCORE_PROMPT = (
    "You are a video producer scoring stock footage relevance.\n"
    "Given the image and the text below, respond with ONLY valid JSON:\n"
    '{{"score": <1-5>, "reason": "<one sentence>", '
    '"subject_bbox": {{"x1":0.0,"y1":0.0,"x2":1.0,"y2":1.0}}, '
    '"flags": []}}\n\nText: {text}'
)


def _resize_if_needed(image_bytes: bytes, max_px: int) -> bytes:
    """Resize image so its longest side <= max_px. Returns original if small."""
    try:
        from PIL import Image  # type: ignore[import]
        import io
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        if max(w, h) <= max_px:
            return image_bytes
        ratio = max_px / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def _cache_key_for_image(image_bytes: bytes, text: str, idx: int) -> bytes:
    return make_cache_key_bytes(image_bytes, text, str(idx))


def _to_cache_score(score: VLMScore) -> dict[str, Any]:
    return {
        "score": score.score, "reason": score.reason,
        "bbox": {"x1": score.subject_bbox.x1, "y1": score.subject_bbox.y1,
                 "x2": score.subject_bbox.x2, "y2": score.subject_bbox.y2},
        "flags": score.flags, "image_index": score.image_index,
    }


def _from_cache_score(d: dict[str, Any]) -> VLMScore:
    bbox = d.get("bbox", {})
    return VLMScore(
        score=d["score"], reason=d["reason"],
        subject_bbox=BoundingBox(
            x1=bbox.get("x1", 0), y1=bbox.get("y1", 0),
            x2=bbox.get("x2", 1), y2=bbox.get("y2", 1)),
        flags=d.get("flags", []),
        image_index=d.get("image_index", 0),
    )


def _parse_score_response(body: dict[str, Any], index: int) -> VLMScore:
    content = body["choices"][0]["message"]["content"].strip()
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group()) if m else {}

    bbox_raw = data.get("subject_bbox", {})
    bbox = BoundingBox(
        x1=float(bbox_raw.get("x1", 0.0)), y1=float(bbox_raw.get("y1", 0.0)),
        x2=float(bbox_raw.get("x2", 1.0)), y2=float(bbox_raw.get("y2", 1.0)),
    )
    return VLMScore(
        score=max(1, min(5, int(data.get("score", 3)))),
        reason=str(data.get("reason", ""))[:200],
        subject_bbox=bbox,
        flags=list(data.get("flags", [])),
        image_index=index,
    )


class GroqVLMAdapter(VLMAdapter):
    """Scores images via Groq vision model, one image per request."""

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._base_url = str(s.GROQ_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(s.VLM_TIMEOUT_SECONDS)
        self._cache = AdapterCache("vlm", ttl=s.VLM_CACHE_TTL_SECONDS,
                                   enabled=s.VLM_CACHE_ENABLED)
        self._bucket = get_bucket("vlm", capacity=s.VLM_RATE_LIMIT_TOKENS,
                                  refill_rate=s.VLM_RATE_LIMIT_REFILL_RATE)

    async def score(self, images: list[bytes], text: str) -> list[VLMScore]:
        tasks = [self._score_one(img, text, i)
                 for i, img in enumerate(images)]
        return list(await asyncio.gather(*tasks))

    async def _score_one(self, image_bytes: bytes, text: str,
                         index: int) -> VLMScore:
        s = self._s
        cache_key = _cache_key_for_image(image_bytes, text, index)
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return _from_cache_score(cached)

        processed = _resize_if_needed(image_bytes, s.VLM_IMAGE_RESIZE_MAX_PX)
        if len(base64.b64encode(processed)) > _MAX_B64_BYTES:
            processed = _resize_if_needed(
                image_bytes, s.VLM_IMAGE_RESIZE_MAX_PX // 2)

        await self._bucket.acquire()

        async def _call() -> VLMScore:
            return await self._call_groq(processed, text, index)

        result = await with_retry(
            fn=_call, max_retries=s.VLM_MAX_RETRIES,
            adapter_name="vlm",
            retryable_exceptions=(httpx.TimeoutException, httpx.ConnectError,
                                  GroqServiceError),
        )
        await self._cache.set(cache_key, _to_cache_score(result))
        return result

    async def _call_groq(self, image_bytes: bytes, text: str,
                         index: int) -> VLMScore:
        s = self._s
        url = f"{self._base_url}/chat/completions"
        b64 = _to_b64(image_bytes)
        payload = {
            "model": s.VLM_MODEL,
            "max_tokens": s.VLM_MAX_TOKENS,
            "temperature": s.VLM_TEMPERATURE,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text",
                     "text": _SCORE_PROMPT.format(text=text)},
                ],
            }],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {s.GROQ_API_KEY}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"VLM timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"VLM connect error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Groq VLM rate limit",
                                 details={"retry_after_seconds":
                                          parse_retry_after(response)})
        if not response.is_success:
            raise GroqServiceError(
                f"Groq VLM HTTP {response.status_code}",
                details={"body": response.text[:300]})
        try:
            return _parse_score_response(response.json(), index)
        except Exception as exc:
            raise GroqServiceError("VLM parse error") from exc


class MockVLMAdapter(VLMAdapter):
    """Deterministic mock: score derived from SHA-256 of image bytes."""

    async def score(self, images: list[bytes], text: str) -> list[VLMScore]:
        results = []
        for i, img in enumerate(images):
            h = int(hashlib.sha256(img[:512]).hexdigest(), 16)
            sc = (h % 5) + 1
            results.append(VLMScore(
                score=sc,
                reason=f"Mock score {sc} for image {i}",
                subject_bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.9, y2=0.9),
                flags=[],
                image_index=i,
            ))
        return results


_adapter: VLMAdapter | None = None


def get_vlm_adapter() -> VLMAdapter:
    global _adapter
    if _adapter is None:
        impl = get_settings().VLM_ADAPTER.lower()
        _adapter = MockVLMAdapter() if impl == "mock" else GroqVLMAdapter()
    return _adapter


def reset_vlm_adapter() -> None:
    global _adapter
    _adapter = None
