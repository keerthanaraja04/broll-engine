"""
LLM (Large Language Model) adapter.

Real  : GroqLLMAdapter  — Groq chat in JSON mode + schema validation + repair retry
Mock  : MockLLMAdapter  — deterministic, no API key needed

Factory: get_llm_adapter() reads LLM_ADAPTER from settings ("groq" | "mock").
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx

from app.adapters._cache import AdapterCache, make_cache_key_bytes
from app.adapters._ratelimiter import get_bucket
from app.adapters._retry import parse_retry_after, with_retry
from app.adapters.base import LLMAdapter
from app.core.config import get_settings
from app.core.errors import BrollError, GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMSchemaError(BrollError):
    """LLM response did not conform to the requested JSON schema."""
    code = "LLM_SCHEMA_ERROR"
    http_status = 422
    retryable = True


def _validate_schema(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate *data* against *schema*. Returns list of error messages."""
    try:
        import jsonschema  # type: ignore[import]
        validator = jsonschema.Draft7Validator(schema)
        errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
        return [e.message for e in errors]
    except ImportError:
        if not isinstance(data, dict):
            return ["Response is not a JSON object"]
        return []


def _to_cache(result: dict[str, Any]) -> dict[str, Any]:
    return {"result": result}


def _from_cache(d: dict[str, Any]) -> dict[str, Any]:
    return d["result"]  # type: ignore[return-value]


class GroqLLMAdapter(LLMAdapter):
    """JSON-mode chat completion via Groq with schema validation + repair."""

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self._base_url = str(s.GROQ_BASE_URL).rstrip("/")
        self._timeout = httpx.Timeout(s.LLM_TIMEOUT_SECONDS)
        self._cache = AdapterCache("llm", ttl=s.LLM_CACHE_TTL_SECONDS,
                                   enabled=s.LLM_CACHE_ENABLED)
        self._bucket = get_bucket("llm", capacity=s.LLM_RATE_LIMIT_TOKENS,
                                  refill_rate=s.LLM_RATE_LIMIT_REFILL_RATE)

    async def json_complete(
        self, system: str, prompt: str, schema: dict[str, Any],
    ) -> dict[str, Any]:
        s = self._s
        cache_key = make_cache_key_bytes(system, prompt,
                                          json.dumps(schema, sort_keys=True))
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug("llm_cache_hit")
            return _from_cache(cached)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        validation_errors: list[str] = []

        for repair_attempt in range(s.LLM_SCHEMA_REPAIR_RETRIES + 1):
            if repair_attempt > 0 and validation_errors:
                feedback = (
                    "Your previous response did not match the JSON schema.\n"
                    "Errors:\n" +
                    "\n".join(f"- {e}" for e in validation_errors) +
                    "\n\nOutput ONLY valid JSON matching:\n"
                    + json.dumps(schema, indent=2)
                )
                messages.append({"role": "user", "content": feedback})
                await self._bucket.acquire()

            raw = await with_retry(
                fn=lambda msgs=messages: self._call_groq(msgs),
                max_retries=s.LLM_MAX_RETRIES,
                adapter_name="llm",
                retryable_exceptions=(httpx.TimeoutException,
                                      httpx.ConnectError, GroqServiceError),
            )
            validation_errors = _validate_schema(raw, schema)
            if not validation_errors:
                await self._cache.set(cache_key, _to_cache(raw))
                return raw
            logger.warning("llm_schema_failed",
                            extra={"repair": repair_attempt,
                                   "errors": validation_errors[:3]})

        raise LLMSchemaError(
            f"LLM response did not match schema after "
            f"{s.LLM_SCHEMA_REPAIR_RETRIES + 1} attempt(s)",
            details={"errors": validation_errors,
                     "schema_keys": list(schema.get("properties", {}).keys())},
        )

    async def _call_groq(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        s = self._s
        url = f"{self._base_url}/chat/completions"
        payload = {
            "model": s.LLM_JSON_MODEL,
            "messages": messages,
            "max_tokens": s.LLM_MAX_TOKENS,
            "temperature": s.LLM_TEMPERATURE,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {s.GROQ_API_KEY}",
                    "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"LLM timeout: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"LLM connect error: {exc}") from exc

        if response.status_code == 429:
            raise RateLimitError("Groq LLM rate limit",
                                 details={"retry_after_seconds":
                                          parse_retry_after(response)})
        if not response.is_success:
            raise GroqServiceError(
                f"Groq LLM HTTP {response.status_code}",
                details={"body": response.text[:400]})

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise GroqServiceError(
                "Groq LLM returned non-JSON in JSON mode",
                details={"content": str(body)[:400]}) from exc


class MockLLMAdapter(LLMAdapter):
    """
    Deterministic mock LLM.

    Returns a minimal dict satisfying the schema's required fields
    with stable placeholder values derived from hash(system + prompt).
    """

    async def json_complete(
        self, system: str, prompt: str, schema: dict[str, Any],
    ) -> dict[str, Any]:
        h = int(hashlib.sha256((system + prompt).encode()).hexdigest(), 16)
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        result: dict[str, Any] = {}

        for key in required or list(properties.keys())[:5]:
            prop = properties.get(key, {})
            ptype = prop.get("type", "string")
            if ptype == "string":
                result[key] = f"mock_{key}_{h % 1000}"
            elif ptype in ("number", "integer"):
                result[key] = (h % 5) + 1
            elif ptype == "boolean":
                result[key] = bool(h % 2)
            elif ptype == "array":
                result[key] = [f"item_{h % 10}", f"item_{(h + 1) % 10}"]
            elif ptype == "object":
                result[key] = {}
            else:
                result[key] = f"mock_{key}"

        return result


_adapter: LLMAdapter | None = None


def get_llm_adapter() -> LLMAdapter:
    global _adapter
    if _adapter is None:
        impl = get_settings().LLM_ADAPTER.lower()
        _adapter = MockLLMAdapter() if impl == "mock" else GroqLLMAdapter()
    return _adapter


def reset_llm_adapter() -> None:
    global _adapter
    _adapter = None
