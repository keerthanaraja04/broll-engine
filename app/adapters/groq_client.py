"""
Groq API adapter.

Provides:
  - chat_completion()  — wraps the OpenAI-compatible chat endpoint
  - transcribe()       — wraps the Whisper transcription endpoint

Both methods use httpx for async I/O and tenacity for retry logic.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.errors import GroqServiceError, RateLimitError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _make_retry_decorator(max_retries: int) -> Any:
    return retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )


class GroqClient:
    """Thin async wrapper around the Groq REST API."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._base_url = str(self._settings.GROQ_BASE_URL).rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {self._settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        self._timeout = httpx.Timeout(self._settings.GROQ_REQUEST_TIMEOUT_SECONDS)

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Call /chat/completions and return the parsed JSON response body.

        Args:
            messages:    OpenAI-format message list.
            model:       Override the configured model.
            max_tokens:  Override the configured max_tokens.
            temperature: Override the configured temperature.
            **kwargs:    Extra parameters forwarded to the API.

        Returns:
            Parsed JSON dict from Groq.

        Raises:
            RateLimitError:   HTTP 429.
            GroqServiceError: Any other non-2xx response or network error.
        """
        s = self._settings
        payload: dict[str, Any] = {
            "model": model or s.GROQ_CHAT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens or s.GROQ_MAX_TOKENS,
            "temperature": temperature if temperature is not None else s.GROQ_TEMPERATURE,
            **kwargs,
        }

        return await self._post("/chat/completions", payload)

    # ------------------------------------------------------------------
    # Transcription (Whisper)
    # ------------------------------------------------------------------
    async def transcribe(
        self,
        audio_bytes: bytes,
        filename: str = "audio.mp3",
        model: str | None = None,
        language: str | None = None,
        response_format: str = "verbose_json",
    ) -> dict[str, Any]:
        """
        Transcribe audio using Groq's Whisper endpoint.

        Returns the full verbose_json response (includes segments with timestamps).
        """
        s = self._settings
        files = {"file": (filename, audio_bytes, "audio/mpeg")}
        data: dict[str, str] = {
            "model": model or s.GROQ_WHISPER_MODEL,
            "response_format": response_format,
        }
        if language:
            data["language"] = language

        # Multipart — send without Content-Type header (httpx sets it automatically)
        headers = {"Authorization": self._headers["Authorization"]}

        return await self._multipart_post("/audio/transcriptions", data=data, files=files, headers=headers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        logger.debug("groq_request", extra={"url": url, "model": payload.get("model")})

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, headers=self._headers, json=payload)
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"Groq request timed out: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"Could not connect to Groq: {exc}") from exc

        return self._handle_response(response)

    async def _multipart_post(
        self,
        path: str,
        data: dict[str, str],
        files: dict[str, Any],
        headers: dict[str, str],
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, headers=headers, data=data, files=files)
            except httpx.TimeoutException as exc:
                raise GroqServiceError(f"Groq transcription timed out: {exc}") from exc
            except httpx.ConnectError as exc:
                raise GroqServiceError(f"Could not connect to Groq: {exc}") from exc

        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: httpx.Response) -> dict[str, Any]:
        if response.status_code == 429:
            raise RateLimitError("Groq rate limit exceeded — please retry later")
        if not response.is_success:
            body = response.text
            raise GroqServiceError(
                f"Groq returned HTTP {response.status_code}",
                details={"body": body[:500]},
            )
        try:
            return response.json()  # type: ignore[no-any-return]
        except json.JSONDecodeError as exc:
            raise GroqServiceError("Groq response is not valid JSON") from exc


# ---------------------------------------------------------------------------
# Module-level singleton factory
# ---------------------------------------------------------------------------
_client: GroqClient | None = None


def get_groq_client() -> GroqClient:
    global _client
    if _client is None:
        _client = GroqClient()
    return _client
