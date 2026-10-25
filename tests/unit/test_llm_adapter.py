"""
Unit tests for LLMAdapter — JSON mode, schema validation, repair retry.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("LLM_ADAPTER", "mock")
os.environ.setdefault("ADAPTER_CACHE_ENABLED", "false")

KEYWORD_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["keywords"],
    "properties": {"keywords": {"type": "array",
                                  "items": {"type": "string"}}},
}
SCORE_SCHEMA: dict[str, Any] = {
    "type": "object", "required": ["score", "reason"],
    "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5},
                   "reason": {"type": "string"}},
}


class TestMockLLMAdapter:
    @pytest.mark.asyncio
    async def test_returns_dict(self) -> None:
        from app.adapters.llm_adapter import MockLLMAdapter
        r = await MockLLMAdapter().json_complete("s", "p", KEYWORD_SCHEMA)
        assert isinstance(r, dict)

    @pytest.mark.asyncio
    async def test_required_keys_present(self) -> None:
        from app.adapters.llm_adapter import MockLLMAdapter
        r = await MockLLMAdapter().json_complete("s", "p", KEYWORD_SCHEMA)
        assert "keywords" in r

    @pytest.mark.asyncio
    async def test_deterministic_same_inputs(self) -> None:
        from app.adapters.llm_adapter import MockLLMAdapter
        a = MockLLMAdapter()
        r1 = await a.json_complete("sys", "prompt", SCORE_SCHEMA)
        r2 = await a.json_complete("sys", "prompt", SCORE_SCHEMA)
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_different_prompts_differ(self) -> None:
        from app.adapters.llm_adapter import MockLLMAdapter
        a = MockLLMAdapter()
        r1 = await a.json_complete("sys", "A", SCORE_SCHEMA)
        r2 = await a.json_complete("sys", "B", SCORE_SCHEMA)
        assert r1 != r2

    @pytest.mark.asyncio
    async def test_score_is_integer(self) -> None:
        from app.adapters.llm_adapter import MockLLMAdapter
        r = await MockLLMAdapter().json_complete("s", "p", SCORE_SCHEMA)
        assert isinstance(r.get("score"), int)


class TestLLMFactory:
    def test_mock_selector(self) -> None:
        from app.adapters.llm_adapter import (
            MockLLMAdapter, get_llm_adapter, reset_llm_adapter)
        reset_llm_adapter()
        assert isinstance(get_llm_adapter(), MockLLMAdapter)
        reset_llm_adapter()

    def test_singleton(self) -> None:
        from app.adapters.llm_adapter import get_llm_adapter, reset_llm_adapter
        reset_llm_adapter()
        assert get_llm_adapter() is get_llm_adapter()
        reset_llm_adapter()


class TestSchemaValidation:
    def test_valid_data_passes(self) -> None:
        from app.adapters.llm_adapter import _validate_schema
        assert _validate_schema({"score": 3, "reason": "ok"}, SCORE_SCHEMA) == []

    def test_missing_required_fails(self) -> None:
        from app.adapters.llm_adapter import _validate_schema
        assert len(_validate_schema({"reason": "ok"}, SCORE_SCHEMA)) > 0

    def test_wrong_type_fails(self) -> None:
        from app.adapters.llm_adapter import _validate_schema
        assert len(_validate_schema(
            {"score": "high", "reason": "ok"}, SCORE_SCHEMA)) > 0


class TestGroqLLMRepairRetry:
    @pytest.mark.asyncio
    async def test_repair_retry_succeeds(self) -> None:
        from app.adapters.llm_adapter import GroqLLMAdapter
        call_count = 0

        async def _fake(self_ref: Any, msgs: list) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"wrong": "bad"}
            return {"score": 4, "reason": "fixed"}

        with patch.object(GroqLLMAdapter, "_call_groq", _fake), \
             patch.dict(os.environ,
                        {"LLM_SCHEMA_REPAIR_RETRIES": "1",
                         "LLM_MAX_RETRIES": "1"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            result = await GroqLLMAdapter().json_complete(
                "s", "p", SCORE_SCHEMA)
            assert result["score"] == 4
            assert call_count == 2
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_raises_after_all_retries(self) -> None:
        from app.adapters.llm_adapter import GroqLLMAdapter, LLMSchemaError

        async def _always_bad(self_ref: Any, msgs: list) -> dict[str, Any]:
            return {"wrong": "always"}

        with patch.object(GroqLLMAdapter, "_call_groq", _always_bad), \
             patch.dict(os.environ,
                        {"LLM_SCHEMA_REPAIR_RETRIES": "1",
                         "LLM_MAX_RETRIES": "1"}):
            from app.core.config import get_settings
            get_settings.cache_clear()
            with pytest.raises(LLMSchemaError) as exc_info:
                await GroqLLMAdapter().json_complete("s", "p", SCORE_SCHEMA)
            assert exc_info.value.code == "LLM_SCHEMA_ERROR"
        get_settings.cache_clear()
