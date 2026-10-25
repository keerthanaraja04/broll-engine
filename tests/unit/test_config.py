"""
Unit tests for pydantic-settings configuration.
"""

from __future__ import annotations

import pytest


class TestSettings:
    def test_required_secrets_loaded(self) -> None:
        from app.core.config import get_settings
        s = get_settings()
        assert s.GROQ_API_KEY == "test-groq-key"
        assert s.PIXABAY_API_KEY == "test-pixabay-key"

    def test_database_url_property(self) -> None:
        from app.core.config import get_settings
        s = get_settings()
        url = s.database_url
        assert url.startswith("postgresql+asyncpg://")
        assert s.POSTGRES_DB in url

    def test_redis_url_property(self) -> None:
        from app.core.config import get_settings
        s = get_settings()
        url = s.redis_url
        assert url.startswith("redis://")
        assert str(s.REDIS_PORT) in url

    def test_invalid_log_level_raises(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        import os
        from app.core.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        with pytest.raises(Exception):
            get_settings()

    def test_defaults_are_sane(self) -> None:
        from app.core.config import get_settings
        s = get_settings()
        assert s.BROLL_MIN_RELEVANCE_SCORE >= 0.0
        assert s.BROLL_MIN_RELEVANCE_SCORE <= 1.0
        assert s.BROLL_MAX_CLIPS_PER_JOB > 0
        assert s.MEDIA_CRF in range(0, 52)

    def test_redis_url_with_password(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from app.core.config import get_settings
        get_settings.cache_clear()
        monkeypatch.setenv("REDIS_PASSWORD", "supersecret")
        s = get_settings()
        assert "supersecret" in s.redis_url
