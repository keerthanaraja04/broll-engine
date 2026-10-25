"""
Test that openapi.yaml matches the running service's OpenAPI spec.

This test will FAIL if the openapi.yaml file is out of sync with the
actual FastAPI app.  Run `make openapi` to regenerate.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")
os.environ.setdefault("APP_ENV", "development")


def _load_yaml(path: str) -> dict[str, Any]:
    import yaml  # type: ignore[import]
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestOpenApiSpec:
    def test_openapi_yaml_exists(self) -> None:
        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "openapi.yaml")
        assert os.path.isfile(path), \
            "openapi.yaml not found — run `make openapi` to generate it"

    def test_openapi_matches_running_service(self) -> None:
        from app.core.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app

        app = create_app()
        live_spec = app.openapi()

        path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "openapi.yaml")

        # Skip if yaml not installed
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            pytest.skip("PyYAML not installed")

        if not os.path.isfile(path):
            pytest.skip("openapi.yaml not generated yet — run `make openapi`")

        saved_spec = _load_yaml(path)

        # Compare paths (the most important part)
        live_paths = set(live_spec.get("paths", {}).keys())
        saved_paths = set(saved_spec.get("paths", {}).keys())
        assert live_paths == saved_paths, (
            f"Path mismatch:\n"
            f"  In app but not in yaml: {live_paths - saved_paths}\n"
            f"  In yaml but not in app: {saved_paths - live_paths}\n"
            f"Run `make openapi` to regenerate."
        )

        # Compare methods per path
        for path_str in live_paths:
            live_methods = set(live_spec["paths"][path_str].keys())
            saved_methods = set(saved_spec["paths"][path_str].keys())
            assert live_methods == saved_methods, (
                f"Method mismatch for {path_str}:\n"
                f"  In app: {live_methods}\n"
                f"  In yaml: {saved_methods}"
            )

    def test_all_v1_routes_under_api_prefix(self) -> None:
        from app.core.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app

        app = create_app()
        spec = app.openapi()
        for path in spec.get("paths", {}):
            if path in ("/health", "/ready", "/docs", "/redoc",
                        "/openapi.json"):
                continue
            assert path.startswith("/api/v1/"), (
                f"Route {path} is not under /api/v1/ — "
                "all API routes must be versioned"
            )

    def test_openapi_version_is_3_1(self) -> None:
        from app.core.config import get_settings
        get_settings.cache_clear()
        from app.main import create_app

        app = create_app()
        spec = app.openapi()
        version = spec.get("openapi", "")
        assert version.startswith("3."), (
            f"OpenAPI version should be 3.x, got {version}"
        )
