#!/usr/bin/env python3
"""
Generate openapi.yaml from the FastAPI application.

Usage:
    python scripts/generate_openapi.py [--output openapi.yaml]
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Bootstrap env so Settings doesn't fail on missing keys
os.environ.setdefault("GROQ_API_KEY", "generate-placeholder")
os.environ.setdefault("PIXABAY_API_KEY", "generate-placeholder")
os.environ.setdefault("APP_ENV", "development")

import yaml  # type: ignore[import]


def main() -> None:
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.main import create_app

    app = create_app()
    spec = app.openapi()

    output_path = sys.argv[sys.argv.index("--output") + 1] \
        if "--output" in sys.argv else "openapi.yaml"

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True, width=120)

    print(f"OpenAPI spec written to {output_path}")
    print(f"  Paths: {len(spec.get('paths', {}))}")
    print(f"  Title: {spec.get('info', {}).get('title')}")
    print(f"  Version: {spec.get('info', {}).get('version')}")


if __name__ == "__main__":
    main()
