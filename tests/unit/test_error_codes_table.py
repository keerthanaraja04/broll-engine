"""
Test that the error code table in README.md matches the actual error codes
defined in the codebase.

If this test fails, the README table is out of sync — update it.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest

os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PIXABAY_API_KEY", "test-key")


def _collect_all_error_codes() -> set[str]:
    """Collect every error code string from the codebase."""
    codes: set[str] = set()

    # From app/core/errors.py — BrollError subclasses
    from app.core.errors import BrollError
    import inspect
    for _name, obj in inspect.getmembers(
        __import__("app.core.errors", fromlist=["*"]), inspect.isclass
    ):
        if issubclass(obj, BrollError) and obj is not BrollError:
            if hasattr(obj, "code"):
                codes.add(obj.code)

    # From app/adapters/llm_adapter.py — LLMSchemaError
    from app.adapters.llm_adapter import LLMSchemaError
    codes.add(LLMSchemaError.code)

    # From app/pipeline/scene_planning/validator.py — ScenePlanValidationError
    from app.pipeline.scene_planning.validator import ScenePlanValidationError
    codes.add(ScenePlanValidationError.code)

    # From the http_exception_handler code_map
    codes.update([
        "BAD_REQUEST", "METHOD_NOT_ALLOWED", "UNPROCESSABLE_ENTITY",
        "BAD_GATEWAY", "GATEWAY_TIMEOUT", "HTTP_ERROR",
    ])

    return codes


def _parse_readme_table() -> set[str]:
    """Extract error codes from the README.md error table."""
    readme_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                "README.md")
    if not os.path.isfile(readme_path):
        pytest.skip("README.md not found")

    content = open(readme_path, encoding="utf-8").read()

    # Find the error codes table section
    table_match = re.search(
        r"## Error Codes.*?(?=##|\Z)",
        content,
        re.DOTALL,
    )
    if not table_match:
        pytest.skip("Error Codes section not found in README.md")

    table_text = table_match.group(0)

    # Extract code names from table rows: | CODE | ... |
    codes: set[str] = set()
    for m in re.finditer(r"^\|\s*([A-Z][A-Z_]+)\s*\|", table_text, re.MULTILINE):
        code = m.group(1)
        if code not in ("CODE",):
            codes.add(code)

    return codes


class TestErrorCodesTable:
    def test_all_codebase_codes_in_readme(self) -> None:
        """Every error code in the code must appear in the README table."""
        codebase = _collect_all_error_codes()
        readme = _parse_readme_table()

        missing = codebase - readme
        assert not missing, (
            f"Error codes in codebase but missing from README table: "
            f"{sorted(missing)}\n"
            f"Add them to the error codes table in README.md."
        )

    def test_no_extra_codes_in_readme(self) -> None:
        """No error codes in the README should be absent from the codebase."""
        codebase = _collect_all_error_codes()
        readme = _parse_readme_table()

        extra = readme - codebase
        # Allow the header row label
        extra.discard("CODE")
        assert not extra, (
            f"Error codes in README but not in codebase: {sorted(extra)}\n"
            f"Remove them from the error codes table in README.md."
        )

    def test_exact_match(self) -> None:
        """The two sets should be exactly equal."""
        codebase = _collect_all_error_codes()
        readme = _parse_readme_table()

        assert codebase == readme, (
            f"Mismatch:\n"
            f"  Missing from README: {sorted(codebase - readme)}\n"
            f"  Extra in README:    {sorted(readme - codebase)}"
        )
