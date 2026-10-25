"""
API-token authentication dependency.

How it works
------------
* When ``API_AUTH_ENABLED=true``, every request to protected endpoints must
  include ``Authorization: Bearer <token>``.
* Tokens are validated against ``API_TOKENS`` (comma-separated list in config).
* When ``API_AUTH_ENABLED=false`` (default for local dev), the dependency is a
  no-op and returns ``None`` as the token identity.

Usage in routes::

    from app.core.auth import require_auth, optional_auth

    # Always requires a valid token:
    @router.post("/jobs")
    async def create_job(token: str = Depends(require_auth)):
        ...

    # Token optional — returns None if absent / auth disabled:
    @router.get("/jobs/{job_id}")
    async def get_job(token: str | None = Depends(optional_auth)):
        ...
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import get_settings
from app.core.errors import ForbiddenError, UnauthorizedError
from app.core.logging import get_logger

logger = get_logger(__name__)

# reusable_bearer does NOT auto-error so we can implement optional auth
_bearer_scheme = HTTPBearer(auto_error=False)


def _validate_token(token: str) -> str:
    """
    Validate *token* against the configured token list.

    Returns the token (acts as an identity for now; swap with a DB look-up later).
    Raises UnauthorizedError / ForbiddenError on failure.
    """
    s = get_settings()
    if not s.API_TOKENS:
        raise ForbiddenError(
            "API authentication is enabled but no tokens are configured. "
            "Set API_TOKENS in your environment."
        )
    if token not in s.API_TOKENS:
        raise UnauthorizedError("Invalid or expired API token")
    return token


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI dependency — enforces auth when ``API_AUTH_ENABLED=true``.

    Returns the validated token string.
    When auth is disabled, returns a sentinel ``"__anonymous__"``.
    """
    s = get_settings()
    if not s.API_AUTH_ENABLED:
        return "__anonymous__"

    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Missing Authorization header (Bearer token required)")

    return _validate_token(credentials.credentials)


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> str | None:
    """
    FastAPI dependency — returns token if present & valid, else None.
    Never raises — callers decide what to do with unauthenticated requests.
    """
    s = get_settings()
    if not s.API_AUTH_ENABLED:
        return None

    if credentials is None or not credentials.credentials:
        return None

    try:
        return _validate_token(credentials.credentials)
    except (UnauthorizedError, ForbiddenError):
        return None
