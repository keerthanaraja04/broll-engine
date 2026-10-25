"""
Reusable FastAPI dependencies.

Imported by route modules with::

    from app.api.deps import DbSession, CurrentToken, get_job_or_404
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import optional_auth, require_auth
from app.core.errors import JobNotFoundError
from app.db.session import get_db_session
from app.models.job import Job

# ---------------------------------------------------------------------------
# Type aliases for cleaner route signatures
# ---------------------------------------------------------------------------

# Injects an AsyncSession (committed/rolled-back by FastAPI lifecycle)
DbSession = Annotated[AsyncSession, Depends(get_db_session)]

# Token that is required when API_AUTH_ENABLED=true
RequiredToken = Annotated[str, Depends(require_auth)]

# Token that is optional (returns None when absent or auth disabled)
OptionalToken = Annotated[str | None, Depends(optional_auth)]


# ---------------------------------------------------------------------------
# Job lookup helper
# ---------------------------------------------------------------------------

async def get_job_or_404(job_id: UUID, db: DbSession) -> Job:
    """
    Fetch a Job by primary key.
    Raises JobNotFoundError (404) if not found.
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise JobNotFoundError(f"Job '{job_id}' not found")
    return job
