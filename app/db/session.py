"""
Async SQLAlchemy engine + session factory.

Usage in FastAPI dependencies::

    async with get_db() as session:
        result = await session.execute(select(Job).where(Job.id == job_id))
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine (created once at module import time; re-created if settings change)
# ---------------------------------------------------------------------------
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        s = get_settings()
        url = s.database_url
        if "sqlite" in url:
            from sqlalchemy.pool import StaticPool
            _engine = create_async_engine(
                url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
                echo=s.POSTGRES_ECHO_SQL,
            )
        else:
            _engine = create_async_engine(
                url,
                pool_size=s.POSTGRES_POOL_SIZE,
                max_overflow=s.POSTGRES_MAX_OVERFLOW,
                pool_timeout=s.POSTGRES_POOL_TIMEOUT_SECONDS,
                pool_pre_ping=True,
                echo=s.POSTGRES_ECHO_SQL,
            )
    return _engine


def _get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=_get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# Context-manager helper (for service-layer use)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_db() -> AsyncIterator[AsyncSession]:
    """Yield a transactional AsyncSession; commits on exit, rolls back on error."""
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# FastAPI dependency (yields session, letting FastAPI manage lifecycle)
# ---------------------------------------------------------------------------
async def get_db_session() -> AsyncIterator[AsyncSession]:
    """
    Dependency injected by FastAPI.
    Yields an AsyncSession that is committed (or rolled back) automatically.
    """
    factory = _get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------
async def create_all_tables() -> None:
    """Create all tables defined in the ORM models. Idempotent (CREATE IF NOT EXISTS)."""
    from app.models.job import Base  # noqa: PLC0415 — local import avoids circular deps

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("db_tables_created_or_verified")


async def drop_all_tables() -> None:
    """Drop all ORM tables. USE ONLY IN TESTS."""
    from app.models.job import Base  # noqa: PLC0415

    engine = _get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def dispose_engine() -> None:
    """Dispose the engine connection pool (call on app shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
