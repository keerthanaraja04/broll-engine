"""
ARQ WorkerSettings — tells ARQ which tasks to handle and how.
"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.worker.tasks import generate_broll


async def startup(ctx: dict) -> None:  # type: ignore[type-arg]
    s = get_settings()
    configure_logging(level=s.LOG_LEVEL, fmt=s.LOG_FORMAT)
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("worker_startup", extra={"queue": s.WORKER_QUEUE_NAME})


async def shutdown(ctx: dict) -> None:  # type: ignore[type-arg]
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("worker_shutdown")


class WorkerSettings:
    """ARQ worker configuration class (picked up by `arq app.worker.settings.WorkerSettings`)."""

    functions = [generate_broll]
    on_startup = startup
    on_shutdown = shutdown

    _s = get_settings()
    redis_settings = RedisSettings(
        host=_s.REDIS_HOST,
        port=_s.REDIS_PORT,
        database=_s.REDIS_DB,
        password=_s.REDIS_PASSWORD or None,
    )

    queue_name = get_settings().WORKER_QUEUE_NAME
    max_jobs = get_settings().WORKER_MAX_JOBS
    poll_delay = get_settings().WORKER_POLL_DELAY_SECONDS
    job_timeout = get_settings().WORKER_JOB_TIMEOUT_SECONDS
    keep_result = get_settings().WORKER_KEEP_RESULT_SECONDS
    retry_jobs = get_settings().WORKER_RETRY_JOBS
    max_tries = get_settings().WORKER_MAX_TRIES
