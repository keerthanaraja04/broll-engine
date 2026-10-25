"""
broll-engine CLI

Entry point registered as `broll` in pyproject.toml.

Commands:
  broll worker          — start the ARQ worker
  broll job submit      — submit a broll generation job
  broll job status      — check job status from Redis
  broll health          — call the /health endpoint
  broll settings show   — print the resolved configuration (secrets redacted)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="broll",
    help="broll-engine management CLI",
    add_completion=False,
)
console = Console()
job_app = typer.Typer(help="Job management commands")
app.add_typer(job_app, name="job")


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
@app.command()
def worker(
    burst: bool = typer.Option(False, "--burst", help="Exit after processing all pending jobs"),
) -> None:
    """Start the ARQ background worker."""
    try:
        from arq import run_worker  # type: ignore[import]
        from app.worker.settings import WorkerSettings
    except ImportError as e:
        console.print(f"[red]Import error:[/red] {e}")
        raise typer.Exit(1)

    console.print("[green]Starting broll worker…[/green]")
    run_worker(WorkerSettings, burst=burst)


# ---------------------------------------------------------------------------
# Job commands
# ---------------------------------------------------------------------------
@job_app.command("submit")
def job_submit(
    audio_object: str = typer.Argument(..., help="Object key in raw storage bucket"),
    bucket: Optional[str] = typer.Option(None, "--bucket", help="Override raw storage bucket"),
) -> None:
    """Submit a broll generation job and print the job ID."""
    import uuid
    from arq import create_pool  # type: ignore[import]
    from arq.connections import RedisSettings
    from app.core.config import get_settings

    async def _submit() -> None:
        s = get_settings()
        redis_settings = RedisSettings(
            host=s.REDIS_HOST, port=s.REDIS_PORT, database=s.REDIS_DB,
            password=s.REDIS_PASSWORD or None,
        )
        pool = await create_pool(redis_settings)
        job_id = str(uuid.uuid4())
        payload = {"audio_object": audio_object, "audio_bucket": bucket or s.STORAGE_BUCKET_RAW}
        job = await pool.enqueue_job("generate_broll", job_id, payload, _job_id=job_id)
        console.print(f"[green]Job submitted:[/green] {job_id}")
        await pool.aclose()

    asyncio.run(_submit())


@job_app.command("status")
def job_status(
    job_id: str = typer.Argument(..., help="Job ID returned by `broll job submit`"),
) -> None:
    """Check the status of a job from Redis."""
    from arq import create_pool  # type: ignore[import]
    from arq.connections import RedisSettings
    from arq.jobs import Job
    from app.core.config import get_settings

    async def _status() -> None:
        s = get_settings()
        redis_settings = RedisSettings(
            host=s.REDIS_HOST, port=s.REDIS_PORT, database=s.REDIS_DB,
            password=s.REDIS_PASSWORD or None,
        )
        pool = await create_pool(redis_settings)
        job = Job(job_id, pool)
        info = await job.info()
        if info is None:
            console.print(f"[yellow]Job {job_id} not found (may have expired)[/yellow]")
        else:
            console.print_json(json.dumps({"job_id": job_id, "status": str(info.status), "result": str(info.result)}))
        await pool.aclose()

    asyncio.run(_status())


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.command()
def health(
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Call the /health endpoint and print the result."""
    import httpx

    url = f"http://{host}:{port}/health"
    try:
        r = httpx.get(url, timeout=5)
        console.print_json(r.text)
    except Exception as exc:
        console.print(f"[red]Request failed:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def ready(
    host: str = typer.Option("localhost", "--host"),
    port: int = typer.Option(8000, "--port"),
) -> None:
    """Call the /ready endpoint and print dependency status."""
    import httpx

    url = f"http://{host}:{port}/ready"
    try:
        r = httpx.get(url, timeout=10)
        console.print_json(r.text)
        if r.status_code != 200:
            raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Request failed:[/red] {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@app.command("settings")
def show_settings() -> None:
    """Print resolved settings (secrets are redacted)."""
    from app.core.config import get_settings

    s = get_settings()
    data = s.model_dump()
    # Redact secrets
    for key in ("GROQ_API_KEY", "PIXABAY_API_KEY", "POSTGRES_PASSWORD", "REDIS_PASSWORD", "STORAGE_SECRET_KEY"):
        if key in data:
            data[key] = "***REDACTED***"

    table = Table(title="broll-engine settings", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    for k, v in sorted(data.items()):
        table.add_row(k, str(v))
    console.print(table)


if __name__ == "__main__":
    app()
