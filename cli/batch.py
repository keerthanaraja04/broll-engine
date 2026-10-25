"""
Batch CLI — submit multiple jobs, poll for completion, download outputs.

Usage:
    python -m cli.batch --input ./inputs --api http://localhost:8000 --out ./results

Input directory structure:
    ./inputs/
        job1.json          — JSON body for POST /api/v1/broll/jobs
        job2.json
        audio1.mp3         — audio file for multipart upload
        script1.txt        — text content → script job

Output:
    ./results/
        summary.csv        — one row per input file
        <job_id>/
            video.mp4
            subtitles.srt
            timeline.json
            manifest.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="broll-batch",
    help="Batch submit jobs, poll, and download outputs",
    add_completion=False,
)
console = Console()

POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600
AUDIO_EXTS = frozenset([".mp3", ".wav", ".m4a", ".ogg", ".webm", ".flac", ".mp4"])


def _submit_json(client: httpx.Client, api: str, body: dict) -> str:
    """Submit a JSON-body job and return the job_id."""
    headers: dict[str, str] = {}
    idem_key = body.pop("_idempotency_key", None)
    if idem_key:
        headers["Idempotency-Key"] = idem_key
    r = client.post(f"{api}/api/v1/broll/jobs", json=body, headers=headers,
                    timeout=30)
    r.raise_for_status()
    return r.json()["job_id"]


def _submit_upload(client: httpx.Client, api: str, filepath: str) -> str:
    """Submit a multipart upload job and return the job_id."""
    filename = os.path.basename(filepath)
    with open(filepath, "rb") as f:
        r = client.post(
            f"{api}/api/v1/broll/jobs/upload",
            files={"audio": (filename, f, "audio/mpeg")},
            data={"aspect_ratios": "16:9"},
            timeout=120,
        )
    r.raise_for_status()
    return r.json()["job_id"]


def _submit_script(client: httpx.Client, api: str, filepath: str) -> str:
    """Submit a text file as a script job."""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()
    body = {"script": text, "aspect_ratios": ["16:9"]}
    return _submit_json(client, api, body)


def _poll_job(client: httpx.Client, api: str, job_id: str,
              timeout: int = POLL_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Poll job status until terminal or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"{api}/api/v1/jobs/{job_id}", timeout=10)
        if r.status_code == 404:
            return {"status": "not_found", "error": "Job not found"}
        r.raise_for_status()
        data = r.json()
        status_val = data.get("status", "unknown")
        if status_val in ("succeeded", "failed"):
            return data
        time.sleep(POLL_INTERVAL_SECONDS)
    return {"status": "timeout", "error": f"Timed out after {timeout}s"}


def _download_url(client: httpx.Client, url: str, dest: str) -> bool:
    """Download a URL to a local file. Returns True on success."""
    if not url:
        return False
    try:
        with client.stream("GET", url, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes(1_048_576):
                    f.write(chunk)
        return True
    except Exception:
        return False


def _process_input(client: httpx.Client, api: str, filepath: str,
                    out_dir: str) -> dict[str, Any]:
    """Process one input file: submit, poll, download."""
    name = os.path.basename(filepath)
    ext = os.path.splitext(name)[1].lower()
    result: dict[str, Any] = {"input_file": name}

    try:
        # Submit
        if ext == ".json":
            with open(filepath, encoding="utf-8") as f:
                body = json.load(f)
            job_id = _submit_json(client, api, body)
        elif ext in AUDIO_EXTS:
            job_id = _submit_upload(client, api, filepath)
        elif ext == ".txt":
            job_id = _submit_script(client, api, filepath)
        else:
            result["status"] = "skipped"
            result["error"] = f"Unsupported file type: {ext}"
            return result

        result["job_id"] = job_id
        console.print(f"  [green]Submitted:[/green] {name} → {job_id}")

        # Poll
        status = _poll_job(client, api, job_id)
        result["status"] = status.get("status", "unknown")

        if result["status"] == "succeeded":
            # Download outputs
            job_dir = os.path.join(out_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)

            video_url = (status.get("result") or {}).get("video_url")
            srt_url = (status.get("result") or {}).get("srt_url")

            if video_url and _download_url(client, video_url,
                                            os.path.join(job_dir, "video.mp4")):
                result["video_downloaded"] = True
            else:
                result["video_downloaded"] = False

            if srt_url and _download_url(client, srt_url,
                                          os.path.join(job_dir, "subtitles.srt")):
                result["srt_downloaded"] = True
            else:
                result["srt_downloaded"] = False

            result["warnings"] = status.get("warnings", [])
        elif result["status"] == "failed":
            result["error"] = status.get("error_message", "Job failed")
            result["error_code"] = status.get("error_code", "")
        else:
            result["error"] = status.get("error", "Unknown status")

    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    return result


@app.command()
def main(
    input_dir: str = typer.Option(..., "--input", "-i",
                                   help="Input directory with .json/.mp3/.txt files"),
    api: str = typer.Option("http://localhost:8000", "--api", "-a",
                             help="API base URL"),
    out: str = typer.Option("./results", "--out", "-o",
                              help="Output directory for results"),
) -> None:
    """Submit files from --input, poll, download, write summary.csv."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in input_path.iterdir()
                    if f.suffix in {".json", ".mp3", ".wav", ".m4a",
                                     ".ogg", ".webm", ".flac", ".txt", ".mp4"})

    if not files:
        console.print(f"[yellow]No input files found in {input_dir}[/yellow]")
        raise typer.Exit(1)

    console.print(f"[cyan]Found {len(files)} input file(s)[/cyan]")

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=120) as client:
        for f in files:
            console.print(f"\n[bold]Processing:[/bold] {f.name}")
            result = _process_input(client, api, str(f), out)
            results.append(result)

    # Write summary.csv
    csv_path = os.path.join(out, "summary.csv")
    fieldnames = ["input_file", "job_id", "status", "video_downloaded",
                   "srt_downloaded", "error", "error_code"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames,
                                  extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    console.print(f"\n[green]Summary written to {csv_path}[/green]")

    # Print summary table
    succeeded = sum(1 for r in results if r.get("status") == "succeeded")
    failed = sum(1 for r in results if r.get("status") == "failed")
    errors = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Succeeded: {succeeded}")
    console.print(f"  Failed:   {failed}")
    console.print(f"  Errors:   {errors}")
    console.print(f"  Skipped:  {skipped}")
    console.print(f"  Total:    {len(results)}")

    if failed or errors:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
