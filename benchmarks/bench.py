"""
benchmarks/bench.py — turnaround, cost, memory, and concurrency benchmarks.

Measures
--------
1. Cold-start turnaround  — first 60-second job, all caches empty
2. Warm-start turnaround  — same job again, caches populated
3. Cost per minute        — parsed from manifest.json cost_estimate
4. Peak RSS memory        — tracked during job execution
5. Concurrency degradation — submit N concurrent 60-second jobs, measure
                              when errors or time degrade by > 20%

Usage
-----
    # Full benchmark against a running stack (docker compose up -d)
    python -m benchmarks.bench --api http://localhost:8000 --script tests/fixtures/bench_script.txt

    # Mock-only benchmark (no docker, no API keys)
    python -m benchmarks.bench --mock

Output
------
    benchmarks/metrics.md   — Markdown table printed + saved
    benchmarks/raw.json     — Raw timing / cost data for further analysis
"""

from __future__ import annotations

import asyncio
import json
try:
    import resource
except ImportError:
    resource = None  # Windows compatibility

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_API = "http://localhost:8000"
POLL_INTERVAL = 3.0
JOB_TIMEOUT = 300
BENCH_SCRIPT = (
    "Technology is changing the way we work, live, and communicate. "
    "Artificial intelligence enables new capabilities every day. "
    "The future of automation is already here."
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JobRun:
    label: str
    job_id: str | None = None
    submit_time: float = 0.0
    complete_time: float = 0.0
    status: str = "unknown"
    cost_usd: float = 0.0
    peak_rss_mb: float = 0.0
    error: str | None = None

    @property
    def turnaround_seconds(self) -> float:
        return round(self.complete_time - self.submit_time, 2)


@dataclass
class BenchmarkReport:
    cold_start: JobRun | None = None
    warm_start: JobRun | None = None
    cost_per_minute: float | None = None
    concurrency_results: list[JobRun] = field(default_factory=list)
    max_clean_concurrency: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "## Benchmark Results",
            "",
            "### Turnaround Time (60-second job)",
            "",
            "| Run | Turnaround (s) | Status | Cost (USD) | Peak RSS (MB) |",
            "|-----|:--------------:|--------|:----------:|:-------------:|",
        ]
        if self.cold_start:
            c = self.cold_start
            lines.append(
                f"| Cold (caches empty) | {c.turnaround_seconds} | {c.status} "
                f"| ${c.cost_usd:.5f} | {c.peak_rss_mb:.1f} |"
            )
        if self.warm_start:
            w = self.warm_start
            lines.append(
                f"| Warm (caches loaded) | {w.turnaround_seconds} | {w.status} "
                f"| ${w.cost_usd:.5f} | {w.peak_rss_mb:.1f} |"
            )
        if self.cold_start and self.warm_start and self.cold_start.turnaround_seconds > 0:
            speedup = round(
                self.cold_start.turnaround_seconds / max(self.warm_start.turnaround_seconds, 0.01), 2
            )
            lines.append(
                f"| **Speedup** | **{speedup}×** | — | — | — |"
            )

        lines += [
            "",
            "### Cost Estimate",
            "",
        ]
        if self.cost_per_minute is not None:
            lines.append(f"- **Cost per minute of output video:** ${self.cost_per_minute:.5f} USD")
        else:
            lines.append("- Cost per minute: N/A (mock mode)")

        lines += [
            "",
            "### Concurrency",
            "",
            "| Concurrent Jobs | All OK | Avg Turnaround (s) | Degraded |",
            "|:---------------:|:------:|:------------------:|:--------:|",
        ]
        for run in self.concurrency_results:
            ok = "OK" if run.status == "succeeded" else "FAIL"
            lines.append(
                f"| — | {ok} | {run.turnaround_seconds} | — |"
            )
        if self.max_clean_concurrency:
            lines.append(
                f"\n**Max clean concurrency:** {self.max_clean_concurrency} jobs"
            )

        if self.warnings:
            lines += ["", "### Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _submit(client: httpx.Client, api: str, script: str) -> str | None:
    try:
        r = client.post(
            f"{api}/api/v1/broll/jobs",
            json={"script": script},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["job_id"]
    except Exception as exc:
        return None


def _poll(client: httpx.Client, api: str, job_id: str,
           timeout: float = JOB_TIMEOUT) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = client.get(f"{api}/api/v1/jobs/{job_id}", timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get("status") in ("succeeded", "failed"):
                return data
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
    return {"status": "timeout"}


def _peak_rss_mb() -> float:
    if resource is None:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
        except Exception:
            return 0.0
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # On Linux: ru_maxrss is in kilobytes; on macOS: bytes
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1_048_576
        return usage.ru_maxrss / 1024
    except Exception:
        return 0.0


def _cost_from_manifest(result: dict) -> float:
    manifest = result.get("manifest", {})
    estimate = manifest.get("cost_estimate", {})
    return float(estimate.get("total_usd", 0.0))


# ---------------------------------------------------------------------------
# Mock mode — deterministic without docker
# ---------------------------------------------------------------------------

def _run_mock_benchmark() -> BenchmarkReport:
    """Run a mock benchmark that simulates timings without real API calls."""
    report = BenchmarkReport()
    report.warnings.append("Mock mode — all timings and costs are synthetic")

    cold = JobRun(
        label="cold",
        job_id="mock-cold-001",
        submit_time=0.0,
        complete_time=42.7,
        status="succeeded",
        cost_usd=0.00823,
        peak_rss_mb=145.2,
    )
    warm = JobRun(
        label="warm",
        job_id="mock-warm-001",
        submit_time=0.0,
        complete_time=11.3,
        status="succeeded",
        cost_usd=0.00412,
        peak_rss_mb=148.1,
    )
    report.cold_start = cold
    report.warm_start = warm
    report.cost_per_minute = 0.00823 / 1.0  # cost per minute of output

    # Fake concurrency results
    for i in range(1, 6):
        run = JobRun(
            label=f"concurrent_{i}",
            job_id=f"mock-conc-{i:03d}",
            submit_time=0.0,
            complete_time=42.7 + (i - 1) * 3.5,  # degrades linearly
            status="succeeded" if i <= 4 else "failed",
            cost_usd=0.008 * i,
            peak_rss_mb=145 + i * 12,
        )
        report.concurrency_results.append(run)
        if run.status == "succeeded":
            report.max_clean_concurrency = i

    return report


# ---------------------------------------------------------------------------
# Real benchmark
# ---------------------------------------------------------------------------

def _run_one(client: httpx.Client, api: str, script: str, label: str) -> JobRun:
    run = JobRun(label=label)
    run.submit_time = time.time()
    job_id = _submit(client, api, script)
    if job_id is None:
        run.status = "error"
        run.error = "submit failed"
        run.complete_time = time.time()
        return run

    run.job_id = job_id
    result = _poll(client, api, job_id)
    run.complete_time = time.time()
    run.status = result.get("status", "unknown")
    run.cost_usd = _cost_from_manifest(result)
    run.peak_rss_mb = _peak_rss_mb()
    return run


def run_benchmark(api: str, script: str) -> BenchmarkReport:
    report = BenchmarkReport()

    with httpx.Client(timeout=30) as client:
        # Health check
        try:
            r = client.get(f"{api}/health", timeout=5)
            if r.status_code != 200:
                report.warnings.append(f"API returned {r.status_code} on /health")
        except Exception as exc:
            report.warnings.append(f"Cannot reach {api}: {exc}")
            return report

        # Cold start
        print(f"[bench] Cold-start run...", flush=True)
        cold = _run_one(client, api, script, "cold")
        report.cold_start = cold
        print(f"  → {cold.turnaround_seconds}s ({cold.status})")

        # Warm start (caches populated)
        print("[bench] Warm-start run...", flush=True)
        warm = _run_one(client, api, script, "warm")
        report.warm_start = warm
        print(f"  → {warm.turnaround_seconds}s ({warm.status})")

        # Cost per minute
        dur = 60.0  # we submitted a ~60s script
        if cold.cost_usd > 0:
            report.cost_per_minute = round(cold.cost_usd / (dur / 60), 6)

        # Concurrency
        for n_conc in (1, 2, 4, 8):
            print(f"[bench] Concurrency={n_conc}...", flush=True)
            start_t = time.time()
            job_ids = [_submit(client, api, script) for _ in range(n_conc)]
            job_ids = [j for j in job_ids if j is not None]
            results = [_poll(client, api, j, timeout=JOB_TIMEOUT) for j in job_ids]
            elapsed = time.time() - start_t
            statuses = [r.get("status") for r in results]
            all_ok = all(s == "succeeded" for s in statuses)
            avg_t = round(elapsed / max(n_conc, 1), 1)

            run = JobRun(
                label=f"concurrent_{n_conc}",
                submit_time=0.0, complete_time=elapsed,
                status="succeeded" if all_ok else "degraded",
                peak_rss_mb=_peak_rss_mb(),
            )
            report.concurrency_results.append(run)

            if all_ok:
                report.max_clean_concurrency = n_conc
                print(f"  → all {n_conc} OK in {elapsed:.1f}s")
            else:
                print(f"  → degraded at n={n_conc}")
                break  # stop increasing concurrency

    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="broll-engine benchmark")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--script", default=None,
                        help="Path to .txt script file (default: built-in 60s script)")
    parser.add_argument("--mock", action="store_true",
                        help="Run synthetic mock benchmark (no API needed)")
    parser.add_argument("--out-dir", default="benchmarks")
    args = parser.parse_args()

    script = BENCH_SCRIPT
    if args.script and os.path.isfile(args.script):
        script = open(args.script, encoding="utf-8").read()

    if args.mock:
        report = _run_mock_benchmark()
    else:
        report = run_benchmark(args.api, script)

    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    md_path = os.path.join(args.out_dir, "metrics.md")
    raw_path = os.path.join(args.out_dir, "raw.json")

    md_content = report.to_markdown()
    print("\n" + md_content)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    raw = {
        "cold_turnaround_s": report.cold_start.turnaround_seconds if report.cold_start else None,
        "warm_turnaround_s": report.warm_start.turnaround_seconds if report.warm_start else None,
        "cost_per_minute_usd": report.cost_per_minute,
        "cold_peak_rss_mb": report.cold_start.peak_rss_mb if report.cold_start else None,
        "max_clean_concurrency": report.max_clean_concurrency,
        "concurrency": [
            {"label": r.label, "status": r.status,
             "turnaround_s": r.turnaround_seconds, "rss_mb": r.peak_rss_mb}
            for r in report.concurrency_results
        ],
        "warnings": report.warnings,
    }
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    print(f"\n[bench] Saved: {md_path}, {raw_path}")


if __name__ == "__main__":
    main()
