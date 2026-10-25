"""
eval/vlm_eval.py — independent VLM evaluation of chosen clips.

Scores each clip in a timeline.json by sampling 3 frames and scoring each
frame against the scene narration using a DIFFERENT Groq vision model than
the one used during selection (VLM_EVAL_MODEL, default: llava-v1.5-13b).

Also reads an optional human_ratings.csv and computes:
  - Spearman rank correlation
  - Mean Absolute Error (MAE)
  - % predictions within ±1 of human rating

Outputs a Markdown table suitable for inclusion in REPORT.md.

Usage
-----
    python -m eval.vlm_eval \\
        --timeline results/timeline.json \\
        --human    eval/human_ratings.csv \\
        --out      eval/eval_results.json

Environment
-----------
    GROQ_API_KEY     — required for real scoring
    VLM_EVAL_MODEL   — overrides config default
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str | None,
    video_url: str | None,
    n_frames: int = 3,
    tmp_dir: str | None = None,
) -> list[bytes]:
    """
    Extract *n_frames* evenly-spaced JPEG frames from a video.

    If *video_path* is not a local file, tries to download from *video_url*.
    Returns list of JPEG bytes (may be shorter than n_frames on error).
    """
    tmp_dir = tmp_dir or tempfile.gettempdir()
    local: str | None = None

    if video_path and os.path.isfile(video_path):
        local = video_path
    elif video_url:
        try:
            r = httpx.get(video_url, timeout=30, follow_redirects=True)
            r.raise_for_status()
            ext = ".mp4"
            fd, local = tempfile.mkstemp(suffix=ext, dir=tmp_dir)
            with os.fdopen(fd, "wb") as f:
                f.write(r.content)
        except Exception as exc:
            print(f"[warn] Could not download {video_url}: {exc}", file=sys.stderr)
            return []

    if local is None:
        return []

    # Get duration
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", local,
    ]
    try:
        result = subprocess.run(probe_cmd, capture_output=True, timeout=15)
        probe = json.loads(result.stdout)
        duration = float(probe.get("format", {}).get("duration", 5.0))
    except Exception:
        duration = 5.0

    frames: list[bytes] = []
    for i in range(n_frames):
        t = duration * (i + 1) / (n_frames + 1)
        frame_path = os.path.join(tmp_dir, f"frame_{i}_{os.getpid()}.jpg")
        cmd = [
            "ffmpeg", "-y", "-ss", str(t),
            "-i", local, "-vframes", "1",
            "-q:v", "5", frame_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=15, check=True)
            if os.path.isfile(frame_path):
                with open(frame_path, "rb") as f:
                    frames.append(f.read())
                os.unlink(frame_path)
        except Exception as exc:
            print(f"[warn] Frame extraction failed at t={t:.1f}s: {exc}", file=sys.stderr)

    return frames


# ---------------------------------------------------------------------------
# Groq VLM scoring
# ---------------------------------------------------------------------------

async def score_frame_with_groq(
    frame_bytes: bytes,
    narration: str,
    model: str,
    api_key: str,
    base_url: str = "https://api.groq.com/openai/v1",
) -> tuple[int, str]:
    """
    Score a single frame against narration text.

    Returns (score_1_to_5, reason).
    """
    b64 = base64.b64encode(frame_bytes).decode()
    prompt = (
        f"Rate how well this video frame illustrates the following narration "
        f"on a scale of 1 (completely irrelevant) to 5 (perfect match).\n\n"
        f"Narration: {narration}\n\n"
        f"Respond ONLY with JSON: {{\"score\": <1-5>, \"reason\": \"<one sentence>\"}}"
    )
    payload = {
        "model": model,
        "max_tokens": 150,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json=payload,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

    import re
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group()) if m else {}
    score = max(1, min(5, int(data.get("score", 3))))
    reason = str(data.get("reason", ""))[:200]
    return score, reason


# ---------------------------------------------------------------------------
# Evaluation dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SceneEvalResult:
    scene_index: int
    narration: str
    clip_id: str | None
    eval_score: float         # mean across frames
    eval_scores_per_frame: list[int]
    eval_reasons: list[str]
    human_score: float | None = None
    selection_score: float | None = None   # from timeline.json


@dataclass
class EvalReport:
    scene_results: list[SceneEvalResult]
    spearman_rho: float | None = None
    mae: float | None = None
    pct_within_1: float | None = None
    n_human_rated: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_markdown_table(self) -> str:
        lines: list[str] = [
            "## VLM Evaluation Results",
            "",
            "| Scene | Clip | Eval Score | Human | Diff | Selection Score |",
            "|------:|------|:----------:|:-----:|:----:|:---------------:|",
        ]
        for sr in self.scene_results:
            delta = ""
            if sr.human_score is not None:
                d = round(sr.eval_score - sr.human_score, 2)
                delta = f"{d:+.2f}"
            lines.append(
                f"| {sr.scene_index} "
                f"| {(sr.clip_id or 'n/a')[:20]} "
                f"| {sr.eval_score:.2f} "
                f"| {sr.human_score or '-'} "
                f"| {delta} "
                f"| {sr.selection_score or '-'} |"
            )

        lines += ["", "### Aggregate Metrics", ""]
        if self.n_human_rated > 0:
            lines += [
                f"- **Scenes rated by humans:** {self.n_human_rated}",
                f"- **Spearman rho (eval vs human):** "
                f"{self.spearman_rho:.3f}" if self.spearman_rho is not None else "- **Spearman rho:** N/A",
                f"- **MAE:** "
                f"{self.mae:.3f}" if self.mae is not None else "- **MAE:** N/A",
                f"- **% within +/-1:** "
                f"{self.pct_within_1:.1f}%" if self.pct_within_1 is not None else "- **% within +/-1:** N/A",
            ]
        else:
            lines.append(
                "_No human ratings provided. Supply `--human eval/human_ratings.csv` "
                "to compute Spearman/MAE/±1 metrics._"
            )

        if self.warnings:
            lines += ["", "### Warnings", ""]
            for w in self.warnings:
                lines.append(f"- {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Human ratings loader
# ---------------------------------------------------------------------------

def load_human_ratings(csv_path: str) -> dict[int, float]:
    """
    Load human ratings from a CSV with columns: scene_index, human_score.

    Returns {scene_index: human_score}.
    """
    ratings: dict[int, float] = {}
    if not os.path.isfile(csv_path):
        return ratings
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["scene_index"])
                score = float(row["human_score"])
                ratings[idx] = max(1.0, min(5.0, score))
            except (KeyError, ValueError):
                continue
    return ratings


# ---------------------------------------------------------------------------
# Spearman / MAE helpers
# ---------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Return the rank of each value (1-based, avg for ties)."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[order[j + 1]] == values[order[j]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def compute_spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    assert len(x) == len(y) and len(x) >= 2
    rx, ry = _rank(x), _rank(y)
    n = len(x)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den = (
        sum((rx[i] - mean_rx) ** 2 for i in range(n)) *
        sum((ry[i] - mean_ry) ** 2 for i in range(n))
    ) ** 0.5
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

async def evaluate_timeline(
    timeline: dict[str, Any],
    model: str,
    api_key: str,
    human_ratings: dict[int, float] | None = None,
    n_frames: int = 3,
    tmp_dir: str | None = None,
    use_mock: bool = False,
) -> EvalReport:
    """
    Score every scene in *timeline* and return an EvalReport.

    Args:
        timeline:       Parsed timeline.json dict.
        model:          Groq vision model name (should differ from selection model).
        api_key:        Groq API key.
        human_ratings:  Optional {scene_index: score} from human_ratings.csv.
        n_frames:       Frames to sample per clip.
        tmp_dir:        Temp directory for frame extraction.
        use_mock:       Skip real API calls; return deterministic mock scores.
    """
    scenes = timeline.get("scenes", [])
    human_ratings = human_ratings or {}
    results: list[SceneEvalResult] = []
    warnings: list[str] = []
    tmp = tmp_dir or tempfile.gettempdir()

    for scene in scenes:
        idx = scene.get("index", 0)
        narration = scene.get("text", "")
        clip = scene.get("clip") or {}
        clip_id = clip.get("id")
        clip_url = clip.get("url", "")
        selection_score = scene.get("relevance_score")

        if use_mock:
            import hashlib
            h = int(hashlib.sha256(f"{idx}{narration}".encode()).hexdigest(), 16)
            scores = [(h % 5) + 1 for _ in range(n_frames)]
            reasons = [f"mock score for scene {idx}"] * n_frames
        elif not clip_url:
            warnings.append(f"Scene {idx}: no clip URL — skipping")
            scores, reasons = [], []
        else:
            frames = extract_frames(None, clip_url, n_frames=n_frames, tmp_dir=tmp)
            scores, reasons = [], []
            for frame in frames:
                try:
                    sc, reason = await score_frame_with_groq(
                        frame, narration, model, api_key,
                    )
                    scores.append(sc)
                    reasons.append(reason)
                except Exception as exc:
                    warnings.append(f"Scene {idx}: VLM eval error: {exc}")
                    scores.append(3)  # neutral fallback
                    reasons.append("error")

        eval_score = sum(scores) / len(scores) if scores else 0.0

        results.append(SceneEvalResult(
            scene_index=idx,
            narration=narration,
            clip_id=clip_id,
            eval_score=round(eval_score, 3),
            eval_scores_per_frame=scores,
            eval_reasons=reasons,
            human_score=human_ratings.get(idx),
            selection_score=selection_score,
        ))

    # ── Compute aggregate metrics ─────────────────────────────────────
    report = EvalReport(
        scene_results=results, warnings=warnings,
        n_human_rated=sum(1 for r in results if r.human_score is not None),
    )

    rated = [(r.eval_score, r.human_score)
             for r in results if r.human_score is not None]
    if len(rated) >= 5:
        evals, humans = [x[0] for x in rated], [x[1] for x in rated]  # type: ignore[misc]
        report.spearman_rho = round(compute_spearman(evals, humans), 3)  # type: ignore[arg-type]
        report.mae = round(sum(abs(e - h) for e, h in zip(evals, humans)) / len(evals), 3)  # type: ignore[misc]
        within1 = sum(1 for e, h in zip(evals, humans) if abs(e - h) <= 1.0)  # type: ignore[misc]
        report.pct_within_1 = round(within1 / len(evals) * 100, 1)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="VLM evaluation of broll clips")
    parser.add_argument("--timeline", required=True, help="Path to timeline.json")
    parser.add_argument("--human", default="eval/human_ratings.csv",
                        help="Path to human_ratings.csv (optional)")
    parser.add_argument("--out", default="eval/eval_results.json",
                        help="Output path for eval_results.json")
    parser.add_argument("--frames", type=int, default=3,
                        help="Frames per clip to sample (default: 3)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock scores (no API calls)")
    args = parser.parse_args()

    os.environ.setdefault("GROQ_API_KEY", "placeholder")
    os.environ.setdefault("PIXABAY_API_KEY", "placeholder")

    from app.core.config import get_settings
    s = get_settings()

    api_key = os.environ.get("GROQ_API_KEY", "")
    model = os.environ.get("VLM_EVAL_MODEL", s.VLM_EVAL_MODEL)

    with open(args.timeline, encoding="utf-8") as f:
        timeline = json.load(f)

    human_ratings = load_human_ratings(args.human) if os.path.isfile(args.human) else {}

    report = asyncio.run(evaluate_timeline(
        timeline, model=model, api_key=api_key,
        human_ratings=human_ratings,
        n_frames=args.frames, use_mock=args.mock,
    ))

    # Save JSON
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "spearman_rho": report.spearman_rho,
            "mae": report.mae,
            "pct_within_1": report.pct_within_1,
            "n_human_rated": report.n_human_rated,
            "warnings": report.warnings,
            "scenes": [
                {
                    "scene_index": r.scene_index,
                    "eval_score": r.eval_score,
                    "human_score": r.human_score,
                    "selection_score": r.selection_score,
                    "clip_id": r.clip_id,
                }
                for r in report.scene_results
            ],
        }, f, indent=2)

    # Print Markdown table
    print(report.to_markdown_table())
    print(f"\n[eval] Results saved to {args.out}")


if __name__ == "__main__":
    main()
