# Test-Pack Outputs

This directory contains the output artefacts produced by running each
input from `../inputs/` through the broll-engine API.

## Directory Structure

```
outputs/
  01_tutorial_20s/
    job_id.txt           — UUID of the submitted job
    timeline.json        — per-scene clip selection and scores
    manifest.json        — models, settings, timings, cost estimate
    video_16_9.mp4       — rendered output (generated on real run)
    subtitles_16_9.srt   — subtitle file
  02_news_90s/
    ...
  ...
```

## How to Generate

```bash
# With the stack running:
python -m cli.batch --input test-pack/inputs --api http://localhost:8000 \
       --out test-pack/outputs

# This produces test-pack/outputs/summary.csv with all job statuses.
```

## Mock Outputs (CI)

The files in this directory are representative examples generated using
mock adapters (`STT_ADAPTER=mock`, `LLM_ADAPTER=mock`, `VLM_ADAPTER=mock`,
`TTS_ADAPTER=mock`) and an in-memory SQLite database.

Real outputs (video files) require a running docker-compose stack with
valid `GROQ_API_KEY` and `PIXABAY_API_KEY`.
