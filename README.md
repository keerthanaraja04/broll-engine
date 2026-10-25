# broll-engine

B-roll video generation engine powered by Groq and Pixabay.

## Architecture

```
                              ┌─────────────────────────────────────────────────┐
                              │                    API Layer                     │
                              │                                                  │
   POST /api/v1/broll/jobs ──►│  FastAPI (uvicorn)   GET /health   GET /ready   │
   GET  /api/v1/jobs/{id} ───►│  Rate-limit middleware  CorrelationId middleware │
                              │  Idempotency-Key   API-token auth               │
                              └──────────────┬──────────────────────────────────┘
                                             │ enqueue
                                             ▼
                              ┌─────────────────────────────────────────────────┐
                              │                 Worker (ARQ)                      │
                              │                                                 │
                              │  InputStage → SafetyStep → ScenePlanning →       │
                              │  ClipSelection → RenderStep → OutputStep         │
                              └──┬───────┬───────┬───────┬───────┬───────────────┘
                                 │       │       │       │       │
                          ┌──────▼──┐ ┌──▼───┐ ┌─▼────┐ ┌▼─────┐ ┌▼──────────┐
                          │ STT     │ │ LLM  │ │ VLM  │ │ TTS  │ │ Footage     │
                          │ Adapter │ │Adapter│ │Adapter│ │Adapter│ │ Router      │
                          │(Groq)   │ │(Groq)│ │(Groq)│ │(Groq)│ │(Pixabay→    │
                          │         │ │      │ │      │ │      │ │ Ken Burns → │
                          │ cache   │ │cache │ │cache │ │cache │ │ Cache → Card)│
                          │ rate-   │ │rate- │ │rate- │ │rate- │ │             │
                          │ limit   │ │limit │ │limit │ │limit │ │             │
                          └─────────┘ └──────┘ └──────┘ └──────┘ └─────────────┘
                                 │       │       │       │       │
                          ┌──────▼───────▼───────▼───────▼───────▼──────────────┐
                          │          Infrastructure                          │
                          │                                                 │
                          │  PostgreSQL    Redis    MinIO (S3)    ffmpeg     │
                          │  (jobs table)  (queue   (raw/processed  (render)  │
                          │                +cache)  /final buckets)         │
                          └─────────────────────────────────────────────────┘
```

## Setup (10 commands)

```bash
# 1. Clone
git clone <repo-url> broll-engine && cd broll-engine

# 2. Copy env
cp .env.example .env

# 3. Install dev deps
pip install -e ".[dev]"

# 4. Generate OpenAPI spec
python scripts/generate_openapi.py

# 5. Run unit tests
make test-unit

# 6. Run integration tests
make test-integration

# 7. Build Docker images
docker compose build

# 8. Start the stack
docker compose up -d

# 9. Check health
curl http://localhost:8000/health

# 10. Check readiness
curl http://localhost:8000/ready
```

## API Routes

All API routes are under `/api/v1/`:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/broll/jobs` | Submit a job (JSON body: audio_url, script, or scenes) |
| POST | `/api/v1/broll/jobs/upload` | Submit a job (multipart audio upload) |
| GET | `/api/v1/jobs/{job_id}` | Get job status, progress, and result links |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (checks PostgreSQL, Redis, MinIO) |

## Settings

All settings are read from environment variables. See `.env.example` for the full list with defaults.

### Required Secrets

| Setting | Description |
|---------|-------------|
| `GROQ_API_KEY` | Groq API key |
| `PIXABAY_API_KEY` | Pixabay API key |

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `APP_ENV` | development | development \| staging \| production |
| `STT_ADAPTER` | groq | groq \| mock |
| `LLM_ADAPTER` | groq | groq \| mock |
| `VLM_ADAPTER` | groq | groq \| mock |
| `TTS_ADAPTER` | groq | groq \| mock |
| `STT_MODEL` | whisper-large-v3 | Groq Whisper model |
| `LLM_JSON_MODEL` | llama3-8b-8192 | Groq chat model for JSON |
| `VLM_MODEL` | llava-v1.5-7b-4096-preview | Groq vision model |
| `TTS_MODEL` | playai-tts | Groq TTS model |
| `AUDIO_MAX_DURATION_SECONDS` | 600.0 | Max audio duration (10 min) |
| `SCENE_MIN_DURATION_SECONDS` | 3.0 | Min scene length |
| `SCENE_MAX_DURATION_SECONDS` | 8.0 | Max scene length |
| `CLIP_THRESHOLD_PIXABAY_VIDEO` | 3.8 | Score threshold for Pixabay video |
| `CLIP_THRESHOLD_KEN_BURNS` | 3.0 | Score threshold for Ken Burns image |
| `RENDER_FPS` | 30 | Output frame rate |
| `RENDER_PIXEL_FORMAT` | yuv420p | Output pixel format |
| `RENDER_CROSSFADE_DURATION` | 0.3 | Crossfade between scenes (seconds) |
| `RENDER_REFRAME_METHOD` | saliency | vlm_bbox \| saliency \| center \| blurred_bg |
| `RATE_LIMIT_ENABLED` | true | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` | 60 | Requests per window |
| `RATE_LIMIT_WINDOW_SECONDS` | 60 | Rate limit window |
| `API_AUTH_ENABLED` | false | Enable API token auth |

## Error Codes

Every error response uses this shape:

```json
{"error": {"code": "UPPER_SNAKE_CASE", "message": "...", "retryable": false, "details": {}}}
```

| Code | HTTP | Retryable | Description |
|------|------|-----------|-------------|
| INTERNAL_ERROR | 500 | false | Unexpected internal error |
| NOT_FOUND | 404 | false | Resource not found |
| VALIDATION_ERROR | 422 | false | Request validation failed |
| CONFLICT | 409 | false | Resource conflict |
| UNAUTHORIZED | 401 | false | Missing or invalid auth |
| FORBIDDEN | 403 | false | Access denied |
| RATE_LIMIT_EXCEEDED | 429 | true | Too many requests |
| EXTERNAL_SERVICE_ERROR | 502 | true | External API failure |
| GROQ_SERVICE_ERROR | 502 | true | Groq API failure |
| PIXABAY_SERVICE_ERROR | 502 | true | Pixabay API failure |
| STORAGE_ERROR | 500 | true | MinIO/S3 storage failure |
| PIPELINE_ERROR | 500 | false | Pipeline step failed |
| JOB_NOT_FOUND | 404 | false | Job ID not found |
| MEDIA_PROCESSING_ERROR | 500 | false | FFmpeg processing failure |
| SERVICE_UNAVAILABLE | 503 | true | Service unavailable |
| EMPTY_SCRIPT | 422 | false | Script is empty or whitespace |
| WRONG_FILE_TYPE | 415 | false | Unsupported file extension |
| BAD_ASPECT_RATIO | 422 | false | Aspect ratio not in 16:9/9:16/1:1 |
| AUDIO_TOO_LONG | 422 | false | Audio exceeds 10 minutes |
| IDEMPOTENCY_KEY_CONFLICT | 409 | false | Same key, different body |
| FOOTAGE_UNAVAILABLE | 422 | false | No footage found after all fallbacks |
| CORRUPTED_DOWNLOAD | 502 | true | Downloaded file failed validation |
| DOWNLOAD_SIZE_LIMIT | 413 | false | File exceeds max download size |
| FFPROBE_VALIDATION_ERROR | 422 | true | ffprobe found no valid video stream |
| PIXABAY_AUTH_ERROR | 401 | false | Pixabay API key invalid |
| FOOTAGE_ROUTER_EXHAUSTED | 500 | false | All 4 footage tiers failed |
| AUDIO_SILENT | 422 | false | Audio is all silence |
| AUDIO_NO_SPEECH | 422 | false | Transcript is empty or low-confidence |
| RENDER_ERROR | 500 | false | Rendering stage failure |
| POST_RENDER_CHECK_FAILED | 500 | false | Post-render quality check failed |
| LICENCE_GATE_FAILED | 422 | false | Clip lacks licence/attribution or is watermarked |
| SAFETY_VIOLATION | 422 | false | Safety rule violated |
| LLM_SCHEMA_ERROR | 422 | true | LLM response did not match JSON schema |
| SCENE_PLAN_VALIDATION_ERROR | 422 | true | LLM scene plan is malformed |
| BAD_REQUEST | 400 | false | HTTP 400 fallback |
| METHOD_NOT_ALLOWED | 405 | false | HTTP 405 fallback |
| UNPROCESSABLE_ENTITY | 422 | false | HTTP 422 fallback |
| BAD_GATEWAY | 502 | true | HTTP 502 fallback |
| GATEWAY_TIMEOUT | 504 | true | HTTP 504 fallback |
| HTTP_ERROR | — | false | Unknown HTTP error fallback |

## Batch CLI

```bash
python -m cli.batch --input ./inputs --api http://localhost:8000 --out ./results
```

Accepts `.json` (job body), `.mp3`/`.wav`/`.m4a` (audio upload), `.txt` (script).
Outputs `summary.csv` + per-job directories with video, SRT, timeline, manifest.

## Testing

```bash
make test          # unit + integration + e2e (mocks only)
make test-unit     # unit tests only
make test-integration  # integration tests only
```
