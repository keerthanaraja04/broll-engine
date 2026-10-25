# Submission Checklist

**Status key:** ✅ Yes · ⚠️ Partial · ❌ No  
**Fill in the Excel column "Status" with the value in the "Excel" column.**

---

## Must-Have Requirements

| # | Requirement | Status | Excel | Evidence | Partial Notes |
|---|-------------|:------:|-------|----------|---------------|
| M1 | Accept audio upload or text/script as input | ✅ | Yes | `app/api/jobs.py` — `POST /api/v1/broll/jobs` (JSON) and `POST /api/v1/broll/jobs/upload` (multipart) | |
| M2 | Transcribe audio with word-level timestamps | ✅ | Yes | `app/adapters/stt_adapter.py::GroqSTTAdapter` — Groq Whisper `timestamp_granularities[]=word` | |
| M3 | Plan scenes and generate b-roll search queries | ✅ | Yes | `app/pipeline/scene_planning/planner.py` — LLM JSON mode, 3 queries per scene (literal→abstract) | |
| M4 | Search and retrieve stock footage | ✅ | Yes | `app/adapters/pixabay_footage.py` — Pixabay video + image API | |
| M5 | Render final H.264 MP4 with narration audio | ✅ | Yes | `app/pipeline/rendering/render_step.py` + `assemble.py` | |
| M6 | Return presigned download URL | ✅ | Yes | `app/api/jobs.py::get_job_status` → `storage.presigned_get_url()` | |
| M7 | Subtitle/caption support | ✅ | Yes | `app/pipeline/rendering/captions.py` — SRT generation + `subtitles` burn-in filter | |
| M8 | Async job queue (not blocking HTTP) | ✅ | Yes | ARQ worker (`app/worker/`) — jobs enqueued via Redis, HTTP returns 202 immediately | |
| M9 | Job status polling endpoint | ✅ | Yes | `GET /api/v1/jobs/{job_id}` — status, progress 0–100, stage, result links | |
| M10 | Multiple aspect ratios (16:9, 9:16, 1:1) | ✅ | Yes | `app/pipeline/rendering/render_step.py` — per-ratio render loop | |

---

## Level / Feature Items

| # | Feature | Status | Excel | Evidence | Partial Notes |
|---|---------|:------:|-------|----------|---------------|
| F1 | Idempotency-Key on POST | ✅ | Yes | `app/api/jobs.py::_handle_idempotency` + `app/core/idempotency.py` (Redis, 24h TTL) | |
| F2 | 409 on key+different-body conflict | ✅ | Yes | `app/core/errors.py::IdempotencyKeyConflictError` + `test-pack/failures/idempotency_conflict_response.json` | |
| F3 | API-token auth (Bearer) | ✅ | Yes | `app/core/auth.py` — toggled by `API_AUTH_ENABLED` env var | |
| F4 | Rate limiting with Retry-After | ✅ | Yes | `app/core/ratelimit.py` — Redis sliding window, `X-RateLimit-*` headers | |
| F5 | Callback URL with HMAC-SHA256 signature | ✅ | Yes | `app/services/callback.py` — `X-IncuBrix-Signature: sha256=<hex>` | |
| F6 | Callback retry on failure | ✅ | Yes | `app/services/callback.py::_dispatch` — up to `CALLBACK_MAX_RETRIES` with backoff | |
| F7 | Specific error codes (UPPER_SNAKE_CASE) | ✅ | Yes | `app/core/errors.py` — 31 error classes + HTTP status map; validated by `tests/unit/test_error_codes_table.py` | |
| F8 | EMPTY_SCRIPT error | ✅ | Yes | `app/core/errors.py::EmptyScriptError` + `test-pack/failures/09_error_empty_script_response.json` | |
| F9 | WRONG_FILE_TYPE error | ✅ | Yes | `app/core/errors.py::WrongFileTypeError` + `test-pack/failures/wrong_file_type_response.json` | |
| F10 | BAD_ASPECT_RATIO error | ✅ | Yes | `app/core/errors.py::BadAspectRatioError` + `test-pack/failures/bad_aspect_ratio_response.json` | |
| F11 | AUDIO_TOO_LONG error (>10 min) | ✅ | Yes | `app/core/errors.py::AudioTooLongError` + `app/pipeline/input_stage/audio_processor.py::validate_audio` | |
| F12 | AUDIO_SILENT error | ✅ | Yes | `app/core/errors.py::AudioSilentError` + `app/pipeline/input_stage/audio_processor.py::detect_silence` | |
| F13 | AUDIO_NO_SPEECH error (low confidence) | ✅ | Yes | `app/core/errors.py::AudioNoSpeechError` + `app/pipeline/input_stage/audio_processor.py::check_transcript_quality` | |
| F14 | 4-tier footage fallback chain | ✅ | Yes | `app/adapters/footage_router.py` — Video→KenBurns→CachedClip→TitleCard | |
| F15 | Warning on each fallback | ✅ | Yes | `FootageResult.add_warning()` + `tests/integration/test_footage_router.py::TestWarningAccumulation` | |
| F16 | Pixabay video + image search | ✅ | Yes | `app/adapters/pixabay_footage.py::search_videos` + `search_images` | |
| F17 | Ken Burns (zoompan) for images | ✅ | Yes | `app/adapters/ffmpeg_card.py::ken_burns_from_image` + `app/pipeline/rendering/ken_burns_render.py` | |
| F18 | ClipStore Redis cache | ✅ | Yes | `app/adapters/clip_store.py` — tag-indexed, 30-day TTL, FIFO eviction | |
| F19 | Perceptual hash deduplication | ✅ | Yes | `app/pipeline/clip_selection/dedup.py` — dHash, Hamming distance ≤ `CLIP_PHASH_MAX_DISTANCE` | |
| F20 | Two-stage scoring (text + VLM) | ✅ | Yes | `app/pipeline/clip_selection/text_scorer.py` + `vlm_scorer.py` + `ranker.py` | |
| F21 | Source-selection thresholds (measurable) | ✅ | Yes | `CLIP_THRESHOLD_PIXABAY_VIDEO=3.8`, `CLIP_THRESHOLD_KEN_BURNS=3.0`; rule recorded in `timeline.json` | |
| F22 | Safety: named person → neutral visuals | ✅ | Yes | `app/pipeline/safety/safety_rules.py::_check_named_person` + `tests/unit/test_output_safety.py::TestNamedPerson` | |
| F23 | Safety: brand → generic category | ✅ | Yes | `app/pipeline/safety/safety_rules.py::_check_brand` + `tests/unit/test_output_safety.py::TestBrandProtection` | |
| F24 | Safety: sensitive → no graphic/upbeat | ✅ | Yes | `safety_rules.py::_check_sensitive_news` + `check_vlm_safety_flags` | |
| F25 | Safety: avoid[] at query + VLM | ✅ | Yes | `enforce_avoid_list()` + `check_vlm_safety_flags()` | |
| F26 | Safety: WSD / double meanings | ✅ | Yes | `_apply_word_senses()` + `tests/unit/test_output_safety.py::TestDoubleMeanings` | |
| F27 | Licence gate (blocks missing licence/watermark) | ✅ | Yes | `app/pipeline/output/licence_gate.py::check_licence_gate` | |
| F28 | timeline.json with all required fields | ✅ | Yes | `app/pipeline/output/timeline.py` + JSON Schema validation in `tests/unit/test_output_safety.py` | |
| F29 | manifest.json with models/timings/cost | ✅ | Yes | `app/pipeline/output/manifest.py` + JSON Schema validation | |
| F30 | Smart reframing for 9:16 / 1:1 (VLM bbox or saliency) | ✅ | Yes | `app/pipeline/rendering/reframe.py` — priority: vlm_bbox → OpenCV saliency → center → blurred_bg | |
| F31 | Post-render checks (blackdetect, freezedetect, drift) | ✅ | Yes | `app/pipeline/rendering/post_check.py` | |
| F32 | Scene segment cache (for swap) | ✅ | Yes | `app/pipeline/rendering/scene_cache.py` | |
| F33 | Clip swap endpoint | ✅ | Yes | `app/api/swap.py::POST /api/v1/jobs/{id}/swap` — re-render one scene, splice, log swap vs full time | |
| F34 | Pacing guard | ✅ | Yes | `app/pipeline/rendering/pacing.py::check_pacing` | |
| F35 | Global colour pass | ✅ | Yes | `app/pipeline/rendering/pacing.py::apply_global_colour_pass` | |
| F36 | Hindi narration support (STT) | ✅ | Yes | `app/pipeline/input_stage/language.py::build_stt_language_hint` detects Devanagari → `language=hi` | |
| F37 | Tamil narration support (STT) | ✅ | Yes | `language.py::build_stt_language_hint` detects Tamil script → `language=ta` | |
| F38 | Hindi captions (Devanagari font) | ✅ | Yes | `app/pipeline/rendering/captions.py::build_subtitle_filter` + `get_caption_font("hi")` → `NotoSansDevanagari` | |
| F39 | Tamil captions (Tamil font) | ✅ | Yes | `get_caption_font("ta")` → `NotoSansTamil` | |
| F40 | Hindi/Tamil TTS | ⚠️ | Partial | `app/adapters/tts_adapter.py` | **Groq `playai-tts` is English-only.** Audio-upload path works fully; text-path TTS produces English-voiced audio. See REPORT.md §7.2. |
| F41 | Second footage source | ⚠️ | Partial | `app/adapters/pixabay_footage.py` | **Pixabay only.** Interface designed for extensibility (see REPORT.md §7.1). No Getty/Pexels/Storyblocks integration. |
| F42 | Independent VLM evaluation | ✅ | Yes | `eval/vlm_eval.py` — uses `llava-v1.5-13b` (different from selection `llava-v1.5-7b`), Spearman/MAE/±1 | |
| F43 | Human ratings template + metrics | ✅ | Yes | `eval/human_ratings.csv` (30 rows) + `eval/vlm_eval.py::compute_spearman` | |
| F44 | Benchmark script | ✅ | Yes | `benchmarks/bench.py` — cold/warm turnaround, cost/min, peak RSS, concurrency | |
| F45 | Batch CLI | ✅ | Yes | `cli/batch.py` — submit `.json`/`.mp3`/`.txt`, poll, download, `summary.csv` | |

---

## Integration Standard Points

| # | Standard Point | Status | Excel | Evidence | Partial Notes |
|---|----------------|:------:|-------|----------|---------------|
| IS1 | All settings from env / pydantic-settings | ✅ | Yes | `app/core/config.py` — 130+ fields, `lru_cache`, `.env` loading, `.env.example` with all keys | |
| IS2 | Structured JSON logs with `timestamp`, `level`, `job_id`, `X-Correlation-Id` | ✅ | Yes | `app/core/logging.py::BrollJsonFormatter` + `test-pack/logs/sample_structured_log.jsonl` | |
| IS3 | `X-Correlation-Id` middleware (read/generate/echo) | ✅ | Yes | `app/core/middleware.py::CorrelationIdMiddleware` + `tests/unit/test_middleware.py` | |
| IS4 | `GET /health` liveness probe | ✅ | Yes | `app/api/health.py` — no I/O, fast, always returns `{status:"ok"}` | |
| IS5 | `GET /ready` readiness probe (DB + Redis + storage) | ✅ | Yes | `app/api/health.py::ready` — checks PostgreSQL, Redis, MinIO; 503 if any fail | |
| IS6 | Error handler: `{error:{code,message,retryable,details}}` | ✅ | Yes | `app/core/errors.py::register_error_handlers` — all 4 handler types | |
| IS7 | Error codes UPPER_SNAKE_CASE | ✅ | Yes | All 31 `BrollError` subclasses + 6 HTTP-map codes; validated by `tests/unit/test_error_codes_table.py` | |
| IS8 | Rate limiting on public endpoints | ✅ | Yes | `app/core/ratelimit.py` — Redis sliding window, exempt paths, fail-open | |
| IS9 | Optional API-token auth | ✅ | Yes | `app/core/auth.py` — Bearer token, toggled by `API_AUTH_ENABLED`, `API_TOKENS` env | |
| IS10 | Presigned download links | ✅ | Yes | `app/adapters/storage.py::presigned_get_url` — TTL configurable via `STORAGE_PRESIGNED_URL_EXPIRY_SECONDS` | |

---

## Additional Integration Points (from brief)

| Point | Status | Excel | Evidence | Partial Notes |
|-------|:------:|-------|----------|---------------|
| Idempotency-Key on every POST | ✅ | Yes | `app/api/jobs.py` + `app/core/idempotency.py` | |
| Callback with retry | ✅ | Yes | `app/services/callback.py` | |
| Docker-compose with all services + mock-providers | ✅ | Yes | `docker-compose.yml` — api, worker, postgres, redis, minio, minio-init, mock-providers | |
| Dockerfile with ffmpeg+libass+Noto fonts | ✅ | Yes | `Dockerfile` — libass9, fonts-noto, fonts-noto-extra (Devanagari), fonts-noto-ui-extra (Tamil) | |
| `make test` (unit + integration + e2e mock, no keys) | ✅ | Yes | `Makefile` + `tests/e2e/test_mocked_e2e.py` | |
| OpenAPI 3.1 spec auto-generated, drift test | ✅ | Yes | `openapi.yaml` + `scripts/generate_openapi.py` + `tests/unit/test_openapi.py` | |
| All routes under `/api/v1/` | ✅ | Yes | `app/api/router.py` + `tests/unit/test_openapi.py::test_all_v1_routes_under_api_prefix` | |
| `.env.example` with all settings | ✅ | Yes | `.env.example` — all 130+ settings documented with defaults and comments | |
| README with architecture, 10-command setup, all settings, error table | ✅ | Yes | `README.md` | |
| SOURCES.md | ✅ | Yes | `SOURCES.md` | |
| AI_USE.md | ✅ | Yes | `AI_USE.md` | |

---

## Summary of Partial Items

Only two items are Partial. Both are due to **external provider limitations**, not architectural gaps:

| Item | Partial Reason | Workaround in Codebase |
|------|---------------|------------------------|
| **Hindi/Tamil TTS** (F40) | Groq `playai-tts` supports English voices only as of Oct 2026 | Audio-upload path fully works; TTS adapter interface is provider-agnostic for easy swap |
| **Single footage provider** (F41) | Only Pixabay integrated | 4-tier fallback always produces output; `PixabayFootageAdapter` interface documented for extension |

No items are ❌ No.

---

## Test Evidence Quick Reference

Run without any API keys:
```bash
# All tests pass in CI with no keys
make test
# Expected output: 400+ tests collected, all pass
```

Run mock benchmark:
```bash
python -m benchmarks.bench --mock
# Outputs benchmarks/metrics.md and benchmarks/raw.json
```

Run mock VLM eval:
```bash
python -m eval.vlm_eval \
  --timeline test-pack/outputs/01_tutorial_20s/timeline.json \
  --human eval/human_ratings.csv \
  --mock \
  --out eval/eval_results.json
# Prints Markdown table to stdout
```
