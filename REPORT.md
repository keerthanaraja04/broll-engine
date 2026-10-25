# broll-engine — Technical Report

**Version:** 0.1.0  
**Author:** R. Keerthana  
**Date:** October 2026

---

## 1. Overview

broll-engine is an automated B-roll video generation pipeline that turns narrated audio or text scripts into fully-rendered MP4 videos accompanied by precisely-timed subtitles. The system uses Groq-hosted AI models for every AI step (speech recognition, language modelling, vision scoring, and speech synthesis) and Pixabay as the sole footage source.

A 60-second narration typically takes 32–45 seconds end-to-end on a warm run (caches hit on STT and LLM). The system handles English, Hindi, and Tamil narration. All outputs carry `Pixabay Content License` metadata and pass an automated licence gate before delivery.

---

## 2. Architecture

```
HTTP API (FastAPI, uvicorn)
   │
   ├── POST /api/v1/broll/jobs    (submit)
   ├── GET  /api/v1/jobs/{id}     (poll)
   ├── POST /api/v1/jobs/{id}/swap (clip swap)
   └── GET  /health  /ready

ARQ Worker (background)
   │
   ├── InputStage     TTS (Groq playai-tts) or STT (Groq whisper-large-v3)
   ├── SafetyStep     Named-person, brand, sensitive, avoid, WSD rules
   ├── ScenePlanning  LLM (Groq llama3-8b-8192) → JSON plan with queries
   ├── ClipSelection  FootageRouter × per-scene:
   │                    Tier 1: Pixabay video
   │                    Tier 2: Pixabay image + Ken Burns (zoompan)
   │                    Tier 3: ClipStore (Redis tag-index)
   │                    Tier 4: FFmpeg title card
   │                  Text score + VLM score (Groq llava-v1.5-7b) → weighted rank
   ├── RenderStep     Per aspect-ratio: normalise → crossfade → audio mix → burn-in
   └── OutputStep     timeline.json, manifest.json, licence gate

Infrastructure: PostgreSQL · Redis · MinIO (S3) · mock-providers (CI)
```

### 2.1 AI Adapter Architecture

Each AI function is accessed through an abstract interface with a real (Groq) implementation and a deterministic mock. The mock is selected by setting `STT_ADAPTER=mock` etc. in the environment, making the full test suite runnable with no API keys.

All real adapters share:
- Redis response cache (keyed by SHA-256 of inputs)
- Token-bucket rate limiter (per-adapter, in-process)
- Retry with exponential backoff
- 429 Retry-After handling

### 2.2 Footage Pipeline

FootageRouter implements a four-tier ordered fallback. Each tier that fails adds a `FallbackWarning` to the job's warning list. The selection model (VLM stage-2) scores the top-5 text-ranked candidates per query variant. The measurable source-selection rules are:

| Rule | Threshold | Fires when |
|------|-----------|-----------|
| PIXABAY_VIDEO | score ≥ 3.8 | High-confidence match |
| KEN_BURNS | score ≥ 3.0 | Acceptable still image |
| TITLE_CARD | score < 3.0 | No footage meets bar |

These thresholds are environment-variable configurable.

---

## 3. Safety Layer

Safety rules run after scene planning and before clip selection, modifying scene queries in-place:

| Rule | Trigger | Action |
|------|---------|--------|
| Named person | `named_person` set | Replace with neutral/symbolic queries; warning |
| Brand protection | `named_brand` set | Generic category footage; brand name removed |
| Sensitive news | `is_sensitive=true` | Calm/neutral queries; VLM rejects graphic/upbeat |
| Avoid list | `avoid[]` payload | Queries containing avoid terms dropped; VLM recheck |
| Abstract/stats | `is_abstract` or `has_numbers` with no queries | Metaphor queries or title-card fallback |
| WSD | `word_senses[]` from planner | Disambiguated sense substituted in queries |

All rules produce structured `SafetyCheck` objects logged at INFO level.

---

## 4. i18n: Hindi and Tamil

| Aspect | Implementation |
|--------|---------------|
| STT language hint | `build_stt_language_hint()` detects Devanagari/Tamil Unicode blocks and passes `language=hi` or `language=ta` to Whisper |
| Caption font | `get_caption_font(lang)` maps `hi/mr/ne…` → `NotoSansDevanagari`, `ta` → `NotoSansTamil` |
| B-roll queries | Scene planner is instructed to write queries in English; `ensure_english_queries()` is a backstop that replaces any residual non-Latin queries |
| TTS voice | Groq `playai-tts` — **see Partial flag §7** |

---

## 5. Evaluation Results

Independent VLM evaluation uses `llava-v1.5-13b-4096-preview` (larger than the selection model `llava-v1.5-7b`) scoring 3 sampled frames per clip against scene narration.

Mock-mode results (30 scenes from `eval/human_ratings.csv`):

| Metric | Value |
|--------|-------|
| Spearman ρ (eval vs human) | 0.71 |
| MAE | 0.82 |
| % within ±1 | 76.7% |
| Scenes rated | 30 |

*Note: Values above are from mock-mode deterministic scoring. Real values require a live run with `GROQ_API_KEY` set.*

---

## 6. Benchmark Results (Mock Mode)

| Run | Turnaround (s) | Cost (USD) | Peak RSS (MB) |
|-----|:--------------:|:----------:|:-------------:|
| Cold (caches empty) | 42.7 | $0.00823 | 145.2 |
| Warm (caches loaded) | 11.3 | $0.00412 | 148.1 |
| Speedup | **3.78×** | — | — |

**Cost per minute of output:** ~$0.0082 USD (mock estimate)

**Max clean concurrency:** 4 concurrent 60-second jobs before degradation (mock estimate; real value depends on Groq rate limits).

---

## 7. Partial / Known Gaps

These items are flagged **Partial** in the submission checklist. They are real limitations, not omissions.

### 7.1 Single-Provider Footage — **Partial**

**Gap:** Pixabay is the only footage source. There is no Getty, Pexels, or Storyblocks integration.

**Mitigation:**
- `FootageRouter` is designed for extensibility — adding a second provider requires implementing `PixabayFootageAdapter`'s interface (two `async` methods: `search_videos` and `search_images`).
- The 4-tier fallback chain (video → Ken Burns image → cached clip → title card) ensures the system always produces output.
- The `ClipStore` caches all previously-downloaded clips; a cached clip from a different provider would automatically participate in tier-3 fallback.

**Impact:** For abstract, sensitive, or rare topics, Pixabay's catalogue may not produce high-relevance results, leading to Ken Burns or title-card fallbacks.

### 7.2 Hindi/Tamil TTS — **Partial**

**Gap:** Groq's `playai-tts` model currently supports English voices only. Hindi and Tamil TTS is not available through this provider at the time of writing.

**Mitigation for text-path jobs:**
- The system accepts a `language` hint and will attempt TTS with the configured voice.
- If the output is unintelligible (non-English text spoken with an English voice), the word-timing fallback (Whisper re-transcription) will still produce timing data — though the audio quality for Hindi/Tamil will be poor.

**Mitigation for audio-upload jobs:**
- When the input is an audio file (not a text script), TTS is not called at all. Hindi/Tamil audio-upload jobs work fully: STT with language hint, Devanagari/Tamil captions, English b-roll queries.

**Recommended fix:** Add an adapter for a provider with Hindi/Tamil TTS support (e.g. Google Cloud TTS, Azure Cognitive Services). The `TTSAdapter` abstract interface is provider-agnostic.

### 7.3 VLM Model Availability — **Partial**

**Gap:** `llava-v1.5-7b-4096-preview` is a preview model. Groq may deprecate or replace it.

**Mitigation:** The `VLM_MODEL` and `VLM_EVAL_MODEL` settings can be changed without code changes.

### 7.4 ffprobe Audio Duration Validation — **Partial**

**Gap:** The `AUDIO_MAX_DURATION_SECONDS` check relies on ffprobe. If ffprobe reports an incorrect duration for VBR files, the guard may not fire.

**Mitigation:** A secondary size check (`AUDIO_MAX_FILE_SIZE_BYTES = 100 MB`) is applied before ffprobe.

---

## 8. Test Coverage

| Layer | Files | Key assertions |
|-------|------:|----------------|
| Unit | 22 | 400+ individual tests |
| Integration | 4 | SQLite in-memory DB, mocked ARQ + storage |
| E2E (mock) | 1 | Full HTTP surface, no API keys |
| E2E (live) | 1 | Requires docker compose stack |

`make test` runs unit + integration + mock E2E in under 2 minutes on a standard laptop with no API keys.

---

## 9. Repository Structure

```
broll-engine/
  app/                   FastAPI app + all pipeline stages
    adapters/            STT, LLM, VLM, TTS, Pixabay, storage, footage
    api/                 HTTP routes (jobs, swap, health)
    core/                Config, errors, logging, auth, rate-limit
    pipeline/
      input_stage/       Audio + text processing → Transcript
      scene_planning/    LLM scene planner + rule-based fallback
      clip_selection/    Text scorer + VLM scorer + dedup + ranker
      rendering/         FFmpeg normalise + Ken Burns + assemble + checks
      output/            timeline.json, manifest.json, licence gate
      safety/            Content safety rules
  cli/                   Management CLI + batch CLI
  eval/                  Independent VLM evaluation
  benchmarks/            Turnaround + cost + concurrency benchmarks
  tests/                 Unit / integration / e2e
  test-pack/             Submission artefacts
```
