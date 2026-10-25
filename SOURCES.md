# Sources, Licences, and Attributions

## AI Models and Providers

### Groq (https://groq.com)

All AI inference is provided by Groq's LPU-accelerated API.

| Model | Purpose | Licence |
|-------|---------|---------|
| `whisper-large-v3` | Speech-to-text (STT) — transcription + word timestamps | OpenAI Whisper (MIT) weights; Groq API terms |
| `llama3-8b-8192` | Scene planning (JSON mode), keyword extraction | Meta Llama 3 Community Licence |
| `llava-v1.5-7b-4096-preview` | VLM clip scoring (selection stage) | LLaVA licence (see below) |
| `llava-v1.5-13b-4096-preview` | VLM clip scoring (independent evaluation) | LLaVA licence |
| `playai-tts` / `Arista-PlayAI` | Text-to-speech synthesis | PlayAI proprietary; used via Groq API |

**LLaVA licence note:** LLaVA-1.5 is released under the LLaVA model licence, which permits research and commercial use subject to the LLaMA 2 Community Licence terms. See: https://github.com/haotian-liu/LLaVA/blob/main/LICENSE

**Groq API Terms:** https://wow.groq.com/terms-of-use/

---

## Footage Source

### Pixabay (https://pixabay.com)

All stock footage and images are sourced from Pixabay under the **Pixabay Content Licence**.

**Pixabay Content Licence summary:**
- Free to use for commercial and non-commercial purposes
- No attribution required (though we provide it as good practice)
- Cannot be redistributed as a standalone stock-footage collection
- Full terms: https://pixabay.com/service/license-summary/

Every clip included in a generated video:
- Has `license: "Pixabay Content License"` in `timeline.json`
- Has `commercial_use: true`
- Has `attribution: "Video/Image by <username> on Pixabay"`
- Is validated by the licence gate before delivery (raises `LICENCE_GATE_FAILED` if missing)

---

## Fonts

All fonts used for subtitle rendering are from the **Noto font family** by Google.

| Font | Script | Licence |
|------|--------|---------|
| NotoSans | Latin, Greek, Cyrillic | SIL Open Font Licence 1.1 |
| NotoSansDevanagari | Devanagari (Hindi, Marathi) | SIL Open Font Licence 1.1 |
| NotoSansTamil | Tamil | SIL Open Font Licence 1.1 |

**SIL OFL 1.1:** https://scripts.sil.org/OFL  
**Noto fonts:** https://fonts.google.com/noto

---

## Open-Source Libraries

### Core Framework

| Library | Version | Licence |
|---------|---------|---------|
| FastAPI | 0.111.0 | MIT |
| Pydantic | 2.7.1 | MIT |
| pydantic-settings | 2.2.1 | MIT |
| uvicorn | 0.29.0 | BSD-3-Clause |
| SQLAlchemy | 2.0.30 | MIT |
| asyncpg | 0.29.0 | Apache-2.0 |
| alembic | 1.13.1 | MIT |

### Queue and Cache

| Library | Version | Licence |
|---------|---------|---------|
| redis (async) | 5.0.4 | MIT |
| arq | 0.25.0 | MIT |

### Storage

| Library | Version | Licence |
|---------|---------|---------|
| minio | 7.2.7 | Apache-2.0 |
| boto3 | 1.34.101 | Apache-2.0 |

### HTTP

| Library | Version | Licence |
|---------|---------|---------|
| httpx | 0.27.0 | BSD-3-Clause |

### Logging and Utilities

| Library | Version | Licence |
|---------|---------|---------|
| python-json-logger | 2.0.7 | BSD-2-Clause |
| structlog | 24.1.0 | MIT |
| tenacity | 8.3.0 | Apache-2.0 |
| jsonschema | 4.22.0 | MIT |
| PyYAML | 6.0.1 | MIT |
| python-multipart | 0.0.9 | Apache-2.0 |
| python-dotenv | 1.0.1 | BSD-3-Clause |

### CLI

| Library | Version | Licence |
|---------|---------|---------|
| typer | 0.12.3 | MIT |
| rich | 13.7.1 | MIT |

### Development / Testing

| Library | Version | Licence |
|---------|---------|---------|
| pytest | 8.2.0 | MIT |
| pytest-asyncio | 0.23.6 | Apache-2.0 |
| pytest-httpx | 0.30.0 | MIT |
| pytest-cov | 5.0.0 | MIT |
| aiosqlite | 0.20.0 | MIT |
| factory-boy | 3.3.0 | MIT |
| ruff | 0.4.4 | MIT |
| mypy | 1.10.0 | MIT |

### System Dependencies (Docker image)

| Package | Version | Licence |
|---------|---------|---------|
| FFmpeg | latest (Debian) | LGPL-2.1 / GPL-2.0 (encoder-dependent) |
| libass | latest | ISC |
| PostgreSQL | 16 (Alpine) | PostgreSQL Licence |
| Redis | 7 (Alpine) | BSD-3-Clause |
| MinIO | latest | AGPL-3.0 |

**FFmpeg Licence note:** The FFmpeg binary distributed in the Dockerfile may
include GPL-licensed components (e.g. libx264). The compiled binary is used
as a standalone process (not linked), which is consistent with LGPL use when
dynamically linked, but please verify against your deployment policy.
See: https://ffmpeg.org/legal.html

---

## Test Fixtures

The CC0 test fixtures in `tests/fixtures/` (`sample_audio.wav`,
`clip_city.jpg`, `clip_nature.jpg`, `clip_abstract.jpg`) were generated
programmatically using pure Python arithmetic (sine wave, solid-colour JPEG).
They are in the public domain and carry no copyright.
