"""
Central application configuration.

All values are read from environment variables (or a .env file).
Only two secrets are required: GROQ_API_KEY and PIXABAY_API_KEY.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_NAME: str = "broll-engine"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_VERSION: str = "0.1.0"
    APP_DEBUG: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_WORKERS: int = 1
    APP_ROOT_PATH: str = ""

    # ------------------------------------------------------------------
    # Secrets (required)
    # ------------------------------------------------------------------
    GROQ_API_KEY: str = Field(..., description="Groq API key — required")
    PIXABAY_API_KEY: str = Field(..., description="Pixabay API key — required")

    # ------------------------------------------------------------------
    # Groq / LLM
    # ------------------------------------------------------------------
    GROQ_BASE_URL: AnyHttpUrl = AnyHttpUrl("https://api.groq.com/openai/v1")
    GROQ_CHAT_MODEL: str = "llama3-8b-8192"
    GROQ_WHISPER_MODEL: str = "whisper-large-v3"
    GROQ_MAX_TOKENS: int = 2048
    GROQ_TEMPERATURE: float = Field(default=0.3, ge=0.0, le=2.0)
    GROQ_REQUEST_TIMEOUT_SECONDS: float = 60.0
    GROQ_MAX_RETRIES: int = 3

    # ------------------------------------------------------------------
    # Pixabay
    # ------------------------------------------------------------------
    PIXABAY_BASE_URL: AnyHttpUrl = AnyHttpUrl("https://pixabay.com/api")
    PIXABAY_VIDEO_BASE_URL: AnyHttpUrl = AnyHttpUrl("https://pixabay.com/api/videos")
    PIXABAY_RESULTS_PER_PAGE: int = Field(default=10, ge=1, le=200)
    PIXABAY_MIN_WIDTH: int = 1280
    PIXABAY_MIN_HEIGHT: int = 720
    PIXABAY_VIDEO_TYPE: str = "film"  # film | animation | all

    # ------------------------------------------------------------------
    # Score / relevance thresholds
    # ------------------------------------------------------------------
    BROLL_MIN_RELEVANCE_SCORE: float = Field(default=0.6, ge=0.0, le=1.0)
    BROLL_MAX_CLIP_DURATION_SECONDS: float = 30.0
    BROLL_MIN_CLIP_DURATION_SECONDS: float = 2.0
    BROLL_MAX_CLIPS_PER_JOB: int = 20
    BROLL_TARGET_CLIPS_PER_SEGMENT: int = 3

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    PIPELINE_CONCURRENCY: int = 4
    PIPELINE_STEP_TIMEOUT_SECONDS: float = 300.0
    PIPELINE_MAX_RETRIES: int = 2

    # ------------------------------------------------------------------
    # FFmpeg / Media
    # ------------------------------------------------------------------
    FFMPEG_BINARY: str = "ffmpeg"
    FFPROBE_BINARY: str = "ffprobe"
    FFMPEG_THREADS: int = 0  # 0 = auto
    MEDIA_OUTPUT_FORMAT: str = "mp4"
    MEDIA_VIDEO_CODEC: str = "libx264"
    MEDIA_AUDIO_CODEC: str = "aac"
    MEDIA_CRF: int = Field(default=23, ge=0, le=51)
    MEDIA_PRESET: str = "medium"
    FONT_DIR: str = "/usr/share/fonts/truetype/noto"
    SUBTITLE_FONT: str = "NotoSans"

    # ------------------------------------------------------------------
    # Database (PostgreSQL)
    # ------------------------------------------------------------------
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "broll"
    POSTGRES_USER: str = "broll"
    POSTGRES_PASSWORD: str = "broll"
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 20
    POSTGRES_POOL_TIMEOUT_SECONDS: float = 30.0
    POSTGRES_ECHO_SQL: bool = False
    DATABASE_URL: str | None = None

    @property
    def database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ------------------------------------------------------------------
    # Redis
    # ------------------------------------------------------------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    REDIS_MAX_CONNECTIONS: int = 20
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 5.0
    REDIS_JOB_TTL_SECONDS: int = 86400  # 24 h

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # ------------------------------------------------------------------
    # Object Storage (MinIO / S3-compatible)
    # ------------------------------------------------------------------
    STORAGE_ENDPOINT: str = "localhost:9000"
    STORAGE_ACCESS_KEY: str = "minioadmin"
    STORAGE_SECRET_KEY: str = "minioadmin"
    STORAGE_BUCKET_RAW: str = "broll-raw"
    STORAGE_BUCKET_PROCESSED: str = "broll-processed"
    STORAGE_BUCKET_FINAL: str = "broll-final"
    STORAGE_REGION: str = "us-east-1"
    STORAGE_USE_SSL: bool = False
    STORAGE_PRESIGNED_URL_EXPIRY_SECONDS: int = 3600

    # ------------------------------------------------------------------
    # ARQ / Worker
    # ------------------------------------------------------------------
    WORKER_QUEUE_NAME: str = "broll:default"
    WORKER_MAX_JOBS: int = 10
    WORKER_POLL_DELAY_SECONDS: float = 0.5
    WORKER_JOB_TIMEOUT_SECONDS: int = 600
    WORKER_KEEP_RESULT_SECONDS: int = 86400
    WORKER_RETRY_JOBS: bool = True
    WORKER_MAX_TRIES: int = 3

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_INCLUDE_TRACE: bool = False

    # ------------------------------------------------------------------
    # Rate limiting (sliding-window, Redis-backed)
    # ------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 60          # max requests per window
    RATE_LIMIT_WINDOW_SECONDS: int = 60    # window size in seconds
    # Paths exempt from rate limiting (comma-separated prefixes)
    RATE_LIMIT_EXEMPT_PATHS: list[str] = ["/health", "/ready", "/docs", "/openapi.json"]

    # ------------------------------------------------------------------
    # API Token authentication
    # ------------------------------------------------------------------
    API_AUTH_ENABLED: bool = False         # set True in staging/production
    # Comma-separated list of valid static bearer tokens.
    # In production, replace with a DB-backed token table.
    API_TOKENS: list[str] = []
    API_TOKEN_HEADER: str = "Authorization"

    # ------------------------------------------------------------------
    # Idempotency
    # ------------------------------------------------------------------
    IDEMPOTENCY_TTL_SECONDS: int = 86400   # 24 h — how long keys are remembered
    IDEMPOTENCY_REDIS_PREFIX: str = "idem:"

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------
    AUDIO_MAX_DURATION_SECONDS: float = 600.0   # 10 minutes
    AUDIO_ALLOWED_EXTENSIONS: list[str] = ["mp3", "mp4", "wav", "m4a", "ogg", "webm", "flac"]
    AUDIO_MAX_FILE_SIZE_BYTES: int = 104_857_600  # 100 MB
    VALID_ASPECT_RATIOS: list[str] = ["16:9", "9:16", "1:1"]
    JOB_PAGE_SIZE_DEFAULT: int = 20
    JOB_PAGE_SIZE_MAX: int = 100

    # ------------------------------------------------------------------
    # Adapter implementation selectors
    # Set to "mock" to skip real API calls (useful in CI / unit tests)
    # ------------------------------------------------------------------
    STT_ADAPTER: str = "groq"    # groq | mock
    LLM_ADAPTER: str = "groq"    # groq | mock
    VLM_ADAPTER: str = "groq"    # groq | mock
    TTS_ADAPTER: str = "groq"    # groq | mock

    # ------------------------------------------------------------------
    # STT (Speech-to-Text) — Groq Whisper
    # ------------------------------------------------------------------
    STT_MODEL: str = "whisper-large-v3"
    STT_WORD_TIMESTAMPS: bool = True          # request word-level timestamps
    STT_TIMEOUT_SECONDS: float = 120.0
    STT_MAX_RETRIES: int = 3
    STT_CACHE_TTL_SECONDS: int = 86400        # 24 h (transcriptions are expensive)
    STT_CACHE_ENABLED: bool = True
    # Token-bucket rate limiter for STT calls
    STT_RATE_LIMIT_TOKENS: float = 20.0       # burst capacity
    STT_RATE_LIMIT_REFILL_RATE: float = 0.5   # tokens/second (~30 RPM)

    # ------------------------------------------------------------------
    # LLM (Language Model) — Groq Chat
    # ------------------------------------------------------------------
    LLM_MODEL: str = "llama3-8b-8192"
    LLM_JSON_MODEL: str = "llama3-8b-8192"   # model used for JSON-mode completions
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 3
    LLM_SCHEMA_REPAIR_RETRIES: int = 2        # extra retries with error feedback for JSON repair
    LLM_CACHE_TTL_SECONDS: int = 3600         # 1 h
    LLM_CACHE_ENABLED: bool = True
    LLM_RATE_LIMIT_TOKENS: float = 60.0
    LLM_RATE_LIMIT_REFILL_RATE: float = 2.0   # tokens/second (~120 RPM)

    # ------------------------------------------------------------------
    # VLM (Vision-Language Model) — Groq vision
    # ------------------------------------------------------------------
    VLM_MODEL: str = "llava-v1.5-7b-4096-preview"
    VLM_MAX_TOKENS: int = 512
    VLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    VLM_TIMEOUT_SECONDS: float = 60.0
    VLM_MAX_RETRIES: int = 3
    VLM_CACHE_TTL_SECONDS: int = 7200         # 2 h
    VLM_CACHE_ENABLED: bool = True
    VLM_RATE_LIMIT_TOKENS: float = 30.0
    VLM_RATE_LIMIT_REFILL_RATE: float = 0.5
    # Groq vision limits: 1 image per request, max 4 MB base64
    VLM_IMAGES_PER_REQUEST: int = 1
    VLM_MAX_IMAGE_BYTES: int = 4_000_000      # 4 MB
    VLM_IMAGE_RESIZE_MAX_PX: int = 1024       # resize images larger than this before sending

    # ------------------------------------------------------------------
    # TTS (Text-to-Speech) — Groq TTS
    # ------------------------------------------------------------------
    TTS_MODEL: str = "playai-tts"
    TTS_DEFAULT_VOICE: str = "Arista-PlayAI"
    TTS_AUDIO_FORMAT: str = "wav"             # wav | mp3 | flac | opus
    TTS_SAMPLE_RATE: int = 24000
    TTS_TIMEOUT_SECONDS: float = 120.0
    TTS_MAX_RETRIES: int = 3
    TTS_CACHE_TTL_SECONDS: int = 86400        # 24 h
    TTS_CACHE_ENABLED: bool = True
    TTS_RATE_LIMIT_TOKENS: float = 20.0
    TTS_RATE_LIMIT_REFILL_RATE: float = 0.33  # tokens/second (~20 RPM)
    # When Groq TTS does not return word timings, run output through Whisper
    TTS_FALLBACK_WORD_TIMINGS: bool = True

    # ------------------------------------------------------------------
    # Adapter shared cache
    # ------------------------------------------------------------------
    ADAPTER_CACHE_REDIS_PREFIX: str = "adapter:"
    # Set to false to disable all adapter caches (useful in tests)
    ADAPTER_CACHE_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Pixabay Footage adapter
    # ------------------------------------------------------------------
    PIXABAY_FOOTAGE_RATE_LIMIT_TOKENS: float = 100.0   # burst: 100 req
    PIXABAY_FOOTAGE_RATE_LIMIT_REFILL: float = 1.667   # ~100 req/60 s sustain
    PIXABAY_FOOTAGE_SEARCH_CACHE_TTL: int = 86400       # 24 h
    PIXABAY_FOOTAGE_SEARCH_CACHE_ENABLED: bool = True
    PIXABAY_FOOTAGE_CACHE_PREFIX: str = "footage:"
    PIXABAY_FOOTAGE_TIMEOUT_SECONDS: float = 30.0
    PIXABAY_FOOTAGE_MAX_RETRIES: int = 3
    # Rendition preference order for target resolution
    PIXABAY_FOOTAGE_TARGET_WIDTH: int = 1920
    PIXABAY_FOOTAGE_TARGET_HEIGHT: int = 1080
    PIXABAY_FOOTAGE_FALLBACK_WIDTH: int = 1280
    PIXABAY_FOOTAGE_FALLBACK_HEIGHT: int = 720

    # ------------------------------------------------------------------
    # Media download / validation
    # ------------------------------------------------------------------
    FOOTAGE_DOWNLOAD_MAX_BYTES: int = 500_000_000       # 500 MB hard cap
    FOOTAGE_DOWNLOAD_CHUNK_SIZE: int = 1_048_576         # 1 MB streaming chunks
    FOOTAGE_DOWNLOAD_MAX_RETRIES: int = 3
    FOOTAGE_DOWNLOAD_TIMEOUT_SECONDS: float = 120.0
    # ffprobe: min acceptable video stream properties after download
    FOOTAGE_MIN_VIDEO_WIDTH: int = 640
    FOOTAGE_MIN_VIDEO_HEIGHT: int = 360
    FOOTAGE_MIN_DURATION_SECONDS: float = 1.0

    # ------------------------------------------------------------------
    # Clip store (local Redis-backed catalogue of downloaded clips)
    # ------------------------------------------------------------------
    CLIP_STORE_REDIS_PREFIX: str = "clipstore:"
    CLIP_STORE_INDEX_KEY: str = "clipstore:index"
    CLIP_STORE_TTL_SECONDS: int = 2_592_000             # 30 days
    CLIP_STORE_MAX_ENTRIES: int = 10_000

    # ------------------------------------------------------------------
    # FFmpeg title card (tier-4 fallback)
    # ------------------------------------------------------------------
    TITLE_CARD_DURATION_SECONDS: float = 5.0
    TITLE_CARD_WIDTH: int = 1920
    TITLE_CARD_HEIGHT: int = 1080
    TITLE_CARD_BG_COLOR: str = "0x1a1a2e"
    TITLE_CARD_FONT_SIZE: int = 64
    TITLE_CARD_FONT_COLOR: str = "white"

    # ------------------------------------------------------------------
    # FootageRouter
    # ------------------------------------------------------------------
    FOOTAGE_ROUTER_ENABLE_VIDEO: bool = True
    FOOTAGE_ROUTER_ENABLE_KEN_BURNS: bool = True
    FOOTAGE_ROUTER_ENABLE_CACHE_FALLBACK: bool = True
    FOOTAGE_ROUTER_ENABLE_TITLE_CARD: bool = True
    # Ken Burns zoom-pan filter parameters
    KEN_BURNS_ZOOM_START: float = 1.0
    KEN_BURNS_ZOOM_END: float = 1.15
    KEN_BURNS_DURATION_SECONDS: float = 6.0

    # ------------------------------------------------------------------
    # Callback / Webhook
    # ------------------------------------------------------------------
    CALLBACK_HMAC_SECRET: str = ""         # leave blank to skip HMAC signing
    CALLBACK_MAX_RETRIES: int = 3
    CALLBACK_RETRY_DELAY_SECONDS: float = 5.0
    CALLBACK_TIMEOUT_SECONDS: float = 10.0
    CALLBACK_SIGNATURE_HEADER: str = "X-IncuBrix-Signature"

    # ------------------------------------------------------------------
    # Input stage — audio processing
    # ------------------------------------------------------------------
    # Groq Whisper file-size limit (25 MB for the API, keep headroom)
    AUDIO_GROQ_MAX_BYTES: int = 24_000_000          # 24 MB — files over this are chunked

    # ffmpeg silencedetect filter parameters
    SILENCE_DETECT_NOISE_DB: float = -40.0           # dBFS threshold for silence
    SILENCE_DETECT_MIN_DURATION: float = 0.5         # minimum silence duration in seconds
    # A file is considered silent when the ratio of silent-to-total duration exceeds this
    SILENCE_DETECT_SILENT_RATIO: float = 0.95

    # Audio chunking (applied when file > AUDIO_GROQ_MAX_BYTES)
    AUDIO_CHUNK_MAX_DURATION_SECONDS: float = 240.0  # 4 min max per chunk
    AUDIO_CHUNK_MIN_SILENCE_SECONDS: float = 0.3     # use silences >= this to split
    AUDIO_CHUNK_OVERLAP_SECONDS: float = 0.1         # overlap to avoid word clipping

    # Conversion target (all audio normalised to this before STT)
    AUDIO_CONVERT_SAMPLE_RATE: int = 16_000          # 16 kHz
    AUDIO_CONVERT_CHANNELS: int = 1                  # mono
    AUDIO_CONVERT_FORMAT: str = "wav"
    # Output temp dir for converted / chunked files (empty = system tmp)
    AUDIO_TEMP_DIR: str = ""

    # Minimum transcript confidence to accept (avg_logprob threshold)
    # Groq Whisper avg_logprob: 0.0 = perfect, negative = worse; -1.0 is very bad
    AUDIO_MIN_AVG_LOGPROB: float = -1.0
    # Fraction of words with no_speech_prob > 0.8 that triggers AUDIO_NO_SPEECH
    AUDIO_MAX_NO_SPEECH_FRACTION: float = 0.7

    # ------------------------------------------------------------------
    # Input stage — text / scenes path
    # ------------------------------------------------------------------
    # Max characters per TTS call (Groq TTS limit ~5000 chars)
    TTS_MAX_CHARS_PER_CALL: int = 4500
    # Scene separator used when joining scenes for TTS
    SCENE_JOIN_SEPARATOR: str = " "

    # ------------------------------------------------------------------
    # Scene planning
    # ------------------------------------------------------------------
    SCENE_MIN_DURATION_SECONDS: float = 3.0     # shortest allowed scene
    SCENE_MAX_DURATION_SECONDS: float = 8.0     # longest allowed scene
    # When LLM plan is invalid after all repairs, fall back to rule-based splitter
    SCENE_PLAN_FALLBACK_ENABLED: bool = True
    # Hard ambiguous-word list for WSD (word sense disambiguation)
    SCENE_WSD_WORDS: list[str] = ["value", "bank", "cloud", "mouse", "python",
                                   "bat", "match", "fair", "kind", "left",
                                   "right", "light", "spring", "bark", "board"]
    # Max scenes the LLM is asked to produce per call
    SCENE_PLAN_MAX_SCENES: int = 50

    # ------------------------------------------------------------------
    # Clip selection — text scoring (stage-1, cheap)
    # ------------------------------------------------------------------
    # Number of top candidates passed to VLM after text scoring
    CLIP_SELECTION_VLM_TOP_N: int = 5
    # Minimum number of candidates to gather per scene before scoring
    CLIP_SELECTION_MIN_CANDIDATES: int = 3
    # Max candidates to fetch from FootageRouter per query variant
    CLIP_SELECTION_CANDIDATES_PER_QUERY: int = 5

    # ------------------------------------------------------------------
    # Clip selection — VLM scoring (stage-2, expensive)
    # ------------------------------------------------------------------
    # Enable VLM scoring; when False only text scores are used
    CLIP_SELECTION_VLM_ENABLED: bool = True
    # Weights for final_score = w_text * text_score + w_vlm * vlm_score
    # Both are normalised to [0, 5] before blending
    CLIP_SELECTION_WEIGHT_TEXT: float = 0.4
    CLIP_SELECTION_WEIGHT_VLM: float = 0.6

    # ------------------------------------------------------------------
    # Clip selection — source-selection thresholds (measurable rules)
    # ------------------------------------------------------------------
    # final_score > this → accept a real Pixabay VIDEO (tier 1)
    CLIP_THRESHOLD_PIXABAY_VIDEO: float = 4.0
    # final_score >= this → accept a Ken-Burns animated IMAGE / motion picture (tier 2)
    CLIP_THRESHOLD_KEN_BURNS: float = 3.0
    # Scores below KEN_BURNS threshold → generated title card (tier 4)

    # ------------------------------------------------------------------
    # Clip selection — duration / loop tolerance
    # ------------------------------------------------------------------
    # Clip is considered "too short" when clip.duration < scene.duration * factor
    CLIP_MIN_DURATION_FACTOR: float = 0.25
    # How much the clip can be slowed down to fill the scene (1.0 = no slow, 0.5 = 2× slow)
    CLIP_SLOW_MOTION_MIN_FACTOR: float = 0.5
    # Max times a clip may be looped to fill the scene
    CLIP_MAX_LOOP_COUNT: int = 3

    # ------------------------------------------------------------------
    # Clip selection — deduplication
    # ------------------------------------------------------------------
    # Perceptual hash distance threshold: hashes with distance <= this are dupes
    CLIP_PHASH_MAX_DISTANCE: int = 10
    # Whether to compute perceptual hash for deduplication (requires imagehash)
    CLIP_PHASH_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Rendering — target resolutions per aspect ratio
    # ------------------------------------------------------------------
    RENDER_FPS: int = 30
    RENDER_PIXEL_FORMAT: str = "yuv420p"
    RENDER_SAR: str = "1"
    # Target resolutions (width x height)
    RENDER_WIDTH_16_9: int = 1920
    RENDER_HEIGHT_16_9: int = 1080
    RENDER_WIDTH_9_16: int = 1080
    RENDER_HEIGHT_9_16: int = 1920
    RENDER_WIDTH_1_1: int = 1080
    RENDER_HEIGHT_1_1: int = 1080

    # ------------------------------------------------------------------
    # Rendering — brightness/colour matching
    # ------------------------------------------------------------------
    RENDER_BRIGHTNESS_MATCH: bool = True
    # Target luma (0.0–1.0); 0.5 = mid-grey
    RENDER_TARGET_LUMA: float = Field(default=0.5, ge=0.0, le=1.0)
    # Tolerance: clips within this luma delta are not adjusted
    RENDER_LUMA_TOLERANCE: float = Field(default=0.05, ge=0.0, le=0.5)

    # ------------------------------------------------------------------
    # Rendering — crossfade
    # ------------------------------------------------------------------
    RENDER_CROSSFADE_DURATION: float = Field(default=0.3, ge=0.0, le=2.0)

    # ------------------------------------------------------------------
    # Rendering — smart reframing
    # ------------------------------------------------------------------
    # "vlm_bbox" | "saliency" | "center" | "blurred_bg"
    RENDER_REFRAME_METHOD: str = "saliency"
    RENDER_BLUR_BG_STRENGTH: int = 20      # gaussian blur radius for blurred-bg fit
    RENDER_SALIENCY_FALLBACK: str = "center"  # if saliency fails: center | blurred_bg

    # ------------------------------------------------------------------
    # Rendering — Ken Burns
    # ------------------------------------------------------------------
    RENDER_KB_ZOOM_START: float = 1.0
    RENDER_KB_ZOOM_END: float = 1.15
    RENDER_KB_DURATION_OVERRIDE: float = 0.0  # 0 = use scene duration

    # ------------------------------------------------------------------
    # Rendering — captions / subtitles
    # ------------------------------------------------------------------
    RENDER_CAPTION_MAX_LINES: int = 2
    RENDER_CAPTION_MAX_CHARS_PER_LINE: int = 42
    RENDER_CAPTION_FONT_SIZE: int = 24
    RENDER_CAPTION_MARGIN_V: int = 40      # vertical margin from bottom
    RENDER_CAPTION_BOX_OPACITY: float = Field(default=0.5, ge=0.0, le=1.0)
    # Font names (must be installed in the Docker image)
    RENDER_CAPTION_FONT_LATIN: str = "NotoSans"
    RENDER_CAPTION_FONT_DEVA: str = "NotoSansDevanagari"
    RENDER_CAPTION_FONT_TAMIL: str = "NotoSansTamil"

    # ------------------------------------------------------------------
    # Rendering — post-render checks
    # ------------------------------------------------------------------
    RENDER_POST_CHECK_ENABLED: bool = True
    # blackdetect: fraction of pixels below threshold = black frame
    RENDER_BLACK_DETECT_THRESHOLD: float = Field(default=0.98, ge=0.0, le=1.0)
    RENDER_BLACK_DETECT_MIN_DURATION: float = Field(default=2.0, ge=0.0)
    # freezedetect: motion threshold (lower = more sensitive)
    RENDER_FREEZE_DETECT_THRESHOLD: float = Field(default=0.01, ge=0.0)
    RENDER_FREEZE_DETECT_MIN_DURATION: float = Field(default=2.0, ge=0.0)
    # audio/video sync drift tolerance in milliseconds
    RENDER_MAX_DRIFT_MS: int = 50

    # ------------------------------------------------------------------
    # Rendering — scene segment cache
    # ------------------------------------------------------------------
    RENDER_SCENE_CACHE_ENABLED: bool = True
    RENDER_SCENE_CACHE_DIR: str = ""      # empty = system tmp
    RENDER_SCENE_CACHE_TTL_SECONDS: int = 86400  # 24 h

    # ------------------------------------------------------------------
    # Pacing guard
    # ------------------------------------------------------------------
    # Min fraction of clips that must differ from their scene type
    # (video vs image) to avoid monotonous pacing
    PACING_MIN_VARIETY_RATIO: float = 0.0
    # Max consecutive scenes with the same clip source
    PACING_MAX_CONSECUTIVE_SAME_SOURCE: int = 3
    # Target scene duration spread coefficient of variation (0=all same, 1=very varied)
    PACING_TARGET_CV: float = 0.2

    # ------------------------------------------------------------------
    # Global colour pass
    # ------------------------------------------------------------------
    # Apply a global colour-normalisation pass across all segments before concat
    COLOUR_PASS_ENABLED: bool = True
    # Target average luma across the full video (0–1)
    COLOUR_PASS_TARGET_LUMA: float = Field(default=0.5, ge=0.0, le=1.0)
    # Max per-clip luma correction magnitude
    COLOUR_PASS_MAX_CORRECTION: float = Field(default=0.3, ge=0.0, le=1.0)
    # Also apply a gentle saturation normalisation
    COLOUR_PASS_TARGET_SATURATION: float = Field(default=1.0, ge=0.5, le=2.0)

    # ------------------------------------------------------------------
    # VLM eval (independent model for evaluation)
    # ------------------------------------------------------------------
    VLM_EVAL_MODEL: str = "llava-v1.5-13b-4096-preview"   # different from selection model
    VLM_EVAL_FRAMES_PER_CLIP: int = 3                      # frames to sample per clip

    # ------------------------------------------------------------------
    # i18n / multi-language
    # ------------------------------------------------------------------
    # Languages that require Devanagari captions (ISO 639-1 codes)
    LANG_DEVANAGARI: list[str] = ["hi", "mr", "ne", "sa", "mai", "kok"]
    # Languages that require Tamil captions
    LANG_TAMIL: list[str] = ["ta"]
    # STT language auto-detection threshold (confidence below which we re-try with hint)
    STT_LANG_CONFIDENCE_THRESHOLD: float = 0.6

    # ------------------------------------------------------------------
    # CORS
    # ------------------------------------------------------------------
    CORS_ALLOW_ORIGINS: list[str] = ["*"]
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
