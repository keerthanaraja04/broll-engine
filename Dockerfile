# =============================================================================
# broll-engine Dockerfile
#
# Stage 1 (builder): install Python deps into a venv
# Stage 2 (runtime): copy venv + install ffmpeg (libass) + Noto fonts
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — dependency builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip==24.0

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the package itself (editable-style, so cli/ and app/ are importable)
COPY pyproject.toml .
COPY app/ app/
COPY cli/ cli/
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="broll-engine" \
      org.opencontainers.image.title="broll-engine" \
      org.opencontainers.image.description="B-roll video generation API and worker"

# ---------------------------------------------------------------------------
# System packages:
#   ffmpeg          — built with libass for subtitle rendering
#   fonts-noto*     — Noto Latin, Devanagari, Tamil
#   libpq5          — PostgreSQL runtime client lib
#   ca-certificates — HTTPS trust
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # FFmpeg with libass subtitle support
    ffmpeg \
    # Noto fonts — Latin (covers most Western scripts)
    fonts-noto \
    # Noto fonts — Devanagari (Hindi, Marathi, Sanskrit, …)
    fonts-noto-extra \
    # Noto fonts — Tamil
    fonts-noto-ui-extra \
    # Runtime library for libass (ASS/SSA subtitle rendering)
    libass9 \
    # PostgreSQL client lib
    libpq5 \
    # TLS certs
    ca-certificates \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Copy venv from builder
# ---------------------------------------------------------------------------
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
WORKDIR /app
COPY --from=builder /build/app ./app
COPY --from=builder /build/cli ./cli

# Non-root user for security
RUN groupadd --gid 1001 broll && \
    useradd --uid 1001 --gid broll --shell /bin/bash --create-home broll && \
    mkdir -p /app/media && \
    chown -R broll:broll /app

USER broll

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    FONT_DIR=/usr/share/fonts/truetype/noto \
    SUBTITLE_FONT=NotoSans \
    PORT=8000

EXPOSE 8000

# Default: run the API with dynamic PORT support for platforms like Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
