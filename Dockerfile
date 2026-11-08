FROM ghcr.io/astral-sh/uv:latest AS uv

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    MOSS_TTS_HOST=0.0.0.0 \
    MOSS_TTS_PORT=8000 \
    MOSS_TTS_MODEL_DIR=/app/models \
    MOSS_TTS_VOICES_DIR=/app/voices \
    MOSS_TTS_CACHE_DIR=/app/.cache/tts

WORKDIR /app

# Install lightweight audio libraries (ffmpeg for mp3/opus/aac/atempo, libsndfile1 for soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

COPY pyproject.toml README.md LICENSE ./
COPY app ./app

RUN uv pip install --no-cache .

# Optional: preload MOSS-TTS-Nano ONNX weights into the image at build time
ARG PRELOAD_MODELS=0
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN mkdir -p /app/models /app/voices /app/.cache/tts && \
    if [ "$PRELOAD_MODELS" = "1" ]; then \
        python -c "from pathlib import Path; from app.engine import ensure_onnx_models; ensure_onnx_models(Path('/app/models'))"; \
    fi

VOLUME ["/app/models", "/app/voices", "/app/.cache/tts"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
