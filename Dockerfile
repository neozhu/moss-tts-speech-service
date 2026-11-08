FROM ghcr.io/astral-sh/uv:latest AS uv

# Download ONNX models natively on $BUILDPLATFORM (avoids slow QEMU emulation on multi-arch builds)
FROM --platform=$BUILDPLATFORM python:3.13-slim AS models
ARG PRELOAD_MODELS=1
ARG HF_ENDPOINT=https://huggingface.co
ENV HF_ENDPOINT=${HF_ENDPOINT}
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/moss-models && \
    if [ "$PRELOAD_MODELS" = "1" ]; then \
        python3 -c "import os, json, fnmatch, subprocess, pathlib; \
base = os.getenv('HF_ENDPOINT', 'https://huggingface.co').rstrip('/'); \
repos = [('OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX', 'MOSS-TTS-Nano-100M-ONNX', ['*.onnx', '*.data', '*.json', 'tokenizer.model']), \
         ('OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX', 'MOSS-Audio-Tokenizer-Nano-ONNX', ['*.onnx', '*.data', '*.json'])]; \
[ [ (dest := pathlib.Path('/opt/moss-models') / sub / f['rfilename'], dest.parent.mkdir(parents=True, exist_ok=True), \
     subprocess.run(['curl', '-fsSL', '--retry', '3', '-o', str(dest), f'{base}/{repo}/resolve/main/{f[\"rfilename\"]}'], check=True)) \
    for f in json.loads(subprocess.check_output(['curl', '-fsSL', f'{base}/api/models/{repo}'])).get('siblings', []) \
    if any(fnmatch.fnmatch(f['rfilename'], p) for p in pats) ] for repo, sub, pats in repos ]"; \
    fi

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
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

COPY --from=models /opt/moss-models /opt/moss-models
RUN mkdir -p /app/models /app/voices /app/.cache/tts

VOLUME ["/app/voices", "/app/.cache/tts"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
