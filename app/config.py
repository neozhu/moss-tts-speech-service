from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MOSS_TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = "sk-moss-tts"

    # Model & Voice paths
    model_dir: Path = Path("./models")
    voices_dir: Path = Path("./voices")
    default_voice: str = "Junhao"

    # CPU & Inference parameters
    cpu_threads: int = min(4, max(1, os.cpu_count() or 4))
    max_new_frames: int = 375
    voice_clone_max_text_tokens: int = 75
    sample_mode: str = "fixed"  # "fixed" | "greedy" | "full"
    enable_wetext: bool = True
    enable_robust_norm: bool = True
    warmup_on_startup: bool = True

    # Caching settings
    cache_enabled: bool = True
    cache_dir: Path = Path("./.cache/tts")
    cache_max_memory_items: int = 256
    cache_max_chunk_items: int = 512
    cache_max_disk_mb: int = 512
    cache_ttl_seconds: int = 86400 * 7


settings = Settings()
