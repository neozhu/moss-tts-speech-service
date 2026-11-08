from __future__ import annotations

import io
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def load_reference_audio(
    audio_path: str | Path,
    target_sample_rate: int = 48000,
    target_channels: int = 2,
) -> np.ndarray:
    """Load and resample reference audio to shape (1, target_channels, samples) float32 without PyTorch."""
    data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    # data has shape (samples, channels)
    if sample_rate != target_sample_rate:
        gcd = math.gcd(int(sample_rate), int(target_sample_rate))
        up = int(target_sample_rate) // gcd
        down = int(sample_rate) // gcd
        data = resample_poly(data, up, down, axis=0).astype(np.float32, copy=False)

    current_channels = data.shape[1]
    if current_channels == target_channels:
        channel_first = data.T
    elif current_channels == 1 and target_channels > 1:
        channel_first = np.repeat(data.T, target_channels, axis=0)
    elif current_channels > 1 and target_channels == 1:
        channel_first = np.mean(data.T, axis=0, keepdims=True)
    else:
        channel_first = data[:, :target_channels].T

    return np.expand_dims(np.ascontiguousarray(channel_first, dtype=np.float32), axis=0)


def waveform_to_pcm16_bytes(waveform: np.ndarray) -> tuple[bytes, int]:
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.round(clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes(), int(pcm16.shape[1])


def waveform_to_wav_bytes(waveform: np.ndarray, sample_rate: int = 48000) -> bytes:
    pcm_bytes, channels = waveform_to_pcm16_bytes(waveform)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _build_atempo_filter(speed: float) -> str:
    """Build chained ffmpeg atempo filters for 0.25x - 4.0x speed."""
    remaining = max(0.25, min(4.0, float(speed)))
    filters: list[str] = []
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _apply_speed_fallback(waveform: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3 or waveform.size == 0:
        return waveform
    up = 100
    down = max(1, int(round(100 * float(speed))))
    gcd = math.gcd(up, down)
    return resample_poly(waveform, up // gcd, down // gcd, axis=0).astype(np.float32, copy=False)


def encode_audio(
    waveform: np.ndarray,
    sample_rate: int = 48000,
    response_format: str = "wav",
    speed: float = 1.0,
) -> tuple[bytes, str, str]:
    """Encode float32 waveform (samples, channels) to requested format and return (bytes, media_type, ext)."""
    fmt = (response_format or "wav").lower().strip()
    if fmt not in MEDIA_TYPES:
        fmt = "wav"
    speed = max(0.25, min(4.0, float(speed or 1.0)))
    has_ffmpeg = shutil.which("ffmpeg") is not None

    # Fast native path when speed == 1.0 (or no ffmpeg) and format is wav / pcm / flac
    if (abs(speed - 1.0) < 1e-3 or not has_ffmpeg) and fmt in {"wav", "pcm", "flac"}:
        adj_waveform = _apply_speed_fallback(waveform, speed) if abs(speed - 1.0) >= 1e-3 else waveform
        if fmt == "pcm":
            pcm_bytes, _ = waveform_to_pcm16_bytes(adj_waveform)
            return pcm_bytes, MEDIA_TYPES["pcm"], "pcm"
        if fmt == "flac":
            buf = io.BytesIO()
            sf.write(buf, adj_waveform, sample_rate, format="FLAC", subtype="PCM_16")
            return buf.getvalue(), MEDIA_TYPES["flac"], "flac"
        return waveform_to_wav_bytes(adj_waveform, sample_rate), MEDIA_TYPES["wav"], "wav"

    if not has_ffmpeg:
        adj_waveform = _apply_speed_fallback(waveform, speed)
        return waveform_to_wav_bytes(adj_waveform, sample_rate), MEDIA_TYPES["wav"], "wav"

    wav_input = waveform_to_wav_bytes(waveform, sample_rate)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
    if abs(speed - 1.0) >= 1e-3:
        cmd.extend(["-filter:a", _build_atempo_filter(speed)])

    codec_args: dict[str, list[str]] = {
        "mp3": ["-f", "mp3", "-c:a", "libmp3lame", "-q:a", "2"],
        "opus": ["-f", "opus", "-c:a", "libopus", "-b:a", "96k"],
        "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "128k"],
        "flac": ["-f", "flac", "-c:a", "flac"],
        "wav": ["-f", "wav", "-c:a", "pcm_s16le"],
        "pcm": ["-f", "s16le", "-c:a", "pcm_s16le"],
    }
    cmd.extend(codec_args[fmt])
    cmd.append("pipe:1")

    proc = subprocess.run(cmd, input=wav_input, capture_output=True, check=True)
    return proc.stdout, MEDIA_TYPES[fmt], fmt
