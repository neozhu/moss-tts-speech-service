from __future__ import annotations

from pathlib import Path
import numpy as np
from fastapi.testclient import TestClient

from app.audio import encode_audio, waveform_to_wav_bytes
from app.cache import TtsCache
from app.config import settings
from app.main import app, state
from app.normalizer import TextNormalizer, _rewrite_hyphens_before_zh_wetext, normalize_tts_text


def test_robust_chinese_normalizer() -> None:
    assert normalize_tts_text("今天update到v2.3.1了") == "今天 update 到 v2.3.1 了。"
    assert normalize_tts_text("【公告】今天 20:00 维护——预计 30 分钟。") == '"公告"今天20:00维护。预计30分钟。'
    assert _rewrite_hyphens_before_zh_wetext("请求接入-身份判定") == "请求接入，身份判定"
    assert _rewrite_hyphens_before_zh_wetext("2026-05-01") == "2026-05-01"

    tn = TextNormalizer(enable_wetext=True, enable_robust=True)
    normalized = tn.normalize("今天的价格是12.5元，预计2026-05-01正式上线！", "Junhao")
    assert normalized == "今天的价格是十二点五元，预计二零二六年五月一日正式上线！"


def test_tts_cache_l1_and_l2(tmp_path: Path) -> None:
    cache = TtsCache(enabled=True, cache_dir=tmp_path / "cache", max_memory_items=2)
    key = TtsCache.make_key(text="你好。", voice="Junhao", fmt="wav", speed=1.0)
    assert cache.get_audio(key, "wav") is None
    cache.set_audio(key, b"RIFF_TEST_WAV", "wav")
    assert cache.get_audio(key, "wav") == b"RIFF_TEST_WAV"
    assert cache.hits_memory == 1

    # Clear memory only and verify L2 disk hit
    cache._memory_audio.clear()
    assert cache.get_audio(key, "wav") == b"RIFF_TEST_WAV"
    assert cache.hits_disk == 1


def test_encode_audio_wav_and_pcm() -> None:
    waveform = np.zeros((4800, 2), dtype=np.float32)
    wav_bytes, media_type, ext = encode_audio(waveform, sample_rate=48000, response_format="wav", speed=1.0)
    assert ext == "wav"
    assert media_type == "audio/wav"
    assert wav_bytes[:4] == b"RIFF"

    pcm_bytes, pcm_type, pcm_ext = encode_audio(waveform, sample_rate=48000, response_format="pcm", speed=1.0)
    assert pcm_ext == "pcm"
    assert pcm_type == "audio/pcm"
    assert len(pcm_bytes) == 4800 * 2 * 2


class _DummyEngine:
    sample_rate = 48000
    channels = 2
    voices_dir = Path("./voices")

    def __init__(self) -> None:
        self.normalizer = TextNormalizer(enable_wetext=False, enable_robust=True)
        self.calls = 0

    def list_voices(self):
        return [{"id": "Junhao", "name": "CN 欢迎关注模思智能", "group": "Chinese Male", "language": "zh", "type": "builtin", "openai_aliases": ["alloy", "onyx"]}]

    def resolve_voice(self, voice: str | None):
        if voice in {None, "Junhao", "alloy", "nova", "Xiaoyu"}:
            return ("Xiaoyu" if voice in {"nova", "Xiaoyu"} else "Junhao"), [[0] * 16]
        raise ValueError(f"Unknown voice: {voice}")

    def synthesize_waveform(self, *, text: str, voice: str | None = None, sample_mode: str | None = None, max_new_frames: int | None = None, seed: int | None = None):
        self.calls += 1
        v_key, _ = self.resolve_voice(voice)
        norm = self.normalizer.normalize(text, v_key)
        return np.zeros((2400, 2), dtype=np.float32), v_key, norm

    def synthesize_chunks_stream(self, *, text: str, voice: str | None = None, sample_mode: str | None = None, max_new_frames: int | None = None, seed: int | None = None):
        yield np.zeros((1200, 2), dtype=np.float32)
        yield np.zeros((1200, 2), dtype=np.float32)


def test_openai_speech_endpoints_and_caching(tmp_path: Path) -> None:
    state.cache = TtsCache(enabled=True, cache_dir=tmp_path / "api_cache")
    state.normalizer = TextNormalizer(enable_wetext=False, enable_robust=True)
    dummy = _DummyEngine()
    state.engine = dummy  # type: ignore[assignment]

    client = TestClient(app)
    headers = {"Authorization": f"Bearer {settings.api_key}"}

    # Unauthenticated request must return 401
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/audio/speech", json={"input": "你好"}).status_code == 401

    r_models = client.get("/v1/models", headers=headers)
    assert r_models.status_code == 200
    assert any(m["id"] == "moss-tts-nano" for m in r_models.json()["data"])

    r_voices = client.get("/v1/audio/voices", headers=headers)
    assert r_voices.status_code == 200
    assert r_voices.json()["default_voice"] == "Junhao"

    # First speech call -> MISS
    r1 = client.post(
        "/v1/audio/speech",
        headers=headers,
        json={"model": "tts-1", "input": "你好，欢迎使用语音服务", "voice": "alloy", "response_format": "wav"},
    )
    assert r1.status_code == 200
    assert r1.headers["X-Cache"] == "MISS"
    assert r1.headers["X-Voice-Resolved"] == "Junhao"
    assert r1.content[:4] == b"RIFF"
    assert dummy.calls == 1

    # Second identical speech call -> HIT (no extra engine synthesis call)
    r2 = client.post(
        "/v1/audio/speech",
        headers=headers,
        json={"model": "tts-1", "input": "你好，欢迎使用语音服务。", "voice": "Junhao", "response_format": "wav"},
    )
    assert r2.status_code == 200
    assert r2.headers["X-Cache"] == "HIT"
    assert dummy.calls == 1
