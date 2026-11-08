from __future__ import annotations

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app import __version__
from app.audio import MEDIA_TYPES, encode_audio, waveform_to_pcm16_bytes, waveform_to_wav_bytes
from app.cache import TtsCache
from app.config import settings
from app.engine import OPENAI_VOICE_MAP, MossTtsEngine
from app.normalizer import TextNormalizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("moss_tts_service")


class SpeechRequest(BaseModel):
    model: str = Field(default="moss-tts-nano", description="TTS model ID (e.g. moss-tts-nano, tts-1, tts-1-hd)")
    input: str = Field(..., min_length=1, max_length=10000, description="The text to generate audio for.")
    voice: str = Field(
        default=settings.default_voice,
        description="Voice ID (Chinese: Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu; OpenAI aliases: alloy, echo, fable, onyx, nova, shimmer, coral; or custom voice name).",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="wav",
        description="The audio output format.",
    )
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="The speed of the generated audio (0.25 to 4.0).")
    stream: bool = Field(default=False, description="Stream audio chunks as each sentence is synthesized.")
    sample_mode: Literal["fixed", "greedy", "full"] = Field(
        default=settings.sample_mode,
        description="Sampling mode: fixed (fast ONNX graph sampling), greedy (deterministic), full (host sampling).",
    )
    max_new_frames: int | None = Field(default=None, ge=16, le=1500, description="Max generated audio frames per chunk.")
    seed: int | None = Field(default=None, description="Optional random seed for reproducible generation.")


class AppState:
    cache: TtsCache
    normalizer: TextNormalizer
    engine: MossTtsEngine | None = None


state = AppState()


def get_engine() -> MossTtsEngine:
    if state.engine is None:
        raise HTTPException(status_code=503, detail="TTS engine is still initializing.")
    return state.engine


async def verify_api_key(authorization: str | None = Header(default=None)) -> None:
    expected_key = settings.api_key.strip()
    if not expected_key:
        raise HTTPException(status_code=500, detail="Server API key (MOSS_TTS_API_KEY) is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer API key in Authorization header.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.cache = TtsCache(
        enabled=settings.cache_enabled,
        cache_dir=settings.cache_dir,
        max_memory_items=settings.cache_max_memory_items,
        max_chunk_items=settings.cache_max_chunk_items,
        max_disk_mb=settings.cache_max_disk_mb,
        ttl_seconds=settings.cache_ttl_seconds,
    )
    state.normalizer = TextNormalizer(
        enable_wetext=settings.enable_wetext,
        enable_robust=settings.enable_robust_norm,
    )
    state.engine = await asyncio.to_thread(
        MossTtsEngine,
        model_dir=settings.model_dir,
        voices_dir=settings.voices_dir,
        cache=state.cache,
        normalizer=state.normalizer,
        cpu_threads=settings.cpu_threads,
        default_voice=settings.default_voice,
        max_new_frames=settings.max_new_frames,
        voice_clone_max_text_tokens=settings.voice_clone_max_text_tokens,
        default_sample_mode=settings.sample_mode,
    )
    if settings.warmup_on_startup:
        await asyncio.to_thread(state.engine.warmup)
    yield


app = FastAPI(
    title="MOSS-TTS-Nano OpenAI Speech Service",
    version=__version__,
    description="OpenAI-compatible Chinese & multilingual CPU TTS service powered by MOSS-TTS-Nano ONNX Runtime.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok" if state.engine is not None else "initializing",
        "version": __version__,
        "default_voice": settings.default_voice,
        "cpu_threads": settings.cpu_threads,
        "cache": state.cache.stats() if hasattr(state, "cache") else {},
    }


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> dict[str, object]:
    now = int(time.time())
    model_ids = ["moss-tts-nano", "tts-1", "tts-1-hd", "gpt-4o-mini-tts"]
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": now,
                "owned_by": "openmoss",
            }
            for mid in model_ids
        ],
    }


@app.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
async def get_model(model_id: str) -> dict[str, object]:
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "openmoss",
    }


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
@app.get("/v1/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(engine: MossTtsEngine = Depends(get_engine)) -> dict[str, object]:
    return {
        "default_voice": settings.default_voice,
        "openai_voice_map": OPENAI_VOICE_MAP,
        "voices": engine.list_voices(),
    }


@app.post("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
async def upload_custom_voice(
    voice_id: str = Form(..., description="Unique voice identifier (alphanumeric, _, -)"),
    file: UploadFile = File(..., description="Reference audio file (.wav, .flac, .mp3)"),
    engine: MossTtsEngine = Depends(get_engine),
) -> dict[str, object]:
    clean_id = re.sub(r"[^A-Za-z0-9_-]", "", voice_id.strip())
    if not clean_id:
        raise HTTPException(status_code=400, detail="Invalid voice_id.")
    suffix = "." + (file.filename or "voice.wav").rsplit(".", 1)[-1].lower()
    if suffix not in {".wav", ".flac", ".mp3", ".ogg"}:
        suffix = ".wav"
    target_path = engine.voices_dir / f"{clean_id}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    target_path.write_bytes(content)

    try:
        codes, audio_hash = await asyncio.to_thread(engine.encode_reference_audio_cached, target_path)
    except Exception as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to encode reference audio: {exc}") from exc

    return {
        "voice_id": clean_id,
        "frames": len(codes),
        "audio_hash": audio_hash,
        "message": f"Custom voice '{clean_id}' registered and cached.",
    }


@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
async def create_speech(
    req: SpeechRequest,
    engine: MossTtsEngine = Depends(get_engine),
):
    try:
        voice_key, _ = engine.resolve_voice(req.voice)
        normalized_text = await asyncio.to_thread(engine.normalizer.normalize, req.input, voice_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not normalized_text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if req.stream:
        return _stream_speech_response(req, engine, voice_key)

    cache_key = TtsCache.make_key(
        text=normalized_text,
        voice=voice_key,
        fmt=req.response_format,
        speed=round(float(req.speed), 3),
        sample_mode=req.sample_mode,
        max_new_frames=req.max_new_frames or settings.max_new_frames,
        seed=req.seed,
    )
    cached_bytes = state.cache.get_audio(cache_key, req.response_format)
    if cached_bytes is not None:
        return Response(
            content=cached_bytes,
            media_type=MEDIA_TYPES.get(req.response_format, "audio/wav"),
            headers={"X-Cache": "HIT", "X-Voice-Resolved": voice_key},
        )

    def _synthesize_and_encode() -> tuple[bytes, str, str]:
        waveform, resolved_voice, _norm = engine.synthesize_waveform(
            text=req.input,
            voice=req.voice,
            sample_mode=req.sample_mode,
            max_new_frames=req.max_new_frames,
            seed=req.seed,
        )
        audio_bytes, media_type, actual_ext = encode_audio(
            waveform,
            sample_rate=engine.sample_rate,
            response_format=req.response_format,
            speed=req.speed,
        )
        state.cache.set_audio(cache_key, audio_bytes, actual_ext)
        return audio_bytes, media_type, resolved_voice

    try:
        audio_bytes, media_type, resolved_voice = await asyncio.to_thread(_synthesize_and_encode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Speech synthesis failed")
        raise HTTPException(status_code=500, detail=f"Speech synthesis error: {exc}") from exc

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"X-Cache": "MISS", "X-Voice-Resolved": resolved_voice},
    )


def _stream_speech_response(req: SpeechRequest, engine: MossTtsEngine, voice_key: str) -> StreamingResponse:
    fmt = req.response_format if req.response_format in {"pcm", "wav"} else "wav"
    media_type = MEDIA_TYPES[fmt]

    def chunk_generator():
        for chunk_wave in engine.synthesize_chunks_stream(
            text=req.input,
            voice=req.voice,
            sample_mode=req.sample_mode,
            max_new_frames=req.max_new_frames,
            seed=req.seed,
        ):
            if fmt == "pcm":
                pcm_bytes, _ = waveform_to_pcm16_bytes(chunk_wave)
                yield pcm_bytes
            else:
                yield waveform_to_wav_bytes(chunk_wave, engine.sample_rate)

    return StreamingResponse(
        chunk_generator(),
        media_type=media_type,
        headers={"X-Voice-Resolved": voice_key, "X-Stream-Format": fmt},
    )


@app.get("/v1/cache/stats", dependencies=[Depends(verify_api_key)])
async def cache_stats() -> dict[str, object]:
    return state.cache.stats()


@app.delete("/v1/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache() -> dict[str, int]:
    return state.cache.clear()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
