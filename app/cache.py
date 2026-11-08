from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class TtsCache:
    """Thread-safe multi-layer cache for TTS audio, sentence chunks, and voice prompt codes."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        cache_dir: Path = Path("./.cache/tts"),
        max_memory_items: int = 256,
        max_chunk_items: int = 512,
        max_disk_mb: int = 512,
        ttl_seconds: int = 86400 * 7,
    ) -> None:
        self.enabled = enabled
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.audio_dir = self.cache_dir / "audio"
        self.prompts_dir = self.cache_dir / "prompts"
        self.max_memory_items = max(1, max_memory_items)
        self.max_chunk_items = max(1, max_chunk_items)
        self.max_disk_bytes = max(1, max_disk_mb) * 1024 * 1024
        self.ttl_seconds = max(60, ttl_seconds)

        self._lock = threading.Lock()
        self._memory_audio: OrderedDict[str, tuple[bytes, str, float]] = OrderedDict()
        self._memory_chunks: OrderedDict[str, np.ndarray] = OrderedDict()
        self._memory_prompts: dict[str, list[list[int]]] = {}

        self.hits_memory = 0
        self.hits_disk = 0
        self.hits_chunk = 0
        self.misses = 0

        if self.enabled:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            self.prompts_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(**kwargs: Any) -> str:
        payload = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_audio(self, key: str, ext: str) -> bytes | None:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            item = self._memory_audio.get(key)
            if item is not None:
                data, _stored_ext, ts = item
                if now - ts <= self.ttl_seconds:
                    self._memory_audio.move_to_end(key)
                    self.hits_memory += 1
                    return data
                del self._memory_audio[key]

        disk_path = self.audio_dir / f"{key}.{ext}"
        if disk_path.is_file():
            try:
                stat = disk_path.stat()
                if now - stat.st_mtime <= self.ttl_seconds:
                    data = disk_path.read_bytes()
                    with self._lock:
                        self.hits_disk += 1
                        self._put_memory_audio_unlocked(key, data, ext, now)
                    return data
                disk_path.unlink(missing_ok=True)
            except OSError:
                pass

        with self._lock:
            self.misses += 1
        return None

    def set_audio(self, key: str, data: bytes, ext: str) -> None:
        if not self.enabled or not data:
            return
        now = time.time()
        with self._lock:
            self._put_memory_audio_unlocked(key, data, ext, now)

        try:
            disk_path = self.audio_dir / f"{key}.{ext}"
            disk_path.write_bytes(data)
            self._evict_disk_if_needed()
        except OSError:
            pass

    def _put_memory_audio_unlocked(self, key: str, data: bytes, ext: str, now: float) -> None:
        self._memory_audio[key] = (data, ext, now)
        self._memory_audio.move_to_end(key)
        while len(self._memory_audio) > self.max_memory_items:
            self._memory_audio.popitem(last=False)

    def _evict_disk_if_needed(self) -> None:
        files = []
        total_bytes = 0
        now = time.time()
        for path in self.audio_dir.glob("*"):
            if not path.is_file():
                continue
            try:
                st = path.stat()
                if now - st.st_mtime > self.ttl_seconds:
                    path.unlink(missing_ok=True)
                    continue
                files.append((st.st_mtime, st.st_size, path))
                total_bytes += st.st_size
            except OSError:
                continue

        if total_bytes <= self.max_disk_bytes:
            return
        files.sort(key=lambda x: x[0])
        for _mtime, size, path in files:
            if total_bytes <= self.max_disk_bytes:
                break
            try:
                path.unlink(missing_ok=True)
                total_bytes -= size
            except OSError:
                pass

    def get_chunk(self, chunk_key: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        with self._lock:
            waveform = self._memory_chunks.get(chunk_key)
            if waveform is not None:
                self._memory_chunks.move_to_end(chunk_key)
                self.hits_chunk += 1
                return waveform
        return None

    def set_chunk(self, chunk_key: str, waveform: np.ndarray) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._memory_chunks[chunk_key] = waveform
            self._memory_chunks.move_to_end(chunk_key)
            while len(self._memory_chunks) > self.max_chunk_items:
                self._memory_chunks.popitem(last=False)

    def get_prompt_codes(self, audio_hash: str) -> list[list[int]] | None:
        with self._lock:
            if audio_hash in self._memory_prompts:
                return self._memory_prompts[audio_hash]
        if self.enabled:
            prompt_file = self.prompts_dir / f"{audio_hash}.json"
            if prompt_file.is_file():
                try:
                    codes = json.loads(prompt_file.read_text(encoding="utf-8"))
                    with self._lock:
                        self._memory_prompts[audio_hash] = codes
                    return codes
                except Exception:
                    pass
        return None

    def set_prompt_codes(self, audio_hash: str, codes: list[list[int]]) -> None:
        with self._lock:
            self._memory_prompts[audio_hash] = codes
        if self.enabled:
            try:
                prompt_file = self.prompts_dir / f"{audio_hash}.json"
                prompt_file.write_text(json.dumps(codes), encoding="utf-8")
            except OSError:
                pass

    def stats(self) -> dict[str, Any]:
        disk_files = list(self.audio_dir.glob("*")) if self.enabled and self.audio_dir.exists() else []
        disk_bytes = sum(f.stat().st_size for f in disk_files if f.is_file())
        with self._lock:
            return {
                "enabled": self.enabled,
                "memory_audio_items": len(self._memory_audio),
                "memory_chunk_items": len(self._memory_chunks),
                "cached_voice_prompts": len(self._memory_prompts),
                "disk_audio_files": len(disk_files),
                "disk_audio_bytes": disk_bytes,
                "hits_memory": self.hits_memory,
                "hits_disk": self.hits_disk,
                "hits_chunk": self.hits_chunk,
                "misses": self.misses,
            }

    def clear(self) -> dict[str, int]:
        removed_disk = 0
        with self._lock:
            mem_count = len(self._memory_audio) + len(self._memory_chunks)
            self._memory_audio.clear()
            self._memory_chunks.clear()
        if self.audio_dir.exists():
            for f in self.audio_dir.glob("*"):
                try:
                    f.unlink(missing_ok=True)
                    removed_disk += 1
                except OSError:
                    pass
        return {"cleared_memory_items": mem_count, "cleared_disk_files": removed_disk}
