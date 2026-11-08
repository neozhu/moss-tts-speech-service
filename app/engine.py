from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import onnxruntime as ort
import sentencepiece as spm
from huggingface_hub import snapshot_download

from app.audio import load_reference_audio
from app.cache import TtsCache
from app.normalizer import TextNormalizer

logger = logging.getLogger(__name__)

TTS_REPO_ID = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX"
CODEC_REPO_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX"

SENTENCE_END_PUNCT = set(".!?。！？；;")
CLAUSE_SPLIT_PUNCT = set(",，、；;：:")
CLOSING_PUNCT = set("\"'”’)]}）】》」』")

# Map standard OpenAI TTS voice names to MOSS-TTS-Nano built-in Chinese voices
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "Junhao",    # 中文男声 - 模思智能
    "echo": "Zhiming",    # 中文男声 - 京味胡同闲聊
    "fable": "Weiguo",    # 中文男声 - 说书
    "onyx": "Junhao",     # 中文男声 - 沉稳
    "nova": "Xiaoyu",     # 中文女声 - 明星
    "shimmer": "Lingyu",  # 中文女声 - 深夜电台
    "coral": "Yuewen",    # 中文女声 - 机车
    "sage": "Zhiming",    # 中文男声
    "ash": "Weiguo",      # 中文男声
}


def ensure_onnx_models(model_dir: Path) -> Path:
    """Ensure MOSS-TTS-Nano ONNX and Audio-Tokenizer ONNX models are downloaded."""
    resolved = model_dir.expanduser().resolve()
    manifest_candidates = (
        resolved / "browser_poc_manifest.json",
        resolved / "MOSS-TTS-Nano-100M-ONNX" / "browser_poc_manifest.json",
    )
    for candidate in manifest_candidates:
        if candidate.is_file():
            return candidate

    tts_dir = resolved / "MOSS-TTS-Nano-100M-ONNX"
    codec_dir = resolved / "MOSS-Audio-Tokenizer-Nano-ONNX"
    logger.info("Downloading MOSS-TTS-Nano ONNX models into %s ...", resolved)
    tts_dir.mkdir(parents=True, exist_ok=True)
    codec_dir.mkdir(parents=True, exist_ok=True)

    _download_with_mirror_fallback(TTS_REPO_ID, tts_dir, ["*.onnx", "*.data", "*.json", "tokenizer.model"])
    _download_with_mirror_fallback(CODEC_REPO_ID, codec_dir, ["*.onnx", "*.data", "*.json"])
    _promote_nested_files(tts_dir, ("browser_poc_manifest.json", "tts_browser_onnx_meta.json", "tokenizer.model"))
    _promote_nested_files(codec_dir, ("codec_browser_onnx_meta.json",))

    for candidate in manifest_candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"browser_poc_manifest.json not found under {resolved} after download.")


def _download_with_mirror_fallback(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
    last_err: Exception | None = None
    for endpoint in (None, "https://hf-mirror.com"):
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                allow_patterns=allow_patterns,
                endpoint=endpoint,
            )
            return
        except Exception as exc:
            last_err = exc
            logger.warning("snapshot_download failed (endpoint=%s): %s", endpoint or "default", exc)

    if shutil.which("curl"):
        logger.info("Falling back to system curl for downloading %s ...", repo_id)
        _curl_download_repo(repo_id, local_dir, allow_patterns)
        return

    if last_err is not None:
        raise last_err


def _curl_download_repo(repo_id: str, local_dir: Path, allow_patterns: list[str]) -> None:
    curl_bin = shutil.which("curl") or "curl"
    base_url = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    meta_out = subprocess.check_output([curl_bin, "-fsSL", f"{base_url}/api/models/{repo_id}"])
    meta = json.loads(meta_out.decode("utf-8-sig"))
    files = [
        s["rfilename"]
        for s in meta.get("siblings", [])
        if any(fnmatch.fnmatch(s["rfilename"], pat) for pat in allow_patterns)
    ]
    for filename in files:
        dest = local_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{repo_id}/resolve/main/{filename}"
        logger.info("Downloading %s/%s ...", repo_id, filename)
        subprocess.run([curl_bin, "-fL", "--retry", "3", "-o", str(dest), url], check=True)


def _promote_nested_files(target_dir: Path, required_names: Sequence[str]) -> None:
    if all((target_dir / name).exists() for name in required_names):
        return
    for candidate in target_dir.rglob(required_names[0]):
        parent = candidate.parent
        if all((parent / name).exists() for name in required_names) and parent.resolve() != target_dir.resolve():
            for child in parent.iterdir():
                dest = target_dir / child.name
                if not dest.exists():
                    shutil.move(str(child), str(dest))
            return


def _contains_cjk(text: str) -> bool:
    return any(
        "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf" or "\u3040" <= ch <= "\u30ff"
        for ch in text
    )


def _split_by_punct(text: str, punct: set[str]) -> list[str]:
    sentences: list[str] = []
    cur: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        cur.append(ch)
        if ch in punct:
            j = i + 1
            while j < len(text) and text[j] in CLOSING_PUNCT:
                cur.append(text[j])
                j += 1
            s = "".join(cur).strip()
            if s:
                sentences.append(s)
            cur.clear()
            while j < len(text) and text[j].isspace():
                j += 1
            i = j
            continue
        i += 1
    tail = "".join(cur).strip()
    if tail:
        sentences.append(tail)
    return sentences


def _softmax(values: np.ndarray) -> np.ndarray:
    max_val = float(np.max(values))
    exps = np.exp(values - max_val)
    total = float(np.sum(exps))
    return exps / total if total > 0 else np.zeros_like(exps)


def _sample_from_scores(
    values: np.ndarray,
    *,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if not do_sample:
        return int(np.argmax(values))
    scores = np.asarray(values, dtype=np.float32).copy() / max(1e-5, float(temperature))
    if 0 < top_k < scores.shape[0]:
        threshold = float(np.sort(scores)[::-1][top_k - 1])
        scores[scores < threshold] = float("-inf")
    if 0.0 < top_p < 1.0:
        order = np.argsort(scores)[::-1]
        sorted_probs = _softmax(scores[order])
        cum = np.cumsum(sorted_probs)
        remove = cum > float(top_p)
        if remove.size > 1:
            remove[1:] = remove[:-1].copy()
            remove[0] = False
        scores[order[remove]] = float("-inf")
    probs = _softmax(scores)
    u = float(rng.random())
    for idx, p in enumerate(probs):
        u -= float(p)
        if u <= 0:
            return int(idx)
    return int(np.argmax(scores))


class MossTtsEngine:
    """Lightweight ONNX Runtime CPU inference engine for MOSS-TTS-Nano."""

    def __init__(
        self,
        *,
        model_dir: Path,
        voices_dir: Path,
        cache: TtsCache,
        normalizer: TextNormalizer,
        cpu_threads: int = 4,
        default_voice: str = "Junhao",
        max_new_frames: int = 375,
        voice_clone_max_text_tokens: int = 75,
        default_sample_mode: str = "fixed",
    ) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.voices_dir = Path(voices_dir).expanduser().resolve()
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache
        self.normalizer = normalizer
        self.cpu_threads = max(1, int(cpu_threads))
        self.default_voice = default_voice
        self.max_new_frames = int(max_new_frames)
        self.voice_clone_max_text_tokens = int(voice_clone_max_text_tokens)
        self.default_sample_mode = default_sample_mode

        self._infer_lock = threading.Lock()
        self.rng = np.random.default_rng(1234)

        manifest_path = ensure_onnx_models(self.model_dir)
        self.manifest_dir = manifest_path.parent
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

        tts_meta_path = self._resolve_rel(self.manifest["model_files"]["tts_meta"])
        codec_meta_path = self._resolve_rel(self.manifest["model_files"]["codec_meta"])
        tokenizer_path = self._resolve_rel(self.manifest["model_files"].get("tokenizer_model", "tokenizer.model"))

        self.tts_meta: dict[str, Any] = json.loads(tts_meta_path.read_text(encoding="utf-8"))
        self.codec_meta: dict[str, Any] = json.loads(codec_meta_path.read_text(encoding="utf-8"))
        self.sample_rate = int(self.codec_meta["codec_config"]["sample_rate"])
        self.channels = int(self.codec_meta["codec_config"]["channels"])

        self.sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
        self.sessions = self._load_sessions(tts_meta_path.parent, codec_meta_path.parent)

        self.builtin_voices: dict[str, dict[str, Any]] = {
            v["voice"]: v for v in self.manifest.get("builtin_voices", [])
        }
        self.builtin_voices_lower: dict[str, str] = {k.lower(): k for k in self.builtin_voices}

    def _resolve_rel(self, rel_path: str) -> Path:
        p = (self.manifest_dir / rel_path).resolve()
        if p.exists():
            return p
        rewritten = (
            str(rel_path)
            .replace("MOSS-TTS-Nano-ONNX-CPU", "MOSS-TTS-Nano-100M-ONNX")
            .replace("MOSS-Audio-Tokenizer-Nano-ONNX-CPU", "MOSS-Audio-Tokenizer-Nano-ONNX")
        )
        return (self.manifest_dir / rewritten).resolve()

    def _make_session(self, onnx_path: Path) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.cpu_threads
        opts.inter_op_num_threads = 1
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def _load_sessions(self, tts_dir: Path, codec_dir: Path) -> dict[str, ort.InferenceSession]:
        t_files = self.tts_meta["files"]
        c_files = self.codec_meta["files"]
        sessions = {
            "prefill": self._make_session(tts_dir / t_files["prefill"]),
            "decode": self._make_session(tts_dir / t_files["decode_step"]),
            "local_decoder": self._make_session(tts_dir / t_files["local_decoder"]),
            "codec_encode": self._make_session(codec_dir / c_files["encode"]),
            "codec_decode": self._make_session(codec_dir / c_files["decode_full"]),
            "codec_decode_step": self._make_session(codec_dir / c_files["decode_step"]),
        }
        for key in ("local_greedy_frame", "local_fixed_sampled_frame", "local_cached_step"):
            if t_files.get(key):
                sessions[key] = self._make_session(tts_dir / t_files[key])
        logger.info("Loaded %d ONNX Runtime CPU sessions (threads=%d)", len(sessions), self.cpu_threads)
        return sessions

    @lru_cache(maxsize=4096)
    def encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(int(tid) for tid in self.sp.encode(str(text or ""), out_type=int))

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        reverse_aliases: dict[str, list[str]] = {}
        for alias, target in OPENAI_VOICE_MAP.items():
            reverse_aliases.setdefault(target, []).append(alias)

        for name, item in self.builtin_voices.items():
            group = str(item.get("group", ""))
            voices.append(
                {
                    "id": name,
                    "name": item.get("display_name", name),
                    "group": group,
                    "language": "zh" if "Chinese" in group else ("en" if "English" in group else "ja"),
                    "type": "builtin",
                    "openai_aliases": reverse_aliases.get(name, []),
                }
            )
        for wav_file in sorted(self.voices_dir.glob("*.*")):
            if wav_file.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}:
                voices.append(
                    {
                        "id": wav_file.stem,
                        "name": f"Custom ({wav_file.name})",
                        "group": "Custom Voice Clone",
                        "language": "zh",
                        "type": "custom",
                        "openai_aliases": [],
                    }
                )
        return voices

    def resolve_voice(self, requested_voice: str | None) -> tuple[str, list[list[int]]]:
        """Resolve voice name/alias/custom file to (canonical_voice_key, prompt_audio_codes)."""
        raw = (requested_voice or self.default_voice).strip()
        lower = raw.lower()

        if lower in OPENAI_VOICE_MAP:
            target = OPENAI_VOICE_MAP[lower]
            return target, list(self.builtin_voices[target]["prompt_audio_codes"])

        if raw in self.builtin_voices:
            return raw, list(self.builtin_voices[raw]["prompt_audio_codes"])

        if lower in self.builtin_voices_lower:
            target = self.builtin_voices_lower[lower]
            return target, list(self.builtin_voices[target]["prompt_audio_codes"])

        # Check custom voice file in voices_dir or direct file path
        candidates = [
            self.voices_dir / raw,
            self.voices_dir / f"{raw}.wav",
            self.voices_dir / f"{raw}.flac",
            self.voices_dir / f"{raw}.mp3",
            Path(raw),
        ]
        for candidate in candidates:
            if candidate.is_file():
                codes, audio_hash = self.encode_reference_audio_cached(candidate)
                return f"custom:{candidate.stem}:{audio_hash[:12]}", codes

        raise ValueError(
            f"Unknown voice '{raw}'. Available Chinese voices: Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu "
            f"(or OpenAI aliases: {', '.join(OPENAI_VOICE_MAP.keys())})."
        )

    def encode_reference_audio_cached(self, audio_path: Path) -> tuple[list[list[int]], str]:
        file_bytes = audio_path.read_bytes()
        audio_hash = hashlib.sha256(file_bytes).hexdigest()
        cached = self.cache.get_prompt_codes(audio_hash)
        if cached is not None:
            return cached, audio_hash

        waveform = load_reference_audio(audio_path, self.sample_rate, self.channels)
        length = int(waveform.shape[-1])
        with self._infer_lock:
            outputs = self.sessions["codec_encode"].run(
                None,
                {
                    "waveform": waveform,
                    "input_lengths": np.asarray([length], dtype=np.int32),
                },
            )
            names = [o.name for o in self.sessions["codec_encode"].get_outputs()]
            named = dict(zip(names, outputs, strict=True))

        audio_codes = np.asarray(named["audio_codes"], dtype=np.int32)
        code_len = int(np.asarray(named["audio_code_lengths"], dtype=np.int32).reshape(-1)[0])
        num_q = int(self.codec_meta["codec_config"]["num_quantizers"])
        codes = [
            [int(audio_codes[0, frame_idx, q_idx]) for q_idx in range(num_q)]
            for frame_idx in range(code_len)
        ]
        self.cache.set_prompt_codes(audio_hash, codes)
        return codes, audio_hash

    def split_text_chunks(self, text: str, max_tokens: int | None = None) -> list[str]:
        budget = max(8, int(max_tokens or self.voice_clone_max_text_tokens))
        cleaned = str(text or "").strip().replace("\r", " ").replace("\n", " ")
        while "  " in cleaned:
            cleaned = cleaned.replace("  ", " ")
        if not cleaned:
            return []
        if _contains_cjk(cleaned):
            if cleaned[-1] not in SENTENCE_END_PUNCT:
                cleaned += "。"
        else:
            if cleaned[:1].islower():
                cleaned = cleaned[:1].upper() + cleaned[1:]
            if cleaned[-1].isalnum():
                cleaned += "."
            if len(cleaned.split()) < 5:
                cleaned = f"        {cleaned}"

        sentences = _split_by_punct(cleaned, SENTENCE_END_PUNCT) or [cleaned]
        slices: list[tuple[int, str]] = []
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            tok_len = len(self.encode_text(s))
            if tok_len <= budget:
                slices.append((tok_len, s))
                continue
            clauses = _split_by_punct(s, CLAUSE_SPLIT_PUNCT) or [s]
            for clause in clauses:
                c = clause.strip()
                if not c:
                    continue
                c_len = len(self.encode_text(c))
                if c_len <= budget:
                    slices.append((c_len, c))
                else:
                    for piece in self._split_by_token_budget(c, budget):
                        slices.append((len(self.encode_text(piece)), piece))

        chunks: list[str] = []
        cur = ""
        cur_tokens = 0
        for tok_len, seg in slices:
            if not cur:
                cur, cur_tokens = seg, tok_len
            elif cur_tokens + tok_len > budget:
                chunks.append(cur.strip())
                cur, cur_tokens = seg, tok_len
            else:
                cur = (cur + seg) if (_contains_cjk(cur) or _contains_cjk(seg)) else f"{cur} {seg}"
                cur_tokens = len(self.encode_text(cur))
        if cur:
            chunks.append(cur.strip())
        return chunks if len(chunks) > 1 else [cleaned]

    def _split_by_token_budget(self, text: str, max_tokens: int) -> list[str]:
        rem = text.strip()
        pieces: list[str] = []
        boundaries = CLAUSE_SPLIT_PUNCT | SENTENCE_END_PUNCT | {" "}
        while rem:
            if len(self.encode_text(rem)) <= max_tokens:
                pieces.append(rem)
                break
            low, high, best = 1, len(rem), 1
            while low <= high:
                mid = (low + high) // 2
                cand = rem[:mid].strip()
                if not cand or len(self.encode_text(cand)) <= max_tokens:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1
            cut = best
            prefix = rem[:best]
            for idx in range(len(prefix) - 1, max(-1, len(prefix) - 25), -1):
                if prefix[idx] in boundaries:
                    cut = idx + 1
                    break
            piece = rem[:cut].strip() or rem[:best].strip()
            pieces.append(piece)
            rem = rem[len(piece) :].strip()
        return pieces

    def _build_request_rows(self, prompt_codes: list[list[int]], text_token_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.manifest["tts_config"]
        tpl = self.manifest["prompt_templates"]
        n_vq = int(cfg["n_vq"])
        row_width = n_vq + 1
        audio_pad = int(cfg["audio_pad_token_id"])
        user_slot = int(cfg["audio_user_slot_token_id"])

        prefix_tokens = [*tpl["user_prompt_prefix_token_ids"], int(cfg["audio_start_token_id"])]
        suffix_tokens = [
            int(cfg["audio_end_token_id"]),
            *tpl["user_prompt_after_reference_token_ids"],
            *text_token_ids,
            *tpl["assistant_prompt_prefix_token_ids"],
            int(cfg["audio_start_token_id"]),
        ]

        total_rows = len(prefix_tokens) + len(prompt_codes) + len(suffix_tokens)
        input_ids = np.full((1, total_rows, row_width), audio_pad, dtype=np.int32)
        row_idx = 0
        for tid in prefix_tokens:
            input_ids[0, row_idx, 0] = int(tid)
            row_idx += 1
        for code_row in prompt_codes:
            input_ids[0, row_idx, 0] = user_slot
            limit = min(len(code_row), n_vq)
            input_ids[0, row_idx, 1 : 1 + limit] = np.asarray(code_row[:limit], dtype=np.int32)
            row_idx += 1
        for tid in suffix_tokens:
            input_ids[0, row_idx, 0] = int(tid)
            row_idx += 1

        attention_mask = np.ones((1, total_rows), dtype=np.int32)
        return input_ids, attention_mask

    def _generate_chunk_waveform(
        self,
        text_chunk: str,
        prompt_codes: list[list[int]],
        sample_mode: str,
        max_new_frames: int,
    ) -> np.ndarray:
        cfg = self.manifest["tts_config"]
        gen_defaults = self.manifest["generation_defaults"]
        n_vq = int(cfg["n_vq"])
        row_width = n_vq + 1
        audio_pad = int(cfg["audio_pad_token_id"])
        assistant_slot = int(cfg["audio_assistant_slot_token_id"])
        codebook_size = int(self.tts_meta["model_config"]["audio_codebook_sizes"][0])
        rep_penalty = float(gen_defaults.get("audio_repetition_penalty", 1.2))

        token_ids = self.encode_text(text_chunk)
        input_ids, attention_mask = self._build_request_rows(prompt_codes, token_ids)

        with self._infer_lock:
            prefill_outs = self.sessions["prefill"].run(
                None,
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )
            prefill_names = [o.name for o in self.sessions["prefill"].get_outputs()]
            named_prefill = dict(zip(prefill_names, prefill_outs, strict=True))

            hidden = named_prefill["global_hidden"]
            global_hidden = hidden[:, -1, :].astype(np.float32, copy=False) if hidden.ndim == 3 else hidden.astype(np.float32, copy=False)
            past_valid_length = int(attention_mask.shape[1])
            past_by_name = {
                name.replace("present_", "past_"): named_prefill[name]
                for name in self.tts_meta["onnx"]["prefill_output_names"][1:]
            }

            generated_frames: list[list[int]] = []
            seen_mask = np.zeros((1, n_vq, codebook_size), dtype=np.int32)
            prev_tokens_by_ch: list[list[int]] = [[] for _ in range(n_vq)]

            decode_input_names = self.tts_meta["onnx"]["decode_input_names"][2:]
            decode_output_names = self.tts_meta["onnx"]["decode_output_names"][1:]
            decode_session_out_names = [o.name for o in self.sessions["decode"].get_outputs()]

            for _step in range(max_new_frames):
                if sample_mode == "greedy" and "local_greedy_frame" in self.sessions:
                    outs = self.sessions["local_greedy_frame"].run(
                        None,
                        {
                            "global_hidden": global_hidden,
                            "repetition_seen_mask": seen_mask,
                            "repetition_penalty": np.asarray([rep_penalty], dtype=np.float32),
                        },
                    )
                    l_names = [o.name for o in self.sessions["local_greedy_frame"].get_outputs()]
                    l_named = dict(zip(l_names, outs, strict=True))
                    should_continue = bool(int(np.asarray(l_named["should_continue"]).reshape(-1)[0]))
                    if not should_continue:
                        break
                    frame = [int(x) for x in np.asarray(l_named["frame_token_ids"]).reshape(-1)]
                elif sample_mode == "fixed" and "local_fixed_sampled_frame" in self.sessions:
                    u_assist = np.asarray([min(0.99999994, max(0.0, float(self.rng.random())))], dtype=np.float32)
                    u_audio = np.asarray(
                        [[min(0.99999994, max(0.0, float(self.rng.random()))) for _ in range(n_vq)]],
                        dtype=np.float32,
                    )
                    outs = self.sessions["local_fixed_sampled_frame"].run(
                        None,
                        {
                            "global_hidden": global_hidden,
                            "repetition_seen_mask": seen_mask,
                            "assistant_random_u": u_assist,
                            "audio_random_u": u_audio,
                        },
                    )
                    l_names = [o.name for o in self.sessions["local_fixed_sampled_frame"].get_outputs()]
                    l_named = dict(zip(l_names, outs, strict=True))
                    should_continue = bool(int(np.asarray(l_named["should_continue"]).reshape(-1)[0]))
                    if not should_continue:
                        break
                    frame = [int(x) for x in np.asarray(l_named["frame_token_ids"]).reshape(-1)]
                else:
                    should_continue, frame = self._sample_full_frame(global_hidden, prev_tokens_by_ch)
                    if not should_continue:
                        break

                for ch_idx, tok in enumerate(frame):
                    prev_tokens_by_ch[ch_idx].append(tok)
                    if 0 <= tok < codebook_size:
                        seen_mask[0, ch_idx, tok] = 1
                generated_frames.append(frame)

                next_row = np.full((1, 1, row_width), audio_pad, dtype=np.int32)
                next_row[0, 0, 0] = assistant_slot
                next_row[0, 0, 1 : 1 + len(frame)] = np.asarray(frame, dtype=np.int32)

                decode_feeds: dict[str, np.ndarray] = {
                    "input_ids": next_row,
                    "past_valid_lengths": np.asarray([past_valid_length], dtype=np.int32),
                }
                for in_name in decode_input_names:
                    decode_feeds[in_name] = past_by_name[in_name]

                d_outs = self.sessions["decode"].run(None, decode_feeds)
                named_d = dict(zip(decode_session_out_names, d_outs, strict=True))
                d_hidden = named_d["global_hidden"]
                global_hidden = d_hidden[:, -1, :].astype(np.float32, copy=False) if d_hidden.ndim == 3 else d_hidden.astype(np.float32, copy=False)
                past_valid_length += 1
                past_by_name = {
                    name.replace("present_", "past_"): named_d[name]
                    for name in decode_output_names
                }

            if not generated_frames:
                return np.zeros((0, self.channels), dtype=np.float32)

            return self._decode_frames_unlocked(generated_frames)

    def _sample_full_frame(
        self,
        global_hidden: np.ndarray,
        prev_tokens_by_ch: list[list[int]],
    ) -> tuple[bool, list[int]]:
        cfg = self.manifest["tts_config"]
        gen = self.manifest["generation_defaults"]
        n_vq = int(cfg["n_vq"])
        assistant_slot = int(cfg["audio_assistant_slot_token_id"])
        audio_end = int(cfg["audio_end_token_id"])
        rep_penalty = float(gen.get("audio_repetition_penalty", 1.2))

        local_layers = int(self.tts_meta["model_config"]["local_layers"])
        local_heads = int(self.tts_meta["model_config"]["local_heads"])
        local_head_dim = int(self.tts_meta["model_config"]["local_head_dim"])
        local_past = {
            name: np.zeros((1, 0, local_heads, local_head_dim), dtype=np.float32)
            for idx in range(local_layers)
            for name in (f"local_past_key_{idx}", f"local_past_value_{idx}")
        }
        out_names = [o.name for o in self.sessions["local_cached_step"].get_outputs()]
        cached_out_names = self.tts_meta["onnx"]["local_cached_output_names"][2:]

        def step_local(text_tok: int, audio_tok: int, ch_idx: int, step_type: int, valid_len: int) -> tuple[np.ndarray, np.ndarray]:
            nonlocal local_past
            outs = self.sessions["local_cached_step"].run(
                None,
                {
                    "global_hidden": global_hidden,
                    "text_token_id": np.asarray([text_tok], dtype=np.int32),
                    "audio_token_id": np.asarray([audio_tok], dtype=np.int32),
                    "channel_index": np.asarray([ch_idx], dtype=np.int32),
                    "step_type": np.asarray([step_type], dtype=np.int32),
                    "past_valid_lengths": np.asarray([valid_len], dtype=np.int32),
                    **local_past,
                },
            )
            named = dict(zip(out_names, outs, strict=True))
            local_past = {k.replace("local_present_", "local_past_"): named[k] for k in cached_out_names}
            return named["text_logits"].reshape(-1), named["audio_logits"].reshape(-1)

        text_logits, _ = step_local(0, 0, 0, 0, 0)
        cand_ids = [assistant_slot, audio_end]
        cand_scores = np.asarray([text_logits[assistant_slot], text_logits[audio_end]], dtype=np.float32)
        chosen_idx = _sample_from_scores(
            cand_scores,
            do_sample=True,
            temperature=float(gen.get("text_temperature", 1.0)),
            top_k=2,
            top_p=float(gen.get("text_top_p", 1.0)),
            rng=self.rng,
        )
        if cand_ids[chosen_idx] != assistant_slot:
            return False, []

        frame: list[int] = []
        prev_tok = 0
        for ch_idx in range(n_vq):
            step_t = 1 if ch_idx == 0 else 2
            t_in = assistant_slot if ch_idx == 0 else 0
            a_in = 0 if ch_idx == 0 else prev_tok
            _, audio_logits = step_local(t_in, a_in, max(0, ch_idx - 1), step_t, ch_idx + 1)
            per_ch = int(audio_logits.shape[0]) // n_vq if audio_logits.shape[0] > codebook_size_hint(self.tts_meta) else int(audio_logits.shape[0])
            ch_scores = audio_logits[ch_idx * per_ch : (ch_idx + 1) * per_ch].copy() if audio_logits.shape[0] > per_ch else audio_logits.copy()
            for seen_id in set(prev_tokens_by_ch[ch_idx]):
                if 0 <= seen_id < ch_scores.shape[0]:
                    ch_scores[seen_id] = ch_scores[seen_id] * rep_penalty if ch_scores[seen_id] < 0 else ch_scores[seen_id] / rep_penalty
            tok = _sample_from_scores(
                ch_scores,
                do_sample=True,
                temperature=float(gen.get("audio_temperature", 0.8)),
                top_k=int(gen.get("audio_top_k", 25)),
                top_p=float(gen.get("audio_top_p", 0.95)),
                rng=self.rng,
            )
            frame.append(tok)
            prev_tok = tok
        return True, frame

    def _decode_frames_unlocked(self, generated_frames: list[list[int]]) -> np.ndarray:
        audio_codes = np.asarray([generated_frames], dtype=np.int32)
        outs = self.sessions["codec_decode"].run(
            None,
            {
                "audio_codes": audio_codes,
                "audio_code_lengths": np.asarray([len(generated_frames)], dtype=np.int32),
            },
        )
        names = [o.name for o in self.sessions["codec_decode"].get_outputs()]
        named = dict(zip(names, outs, strict=True))
        audio_len = int(np.asarray(named["audio_lengths"]).reshape(-1)[0])
        audio = named["audio"]  # shape (1, channels, samples)
        channel_arrays = [audio[0, ch, :audio_len].astype(np.float32, copy=False) for ch in range(audio.shape[1])]
        min_len = min(int(c.shape[0]) for c in channel_arrays)
        return np.stack([c[:min_len] for c in channel_arrays], axis=1)

    def synthesize_chunks_stream(
        self,
        *,
        text: str,
        voice: str | None = None,
        sample_mode: str | None = None,
        max_new_frames: int | None = None,
        seed: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield float32 waveform (samples, channels) per sentence chunk with chunk-level caching."""
        voice_key, prompt_codes = self.resolve_voice(voice)
        normalized_text = self.normalizer.normalize(text, voice=voice_key)
        if not normalized_text:
            raise ValueError("Input text is empty after normalization.")

        mode = (sample_mode or self.default_sample_mode).lower().strip()
        if mode not in {"fixed", "greedy", "full"}:
            mode = "fixed"
        frames_limit = int(max_new_frames or self.max_new_frames)
        if seed is not None:
            with self._infer_lock:
                self.rng = np.random.default_rng(int(seed))

        chunks = self.split_text_chunks(normalized_text)
        for idx, chunk_text in enumerate(chunks):
            chunk_cache_key = TtsCache.make_key(
                chunk=chunk_text,
                voice=voice_key,
                mode=mode,
                frames=frames_limit,
                seed=seed,
            )
            cached_wave = self.cache.get_chunk(chunk_cache_key)
            if cached_wave is not None:
                waveform = cached_wave
            else:
                waveform = self._generate_chunk_waveform(chunk_text, prompt_codes, mode, frames_limit)
                self.cache.set_chunk(chunk_cache_key, waveform)

            if idx < len(chunks) - 1:
                pause_sec = 0.40 if len(chunk_text.split()) <= 4 else 0.24
                pause_samples = max(0, int(round(self.sample_rate * pause_sec)))
                if pause_samples > 0:
                    pause_wave = np.zeros((pause_samples, self.channels), dtype=np.float32)
                    waveform = np.concatenate([waveform, pause_wave], axis=0)
            yield waveform

    def synthesize_waveform(
        self,
        *,
        text: str,
        voice: str | None = None,
        sample_mode: str | None = None,
        max_new_frames: int | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, str, str]:
        """Synthesize full waveform and return (waveform, resolved_voice_key, normalized_text)."""
        voice_key, _ = self.resolve_voice(voice)
        normalized_text = self.normalizer.normalize(text, voice=voice_key)
        parts = list(
            self.synthesize_chunks_stream(
                text=text,
                voice=voice,
                sample_mode=sample_mode,
                max_new_frames=max_new_frames,
                seed=seed,
            )
        )
        non_empty = [p for p in parts if p.size > 0]
        waveform = np.concatenate(non_empty, axis=0) if non_empty else np.zeros((0, self.channels), dtype=np.float32)
        return waveform, voice_key, normalized_text

    def warmup(self) -> None:
        logger.info("Running MOSS-TTS-Nano CPU warmup...")
        self.synthesize_waveform(
            text="你好。",
            voice=self.default_voice,
            sample_mode="greedy",
            max_new_frames=12,
            seed=1234,
        )
        logger.info("Warmup complete.")


def codebook_size_hint(tts_meta: dict[str, Any]) -> int:
    return int(tts_meta["model_config"]["audio_codebook_sizes"][0])
