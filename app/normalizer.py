from __future__ import annotations

import logging
import re
import threading
import unicodedata
from functools import lru_cache

logger = logging.getLogger(__name__)

_CJK_CHARS = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff"
_CJK = f"[{_CJK_CHARS}]"
_PROT = r"___PROT\d+___"
_ZH_WETEXT_KEEP_HYPHEN = "___KEEP_HYPHEN_BEFORE_ZH_WETEXT___"

_URL_RE = re.compile(r"https?://[^\s\u3000，。！？；、）】》〉」』]+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])")
_MENTION_RE = re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}")
_REDDIT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+")
_HASHTAG_RE = re.compile(r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+")
_DOT_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+")
_FILELIKE_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?=[A-Za-z0-9._/+:-]*[A-Za-z])"
    r"(?=[A-Za-z0-9._/+:-]*[./+:-])"
    r"[A-Za-z0-9][A-Za-z0-9._/+:-]*"
    r"(?![A-Za-z0-9_])"
)
_LATINISH = rf"(?:{_PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)"
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
_TRAILING_CLOSERS = set('"\')]}）】》〉」』”’')
ENGLISH_VOICES = frozenset({"Trump", "Ava", "Bella", "Adam", "Nathan"})


def normalize_tts_text(text: str) -> str:
    """Robust non-semantic cleanup for Chinese & multilingual TTS input."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u3000", " ")
    text = _ZERO_WIDTH_RE.sub("", text)
    text = "".join(ch for ch in text if ch in "\n\t " or not unicodedata.category(ch).startswith("C"))

    # Markdown links & lines
    text = re.sub(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)", r"\1 \2", text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^>\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        lines.append(line)

    if not lines:
        return ""
    merged = [lines[0]]
    for line in lines[1:]:
        merged[-1] = _ensure_terminal_punctuation(merged[-1])
        merged.append(line)
    text = "".join(merged)

    text = _normalize_flow_arrows(text)
    text, protected = _protect_spans(text)
    parts = re.split(rf"({_PROT})", text)
    text = "".join(part if re.fullmatch(_PROT, part) else part.replace("_", " ") for part in parts)

    text = _normalize_spaces(text)
    text = _normalize_structural_punctuation(text)
    text = _normalize_repeated_punctuation(text)
    text = _normalize_spaces(text)

    for idx, original in enumerate(protected):
        text = text.replace(f"___PROT{idx}___", original)
    text = text.strip()
    return "\n".join(_ensure_terminal_punctuation(l.strip()) if l.strip() else "" for l in text.split("\n")).strip()


def _protect_spans(text: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def repl(match: re.Match[str]) -> str:
        idx = len(protected)
        protected.append(match.group(0))
        return f"___PROT{idx}___"

    for pattern in (_URL_RE, _EMAIL_RE, _MENTION_RE, _REDDIT_RE, _HASHTAG_RE, _DOT_TOKEN_RE, _FILELIKE_RE):
        text = pattern.sub(repl, text)
    return text, protected


def _normalize_flow_arrows(text: str) -> str:
    return re.sub(r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*", "，", text)


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(rf"({_CJK})\s+(?={_CJK})", r"\1", text)
    text = re.sub(rf"({_CJK})\s+(?=\d)", r"\1", text)
    text = re.sub(rf"(\d)\s+(?={_CJK})", r"\1", text)
    text = re.sub(rf"({_CJK})(?=({_LATINISH}))", r"\1 ", text)
    text = re.sub(rf"(({_LATINISH}))(?={_CJK})", r"\1 ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([，。！？；：、”’」』】）》])", r"\1", text)
    text = re.sub(r"([（【「『《“‘])\s+", r"\1", text)
    text = re.sub(r"([，。！？；：、])\s*", r"\1", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    return re.sub(r" {2,}", " ", text).strip()


def _normalize_structural_punctuation(text: str) -> str:
    text = re.sub(r"\[\s*([^\[\]]+?)\s*\]", r'"\1"', text)
    text = re.sub(r"\{\s*([^{}]+?)\s*\}", r'"\1"', text)
    text = re.sub(r"[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]", r'"\1"', text)
    text = re.sub(
        r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
        r"\1\2",
        text,
    )
    text = _normalize_flow_arrows(text)
    return re.sub(r"\s*(?:—|–|―|-){2,}\s*", "。", text)


def _normalize_repeated_punctuation(text: str) -> str:
    text = re.sub(r"(?:\.{3,}|…{2,}|……+)", "。", text)
    text = re.sub(r"[。．]{2,}", "。", text)
    text = re.sub(r"[，,]{2,}", "，", text)
    text = re.sub(r"[!！]{2,}", "！", text)
    text = re.sub(r"[?？]{2,}", "？", text)

    def _mixed_qe(match: re.Match[str]) -> str:
        s = match.group(0)
        has_q = any(ch in s for ch in "?？")
        has_e = any(ch in s for ch in "!！")
        return "？！" if (has_q and has_e) else ("？" if has_q else "！")

    return re.sub(r"[!?！？]{2,}", _mixed_qe, text)


def _ensure_terminal_punctuation(text: str) -> str:
    if not text:
        return text
    index = len(text) - 1
    while index >= 0 and text[index].isspace():
        index -= 1
    while index >= 0 and text[index] in _TRAILING_CLOSERS:
        index -= 1
    if index >= 0 and unicodedata.category(text[index]).startswith("P"):
        return text
    return text + "。"


def _rewrite_hyphens_before_zh_wetext(text: str) -> str:
    """Prevent Chinese TN from misreading non-numeric hyphens as '减'."""
    rewritten = str(text or "")
    if "-" not in rewritten:
        return rewritten
    rewritten = re.sub(r"(^\s*)-\s*(?=\d)", rf"\1{_ZH_WETEXT_KEEP_HYPHEN}", rewritten)
    rewritten = re.sub(r"([=:+*/,(，:：；;（【\[{])\s*-\s*(?=\d)", rf"\1{_ZH_WETEXT_KEEP_HYPHEN}", rewritten)
    rewritten = re.sub(r"([\u3400-\u9fff])\s*-\s*(?=\d)", rf"\1{_ZH_WETEXT_KEEP_HYPHEN}", rewritten)
    rewritten = re.sub(r"(\d)\s*-\s*(?=\d)", rf"\1{_ZH_WETEXT_KEEP_HYPHEN}", rewritten)
    rewritten = re.sub(r"([\u3400-\u9fff])\s*-\s*(?=[\u3400-\u9fff])", r"\1，", rewritten)
    rewritten = re.sub(r"([^\s-])\s*-\s*(?=[^\s-])", r"\1 ", rewritten)
    rewritten = re.sub(r" {2,}", " ", rewritten).strip()
    return rewritten.replace(_ZH_WETEXT_KEEP_HYPHEN, "-")


def detect_language(text: str, voice: str = "") -> str:
    if re.search(r"[\u3400-\u9fff]", text):
        return "zh"
    if re.search(r"[A-Za-z]", text) and voice in ENGLISH_VOICES:
        return "en"
    if re.search(r"[A-Za-z]", text) and not re.search(r"[\u3400-\u9fff]", text):
        return "en"
    return "zh"


class TextNormalizer:
    """Thread-safe Chinese/English text normalizer with LRU caching."""

    def __init__(self, enable_wetext: bool = True, enable_robust: bool = True) -> None:
        self.enable_wetext = enable_wetext
        self.enable_robust = enable_robust
        self._lock = threading.Lock()
        self._normalizers: dict[str, object] | None = None

    def _get_normalizers(self) -> dict[str, object]:
        if self._normalizers is not None:
            return self._normalizers
        with self._lock:
            if self._normalizers is not None:
                return self._normalizers
            try:
                from wetext import Normalizer

                self._normalizers = {
                    "zh": Normalizer(lang="zh", operator="tn", remove_erhua=False),
                    "en": Normalizer(lang="en", operator="tn"),
                }
                logger.info("Loaded wetext WFST normalizers (zh, en)")
            except Exception as exc:
                logger.warning("wetext unavailable, falling back to robust normalizer only: %s", exc)
                self._normalizers = {}
            return self._normalizers

    @lru_cache(maxsize=2048)
    def normalize(self, text: str, voice: str = "Junhao") -> str:
        current = str(text or "").strip()
        if not current:
            return ""
        if self.enable_robust and self.enable_wetext:
            current = normalize_tts_text(current)
        if self.enable_wetext:
            normalizers = self._get_normalizers()
            lang = detect_language(current, voice)
            normalizer = normalizers.get(lang)
            if normalizer is not None:
                if lang == "zh":
                    current = _rewrite_hyphens_before_zh_wetext(current)
                with self._lock:
                    current = normalizer.normalize(current)
        if self.enable_robust:
            current = normalize_tts_text(current)
        return current
