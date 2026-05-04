from __future__ import annotations

import re
import unicodedata


def normalize_text_basic(s: str) -> str:
    """NFKC + whitespace collapse + strip (MVP)."""
    t = unicodedata.normalize("NFKC", s)
    t = t.replace("\u00a0", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


_JP_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\u3400-\u4dbf]")


def japanese_char_ratio(s: str) -> float:
    if not s:
        return 0.0
    n = len(_JP_RE.findall(s))
    return n / max(len(s), 1)


def is_mostly_japanese(s: str, min_ratio: float = 0.5) -> bool:
    return japanese_char_ratio(s) >= min_ratio


def normalize_for_tts(s: str) -> str:
    """Spec §7 MVP: NFKC, unify spaces; light punctuation (keep 。、)."""
    t = normalize_text_basic(s)
    # Collapse repeated punctuation lightly
    t = re.sub(r"\.{2,}", ".", t)
    return t.strip()
