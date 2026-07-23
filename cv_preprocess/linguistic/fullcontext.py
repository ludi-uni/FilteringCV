"""OpenJTalk full-context label extraction with graceful degradation."""

from __future__ import annotations

import re
from collections.abc import Sequence

_ACCENT_PHRASE_RE = re.compile(r"^F:(\d+)_(\d+)#")
_NUCLEUS_RE = re.compile(r"^E:(\d+)_(\d+)!.*-(\d+)")


def extract_fullcontext_labels(text: str) -> tuple[list[str] | None, str | None]:
    """Return full-context labels and an optional warning. Never raises."""
    if not text.strip():
        return None, "fullcontext skipped: empty text"
    try:
        import pyopenjtalk
    except ImportError as exc:
        return None, f"fullcontext unavailable: pyopenjtalk not installed ({exc})"

    try:
        labels = pyopenjtalk.extract_fullcontext(text)
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for optional dependency
        return None, f"fullcontext unavailable: {exc}"

    if not labels:
        return None, "fullcontext returned no labels"
    return list(labels), None


def parse_accent_phrase_lengths(labels: Sequence[str]) -> list[int]:
    """Best-effort accent phrase lengths from full-context F fields."""
    lengths: list[int] = []
    seen: set[int] = set()
    for label in labels:
        for part in label.split("/"):
            match = _ACCENT_PHRASE_RE.match(part)
            if not match:
                continue
            phrase_len = int(match.group(1))
            if phrase_len > 0 and phrase_len not in seen:
                seen.add(phrase_len)
                lengths.append(phrase_len)
    return lengths


def parse_accent_nucleus_features(labels: Sequence[str]) -> list[str]:
    """Best-effort accent nucleus markers from full-context E fields."""
    nuclei: list[str] = []
    for label in labels:
        phone = label.split("-", maxsplit=1)[0]
        for part in label.split("/"):
            match = _NUCLEUS_RE.match(part)
            if not match:
                continue
            phrase_len = match.group(1)
            nucleus_pos = match.group(3)
            feature = f"nucleus@{phone}:p{phrase_len}_n{nucleus_pos}"
            if feature not in nuclei:
                nuclei.append(feature)
    return nuclei


def accent_phrase_length_band(length: int) -> str:
    if length <= 2:
        return "1-2"
    if length <= 4:
        return "3-4"
    if length <= 7:
        return "5-7"
    return "8+"
