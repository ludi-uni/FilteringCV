"""Japanese G2P via pyopenjtalk. C 層が stderr に出す WARNING を既定で抑制する。
環境変数 ``CV_PHONEMIZE_VERBOSE=1``（true/yes/on 可）で抑制を無効化できる。
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator


@contextlib.contextmanager
def _stderr_fd_to_devnull() -> Iterator[None]:
    """Route FD 2 to devnull so libc/pyopenjtalk C warnings do not clutter stderr."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_copy = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(stderr_copy, 2)
        os.close(devnull_fd)
        os.close(stderr_copy)


def _suppress_openjtalk_stderr() -> bool:
    return os.environ.get("CV_PHONEMIZE_VERBOSE", "").strip().lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def g2p_phonemes(text: str, *, kana: bool = False) -> str:
    """Grapheme-to-phoneme using pyopenjtalk-plus (import name: pyopenjtalk)."""
    import pyopenjtalk

    if _suppress_openjtalk_stderr():
        with _stderr_fd_to_devnull():
            return pyopenjtalk.g2p(text, kana=kana)
    return pyopenjtalk.g2p(text, kana=kana)


def g2p_phonemes_for_dataset(
    text: str,
    *,
    kana: bool = False,
    word_separator: str | None = None,
) -> str:
    """学習用音素列。``text`` の空白区切り語ごとに G2P し、``word_separator`` で結合する（仕様 §13 の ``|`` 相当）。"""
    if kana or not word_separator:
        return g2p_phonemes(text, kana=kana)
    parts = [p for p in text.split() if p.strip()]
    if len(parts) <= 1:
        return g2p_phonemes(text, kana=kana)
    segs = [g2p_phonemes(p, kana=kana).strip() for p in parts]
    segs = [s for s in segs if s]
    if not segs:
        return g2p_phonemes(text, kana=kana)
    sep = f" {word_separator.strip()} "
    return sep.join(segs)
