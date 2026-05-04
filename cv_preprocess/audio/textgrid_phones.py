"""MFA が出力する Praat long TextGrid から音素区間を抽出する（依存ライブラリなし）。"""

from __future__ import annotations

import re
from pathlib import Path


def extract_phone_tokens_from_textgrid(path: Path) -> list[str]:
    """
    ``phones`` を名前に含む最初の IntervalTier から、空でない interval text を順に返す。
    無音扱いのラベル（sp, sil, eps 等）は除外する。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    # item [n]: ... IntervalTier ... name = "..." ... intervals ...
    items = re.split(r"\n\s*item\s*\[\d+\]:\s*\n", raw)
    if not items:
        return []
    phone_block: str | None = None
    for block in items[1:]:
        if "class = " not in block:
            continue
        if "IntervalTier" not in block.split("class = ", 1)[1].split("\n", 1)[0]:
            continue
        mname = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not mname:
            continue
        name = mname.group(1)
        if "phones" not in name.lower():
            continue
        phone_block = block
        break
    if not phone_block:
        return []

    tokens: list[str] = []
    for im in re.finditer(
        r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"([^\"]*)\"",
        phone_block,
    ):
        t = im.group(3).strip()
        if not t:
            continue
        tl = t.lower()
        if tl in ("sp", "sil", "eps", "silence", "<sil>", "<unk>"):
            continue
        tokens.append(t)
    return tokens


def phones_to_space_string(tokens: list[str]) -> str:
    return " ".join(tokens)
