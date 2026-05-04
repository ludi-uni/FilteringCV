"""アライメント由来の音素列マニフェスト（JSON Lines）。各行は TSV の path と対応づける。"""

from __future__ import annotations

import json
from pathlib import Path


def load_alignment_phoneme_manifest(path: Path) -> dict[str, str]:
    """
    JSONL。1 行あたり:

    - ``source_path``: ``validated.tsv`` の ``path`` 列と同一（例: ``clip_xxx.mp3``）
    - ``phonemes``: アライメントで得た音素列（空白区切り推奨。G2P と同じ粒度なら比較可能）

    重複キーは後勝ち。
    """
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            sp = obj.get("source_path")
            ph = obj.get("phonemes")
            if not sp or ph is None:
                raise ValueError(
                    f"{path}:{line_no}: each line must have source_path and phonemes strings"
                )
            out[str(sp).strip()] = str(ph).strip()
    return out
