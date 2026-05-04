from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cv_preprocess.audio.quality_gate import GateResult
from cv_preprocess.io.tsv_loader import ClipRow


@dataclass
class PendingClip:
    row: ClipRow
    y: np.ndarray
    sr: int
    text_raw: str
    text_norm: str
    phonemes: str | None
    excerpt: str
    ameta: dict[str, object]
    mfa_utt_id: str = ""
    nfa_utt_id: str = ""
    #: 最終品質ゲート用。日本語で ``min_sec_per_mora`` 有効時にループ先頭で一度だけ算出した値。
    mora_count: int | None = None
    #: prefilter と本番ゲートが同一かつ ``two_pass_denoise`` 無効時のみ。最終 ``run_quality_gate`` の短絡用。
    prefilter_final_gate_reuse: GateResult | None = None
    prefilter_final_gate_fp: str | None = None
