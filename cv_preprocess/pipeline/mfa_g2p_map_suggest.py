"""MFA phones と OpenJTalk G2P から YAML マップの草案を集計する（人手レビュー前提）。"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import load_clip_rows_for_pipeline
from cv_preprocess.pipeline.g2p_map_suggest_core import (
    Strategy,
    _collect_votes_for_row,
    _g2p_tokens,
    _pairs_proportional,
    _pairs_zip,
    accumulate_pairs_vote,
    finalize_and_write_g2p_map_draft,
    new_mfa_g2p_map_suggest_counts,
    new_method_counts,
    try_load_mfa_textgrid_phone_tokens,
    try_normalize_and_g2p_tokens,
)
from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml


def run_mfa_g2p_map_suggest(
    cfg: PipelineConfig,
    *,
    mfa_textgrid_root: Path,
    output_yaml: Path,
    strategy: Strategy = "adaptive",
    min_votes: int = 2,
    min_ratio: float = 0.55,
    existing_map_path: Path | None = None,
    fill_missing_keys_only: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    ``validated.tsv`` を ``phoneme-manifest`` と同様に走査し、各行で MFA phones と G2P を並べ、
    (MFA トークン → G2P トークン) の投票を集計して YAML 草案を書く。

    * **adaptive**: 長さが一致する行は zip、それ以外は比例配置でペアを作る。
    * **zip_only** / **proportional_only**: 名前どおり。

    比例配置は記号粒度が異なるときの近似であり、**誤提案が混ざる**。必ずレポートと突き合わせて人手で直すこと。
    """
    if not cfg.text.phonemize:
        raise ValueError("suggest-mfa-g2p-map には text.phonemize=true が必要です。")
    if not mfa_textgrid_root.is_dir():
        raise ValueError(f"mfa_textgrid_root must be a directory: {mfa_textgrid_root}")

    loaded = load_clip_rows_for_pipeline(cfg)
    rows = loaded.rows
    load_stats = loaded.load_stats
    tsv_path = cfg.input.corpus_root / cfg.input.clip_tsv

    base_map: dict[str, str] = {}
    if existing_map_path is not None:
        base_map = dict(load_mfa_token_map_yaml(existing_map_path))

    per_mfa: defaultdict[str, Counter[str]] = defaultdict(Counter)
    method_counts = new_method_counts()
    counts = new_mfa_g2p_map_suggest_counts()

    if show_progress:
        print(
            f"[cv-preprocess] suggest-mfa-g2p-map: clips={len(rows)} "
            f"tg_root={mfa_textgrid_root} strategy={strategy}",
            file=sys.stderr,
            flush=True,
        )

    row_iter = tqdm(rows, desc="suggest-mfa-g2p-map", unit="clip", file=sys.stderr, dynamic_ncols=True) if (
        show_progress and rows
    ) else rows

    for row in row_iter:
        text_g2p = try_normalize_and_g2p_tokens(row, cfg, counts)
        if text_g2p is None:
            continue
        _, g2p_toks = text_g2p

        mfa_toks = try_load_mfa_textgrid_phone_tokens(row.path, mfa_textgrid_root, counts)
        if mfa_toks is None:
            continue

        accumulate_pairs_vote(
            mfa_toks,
            g2p_toks,
            strategy,
            per_mfa,
            method_counts,
            counts,
        )

    yaml_header = (
        "# AUTO-GENERATED DRAFT — MFA phones → OpenJTalk G2P single-token hints.\n"
        "# Proportional alignment is approximate; review against docs/音素照合マニフェスト.md.\n"
        "# Empty value \"\" omits a token in map_mfa_space_separated_to_g2p_tokens.\n"
    )
    return finalize_and_write_g2p_map_draft(
        per_source=per_mfa,
        base_map=base_map,
        output_yaml=output_yaml,
        yaml_header=yaml_header,
        report_source_key="mfa",
        stage="mfa_g2p_map_suggest",
        strategy=strategy,
        min_votes=min_votes,
        min_ratio=min_ratio,
        fill_missing_keys_only=fill_missing_keys_only,
        existing_map_path=existing_map_path,
        counts=counts,
        method_counts=method_counts,
        load_stats=load_stats,
        rows_after_filters=len(rows),
        clip_tsv=tsv_path,
        extra_report_fields={"mfa_textgrid_root": str(mfa_textgrid_root.resolve())},
    )
