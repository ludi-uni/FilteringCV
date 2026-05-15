"""NFA（CTM）トークン列と OpenJTalk G2P から ``nfa_to_g2p_token_map_path`` 用 YAML 草案を集計する（人手レビュー前提）。"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cv_preprocess.audio.nfa_batch import close_nfa_worker
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import load_clip_rows_for_pipeline
from cv_preprocess.pipeline.g2p_map_suggest_core import (
    Strategy,
    finalize_and_write_g2p_map_draft,
    new_method_counts,
    try_normalize_and_g2p_tokens,
)
from cv_preprocess.pipeline.nfa_g2p_map_suggest_pipeline import (
    NfaG2pMapBatchVoter,
    build_nfa_map_pass1_context,
    new_nfa_g2p_map_suggest_counts,
    try_build_nfa_map_pending,
)
from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml


def run_nfa_g2p_map_suggest(
    cfg: PipelineConfig,
    *,
    output_yaml: Path,
    strategy: Strategy = "adaptive",
    min_votes: int = 2,
    min_ratio: float = 0.55,
    existing_map_path: Path | None = None,
    fill_missing_keys_only: bool = True,
    show_progress: bool = True,
    max_clips: int | None = None,
    work_parent: Path | None = None,
) -> dict[str, Any]:
    """
    ``validated.tsv`` を preprocess と同様に走査し、pass1 音声パイプライン後の波形で NFA を実行、
    (NFA CTM トークン → G2P トークン) の投票を集計して YAML 草案を書く。

    * **phoneme_alignment_check** はこのコマンドでは参照しない（マニフェスト未整備でも集計できる）。
    * **比例配置**（長さ不一致時）は近似であり誤提案が混ざる。必ず *_report.json を確認して人手で直すこと。
    """
    if not cfg.nfa_gate.enabled:
        raise ValueError("suggest-nfa-g2p-map には nfa_gate.enabled=true が必要です。")
    if not cfg.text.phonemize:
        raise ValueError("suggest-nfa-g2p-map には text.phonemize=true が必要です。")

    loaded = load_clip_rows_for_pipeline(cfg, max_clips_head=max_clips)
    rows = loaded.rows
    load_stats = loaded.load_stats
    tsv_path = cfg.input.corpus_root / cfg.input.clip_tsv

    pass1_ctx = build_nfa_map_pass1_context(cfg)
    work_root = (work_parent if work_parent is not None else output_yaml.parent).resolve()

    base_map: dict[str, str] = {}
    if existing_map_path is not None:
        base_map = dict(load_mfa_token_map_yaml(existing_map_path))

    per_nfa: defaultdict[str, Counter[str]] = defaultdict(Counter)
    method_counts = new_method_counts()
    counts = new_nfa_g2p_map_suggest_counts()

    if show_progress:
        print(
            f"[cv-preprocess] suggest-nfa-g2p-map: clips={len(rows)} "
            f"strategy={strategy} batch_size={cfg.nfa_gate.batch_size}",
            file=sys.stderr,
            flush=True,
        )

    row_iter = tqdm(rows, desc="suggest-nfa-g2p-map", unit="clip", file=sys.stderr, dynamic_ncols=True) if (
        show_progress and rows
    ) else rows

    voter = NfaG2pMapBatchVoter(
        cfg,
        strategy=strategy,
        work_root=work_root,
        per_nfa=per_nfa,
        method_counts=method_counts,
        counts=counts,
    )

    utt_i = 0
    for row in row_iter:
        text_g2p = try_normalize_and_g2p_tokens(row, cfg, counts)
        if text_g2p is None:
            continue
        text_norm, g2p_toks = text_g2p

        utt_i += 1
        pending = try_build_nfa_map_pending(
            row,
            cfg,
            pass1_ctx,
            counts,
            text_norm=text_norm,
            g2p_toks=g2p_toks,
            utter_index=utt_i,
        )
        if pending is None:
            continue
        voter.append(pending)

    voter.flush()

    yaml_header = (
        "# AUTO-GENERATED DRAFT — NeMo NFA CTM tokens → OpenJTalk G2P single-token hints.\n"
        "# Proportional alignment is approximate; review ambiguous_nfa / skipped_low_confidence in *_report.json.\n"
        "# Empty value \"\" omits a token in map_mfa_space_separated_to_g2p_tokens.\n"
    )
    report = finalize_and_write_g2p_map_draft(
        per_source=per_nfa,
        base_map=base_map,
        output_yaml=output_yaml,
        yaml_header=yaml_header,
        report_source_key="nfa",
        stage="nfa_g2p_map_suggest",
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
        extra_report_fields={
            "nfa_pretrained_name": cfg.nfa_gate.pretrained_name,
            "nfa_model_path": str(cfg.nfa_gate.model_path.resolve()) if cfg.nfa_gate.model_path else None,
        },
    )
    if cfg.nfa_gate.enabled:
        close_nfa_worker()
    return report
