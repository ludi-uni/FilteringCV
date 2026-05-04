"""NFA（CTM）トークン列と OpenJTalk G2P から ``nfa_to_g2p_token_map_path`` 用 YAML 草案を集計する（人手レビュー前提）。"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from cv_preprocess.audio.decode import load_audio
from cv_preprocess.audio.nfa_batch import close_nfa_worker, run_nfa_align_batch
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import run_early_audio_gate, run_quality_gate
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import (
    apply_merge_filtered_speakers_as_one,
    filter_by_clip_metadata,
    filter_by_speakers,
    iter_clip_audio_paths,
    load_validated_tsv,
)
from cv_preprocess.pipeline.mfa_g2p_map_suggest import Strategy, _collect_votes_for_row, _g2p_tokens
from cv_preprocess.pipeline.token_map_suggest import (
    finalize_g2p_token_suggestions,
    merge_suggestion_dict_into_base,
)
from cv_preprocess.pipeline.preprocess import (
    _compute_clip_mora_count_once,
    _merged_quality_gate_for_nfa_prefilter,
    _mora_gates_needed,
)
from cv_preprocess.pipeline.preprocess_efficiency import resolve_preprocess_pass1_pipeline
from cv_preprocess.text.language_detect import passes_japanese_policy, passes_locale_expected
from cv_preprocess.text.mfa_token_map import load_mfa_token_map_yaml
from cv_preprocess.text.normalize import normalize_for_tts


@dataclass
class _NfaMapPending:
    nfa_utt_id: str
    y: np.ndarray
    sr: int
    text_norm: str
    g2p_toks: list[str]


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

    root = cfg.input.corpus_root
    tsv_path = root / cfg.input.clip_tsv
    rows, load_stats = load_validated_tsv(tsv_path)
    include_ids = cfg.speakers.include_client_ids or None
    rows = filter_by_speakers(rows, include_ids)
    rows = filter_by_clip_metadata(rows, cfg.speakers.clip_metadata_filters)
    rows = sorted(rows, key=lambda r: r.path)
    if max_clips is not None and max_clips >= 0:
        rows = rows[: int(max_clips)]
    apply_merge_filtered_speakers_as_one(
        rows,
        enabled=cfg.speakers.merge_filtered_speakers_as_one,
        merged_client_id=cfg.speakers.resolved_merged_speaker_client_id(),
    )

    lang = (cfg.input.locale_expected or "ja").split("-")[0]
    pipeline_for_pass1 = resolve_preprocess_pass1_pipeline(cfg)
    nfa_prefilter_qg = (
        _merged_quality_gate_for_nfa_prefilter(cfg)
        if cfg.nfa_gate.prefilter.enabled
        else None
    )
    prefilter_mora_fail_reason = "nfa_prefilter_mora_estimate_failed" if nfa_prefilter_qg is not None else None
    mora_early, mora_pref, mora_fin = _mora_gates_needed(lang, cfg, nfa_prefilter_qg)
    work_root = (work_parent if work_parent is not None else output_yaml.parent).resolve()

    base_map: dict[str, str] = {}
    if existing_map_path is not None:
        base_map = dict(load_mfa_token_map_yaml(existing_map_path))

    per_nfa: defaultdict[str, Counter[str]] = defaultdict(Counter)
    method_counts = {"zip": 0, "proportional": 0, "none": 0}
    counts: dict[str, int] = {
        "rows_considered": 0,
        "skipped_text_locale": 0,
        "skipped_text_not_japanese": 0,
        "skipped_text_length": 0,
        "skipped_phonemize_failed": 0,
        "skipped_missing_audio": 0,
        "skipped_decode_failed": 0,
        "skipped_nan_inf_audio": 0,
        "skipped_early_audio_gate": 0,
        "skipped_audio_pipeline_failed": 0,
        "skipped_nfa_prefilter": 0,
        "skipped_mora_estimate_failed": 0,
        "skipped_empty_nfa_tokens": 0,
        "skipped_empty_g2p": 0,
        "skipped_nfa_align_failed": 0,
        "pairs_collected": 0,
    }

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

    batch_rows: list[_NfaMapPending] = []

    def flush_batch() -> None:
        nonlocal batch_rows
        if not batch_rows:
            return
        items = [(r.nfa_utt_id, r.y, r.sr, r.text_norm) for r in batch_rows]
        results = run_nfa_align_batch(cfg.nfa_gate, items, work_parent=work_root)
        for pend, res in zip(batch_rows, results, strict=True):
            if not res.ok or not (res.token_string or "").strip():
                counts["skipped_nfa_align_failed"] += 1
                continue
            nfa_toks = [t for t in res.token_string.replace("\t", " ").split() if t.strip()]
            if not nfa_toks:
                counts["skipped_empty_nfa_tokens"] += 1
                continue
            counts["rows_considered"] += 1
            pairs, method = _collect_votes_for_row(nfa_toks, pend.g2p_toks, strategy)
            method_counts[method] += 1
            for nt, gt in pairs:
                per_nfa[nt][gt] += 1
                counts["pairs_collected"] += 1
        batch_rows.clear()

    utt_i = 0
    for row in row_iter:
        if not passes_locale_expected(row.locale, cfg.input.locale_expected):
            counts["skipped_text_locale"] += 1
            continue
        text_norm = normalize_for_tts(row.sentence)
        if not passes_japanese_policy(text_norm, cfg.text.require_japanese):
            counts["skipped_text_not_japanese"] += 1
            continue
        tl = len(text_norm)
        if tl < cfg.text.min_text_len or tl > cfg.text.max_text_len:
            counts["skipped_text_length"] += 1
            continue
        try:
            g2p_toks = _g2p_tokens(text_norm, kana=cfg.text.g2p_kana)
        except Exception:
            counts["skipped_phonemize_failed"] += 1
            continue
        if not g2p_toks:
            counts["skipped_empty_g2p"] += 1
            continue

        clip_path = iter_clip_audio_paths(root, cfg.input.audio_subdir, row)
        if not clip_path.is_file():
            counts["skipped_missing_audio"] += 1
            continue
        try:
            y, sr = load_audio(clip_path)
        except Exception:
            counts["skipped_decode_failed"] += 1
            continue
        if not np.isfinite(y).all():
            counts["skipped_nan_inf_audio"] += 1
            continue

        clip_mora_count, mora_reject_reason = _compute_clip_mora_count_once(
            text_norm,
            need_early=mora_early,
            need_pref=mora_pref,
            need_final=mora_fin,
            prefilter_mora_fail_reason=prefilter_mora_fail_reason,
        )
        if mora_reject_reason is not None:
            counts["skipped_mora_estimate_failed"] += 1
            continue

        if cfg.early_audio_gate.enabled:
            tsr = int(pipeline_for_pass1.target_sample_rate)
            y_chk = (
                resample_audio(np.asarray(y, dtype=np.float32), sr, tsr)
                if sr != tsr
                else np.asarray(y, dtype=np.float32)
            )
            mora_for_early = clip_mora_count if mora_early else None
            eg = run_early_audio_gate(
                y_chk,
                tsr,
                text_len=len(text_norm),
                mora_count=mora_for_early,
                main_gate=cfg.quality_gate,
                snr_cfg=cfg.snr,
                early=cfg.early_audio_gate,
            )
            if not eg.ok:
                counts["skipped_early_audio_gate"] += 1
                continue

        try:
            y2, sr2, _ameta = run_steps_on_array(y, sr, pipeline_for_pass1)
        except Exception:
            counts["skipped_audio_pipeline_failed"] += 1
            continue

        if nfa_prefilter_qg is not None:
            mora_pf = clip_mora_count if mora_pref else None
            gate_pf = run_quality_gate(
                y2,
                sr2,
                text_len=len(text_norm),
                gate=nfa_prefilter_qg,
                snr_cfg=cfg.snr,
                mora_count=mora_pf,
            )
            if not gate_pf.ok:
                counts["skipped_nfa_prefilter"] += 1
                continue

        utt_i += 1
        pend = _NfaMapPending(
            nfa_utt_id=f"u{utt_i:08d}",
            y=y2,
            sr=sr2,
            text_norm=text_norm,
            g2p_toks=g2p_toks,
        )
        batch_rows.append(pend)
        if len(batch_rows) >= cfg.nfa_gate.batch_size:
            flush_batch()

    flush_batch()

    suggested, ambiguous, skipped_weak = finalize_g2p_token_suggestions(
        per_nfa,
        min_votes=min_votes,
        min_ratio=min_ratio,
        report_source_key="nfa",
    )

    out_map, added, skipped_existing, overwritten = merge_suggestion_dict_into_base(
        base_map,
        suggested,
        fill_missing_keys_only=fill_missing_keys_only,
    )

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# AUTO-GENERATED DRAFT — NeMo NFA CTM tokens → OpenJTalk G2P single-token hints.\n"
        "# Proportional alignment is approximate; review ambiguous_nfa / skipped_low_confidence in *_report.json.\n"
        "# Empty value \"\" omits a token in map_mfa_space_separated_to_g2p_tokens.\n"
    )
    body = yaml.safe_dump(out_map, allow_unicode=True, default_flow_style=False, sort_keys=True)
    output_yaml.write_text(header + body, encoding="utf-8")

    report_path = output_yaml.parent / f"{output_yaml.stem}_report.json"
    report: dict[str, Any] = {
        "report_schema_version": 1,
        "stage": "nfa_g2p_map_suggest",
        "strategy": strategy,
        "output_yaml": str(output_yaml.resolve()),
        "clip_tsv": str(tsv_path.resolve()),
        "load_stats": load_stats,
        "rows_after_filters": len(rows),
        "counts": counts,
        "method_counts": method_counts,
        "min_votes": min_votes,
        "min_ratio": min_ratio,
        "existing_map_path": str(existing_map_path.resolve()) if existing_map_path else None,
        "fill_missing_keys_only": fill_missing_keys_only,
        "suggested_entries": len(suggested),
        "yaml_keys_written": len(out_map),
        "new_keys_merged": added,
        "suggestions_overwritten_existing": overwritten,
        "skipped_existing_key": skipped_existing,
        "ambiguous_nfa": ambiguous,
        "skipped_low_confidence": skipped_weak,
        "unique_nfa_in_votes": len(per_nfa),
        "nfa_pretrained_name": cfg.nfa_gate.pretrained_name,
        "nfa_model_path": str(cfg.nfa_gate.model_path.resolve()) if cfg.nfa_gate.model_path else None,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path.resolve())
    if cfg.nfa_gate.enabled:
        close_nfa_worker()
    return report
