"""suggest-nfa-g2p-map 用: preprocess 相当の pass1 音声処理と NFA バッチ投票。"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cv_preprocess.audio.decode import load_audio
from cv_preprocess.audio.nfa_batch import run_nfa_align_batch
from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import run_early_audio_gate, run_quality_gate
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import PipelineConfig, QualityGateConfig
from cv_preprocess.config.audio_steps import AudioPipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow, iter_clip_audio_paths
from cv_preprocess.pipeline.g2p_map_suggest_core import Strategy, accumulate_pairs_vote
from cv_preprocess.pipeline.preprocess import (
    _compute_clip_mora_count_once,
    _merged_quality_gate_for_nfa_prefilter,
    _mora_gates_needed,
)
from cv_preprocess.pipeline.preprocess_efficiency import resolve_preprocess_pass1_pipeline


@dataclass(frozen=True)
class NfaMapPass1Context:
    corpus_root: Path
    audio_subdir: str
    pipeline_for_pass1: AudioPipelineConfig
    nfa_prefilter_qg: QualityGateConfig | None
    prefilter_mora_fail_reason: str | None
    mora_early: bool
    mora_pref: bool
    mora_fin: bool


def build_nfa_map_pass1_context(cfg: PipelineConfig) -> NfaMapPass1Context:
    lang = (cfg.input.locale_expected or "ja").split("-")[0]
    nfa_prefilter_qg = (
        _merged_quality_gate_for_nfa_prefilter(cfg)
        if cfg.nfa_gate.prefilter.enabled
        else None
    )
    prefilter_mora_fail_reason = (
        "nfa_prefilter_mora_estimate_failed" if nfa_prefilter_qg is not None else None
    )
    mora_early, mora_pref, mora_fin = _mora_gates_needed(lang, cfg, nfa_prefilter_qg)
    return NfaMapPass1Context(
        corpus_root=cfg.input.corpus_root,
        audio_subdir=cfg.input.audio_subdir,
        pipeline_for_pass1=resolve_preprocess_pass1_pipeline(cfg),
        nfa_prefilter_qg=nfa_prefilter_qg,
        prefilter_mora_fail_reason=prefilter_mora_fail_reason,
        mora_early=mora_early,
        mora_pref=mora_pref,
        mora_fin=mora_fin,
    )


@dataclass
class NfaMapPending:
    nfa_utt_id: str
    y: np.ndarray
    sr: int
    text_norm: str
    g2p_toks: list[str]


def new_nfa_g2p_map_suggest_counts() -> dict[str, int]:
    return {
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


def try_build_nfa_map_pending(
    row: ClipRow,
    cfg: PipelineConfig,
    ctx: NfaMapPass1Context,
    counts: dict[str, int],
    *,
    text_norm: str,
    g2p_toks: list[str],
    utter_index: int,
) -> NfaMapPending | None:
    """1 クリップを decode → モーラ → early gate → pass1 → NFA prefilter まで通し、NFA 投入用 pending を返す。"""
    clip_path = iter_clip_audio_paths(ctx.corpus_root, ctx.audio_subdir, row)
    if not clip_path.is_file():
        counts["skipped_missing_audio"] += 1
        return None
    try:
        y, sr = load_audio(clip_path)
    except Exception:
        counts["skipped_decode_failed"] += 1
        return None
    if not np.isfinite(y).all():
        counts["skipped_nan_inf_audio"] += 1
        return None

    clip_mora_count, mora_reject_reason = _compute_clip_mora_count_once(
        text_norm,
        need_early=ctx.mora_early,
        need_pref=ctx.mora_pref,
        need_final=ctx.mora_fin,
        prefilter_mora_fail_reason=ctx.prefilter_mora_fail_reason,
    )
    if mora_reject_reason is not None:
        counts["skipped_mora_estimate_failed"] += 1
        return None

    if cfg.early_audio_gate.enabled:
        tsr = int(ctx.pipeline_for_pass1.target_sample_rate)
        y_chk = (
            resample_audio(np.asarray(y, dtype=np.float32), sr, tsr)
            if sr != tsr
            else np.asarray(y, dtype=np.float32)
        )
        mora_for_early = clip_mora_count if ctx.mora_early else None
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
            return None

    try:
        y2, sr2, _ameta = run_steps_on_array(y, sr, ctx.pipeline_for_pass1)
    except Exception:
        counts["skipped_audio_pipeline_failed"] += 1
        return None

    if ctx.nfa_prefilter_qg is not None:
        mora_pf = clip_mora_count if ctx.mora_pref else None
        gate_pf = run_quality_gate(
            y2,
            sr2,
            text_len=len(text_norm),
            gate=ctx.nfa_prefilter_qg,
            snr_cfg=cfg.snr,
            mora_count=mora_pf,
        )
        if not gate_pf.ok:
            counts["skipped_nfa_prefilter"] += 1
            return None

    return NfaMapPending(
        nfa_utt_id=f"u{utter_index:08d}",
        y=y2,
        sr=sr2,
        text_norm=text_norm,
        g2p_toks=g2p_toks,
    )


class NfaG2pMapBatchVoter:
    """NFA align バッチを溜め、CTM トークンと G2P の投票を集計する。"""

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        strategy: Strategy,
        work_root: Path,
        per_nfa: defaultdict[str, Counter[str]],
        method_counts: dict[str, int],
        counts: dict[str, int],
    ) -> None:
        self.cfg = cfg
        self.strategy = strategy
        self.work_root = work_root
        self.per_nfa = per_nfa
        self.method_counts = method_counts
        self.counts = counts
        self._batch: list[NfaMapPending] = []

    def append(self, pending: NfaMapPending) -> None:
        self._batch.append(pending)
        if len(self._batch) >= self.cfg.nfa_gate.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._batch:
            return
        items = [(r.nfa_utt_id, r.y, r.sr, r.text_norm) for r in self._batch]
        results = run_nfa_align_batch(self.cfg.nfa_gate, items, work_parent=self.work_root)
        for pend, res in zip(self._batch, results, strict=True):
            if not res.ok or not (res.token_string or "").strip():
                self.counts["skipped_nfa_align_failed"] += 1
                continue
            nfa_toks = [t for t in res.token_string.replace("\t", " ").split() if t.strip()]
            if not nfa_toks:
                self.counts["skipped_empty_nfa_tokens"] += 1
                continue
            accumulate_pairs_vote(
                nfa_toks,
                pend.g2p_toks,
                self.strategy,
                self.per_nfa,
                self.method_counts,
                self.counts,
            )
        self._batch.clear()
