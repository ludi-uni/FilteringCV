from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from cv_preprocess.audio.quality_gate import (
    GateResult,
    quality_gate_configs_equivalent,
    quality_gate_run_fingerprint,
)
from cv_preprocess.config import PipelineConfig, QualityGateConfig


def _merged_quality_gate_for_aligner_prefilter(
    cfg: PipelineConfig, overrides: dict[str, object]
) -> QualityGateConfig:
    d = cfg.quality_gate.model_dump()
    d.update(overrides)
    return QualityGateConfig.model_validate(d)


def _merged_quality_gate_for_mfa_prefilter(cfg: PipelineConfig) -> QualityGateConfig:
    return _merged_quality_gate_for_aligner_prefilter(
        cfg, cfg.mfa_gate.prefilter.quality_gate_overrides
    )


def _merged_quality_gate_for_nfa_prefilter(cfg: PipelineConfig) -> QualityGateConfig:
    return _merged_quality_gate_for_aligner_prefilter(
        cfg, cfg.nfa_gate.prefilter.quality_gate_overrides
    )


def _resolve_quality_gate_profile_dict(
    cfg: PipelineConfig, profile_name: str, *, context: str
) -> dict[str, Any]:
    profiles = cfg.quality_gate_profiles or {}
    if profile_name not in profiles:
        raise ValueError(
            f"{context} {profile_name!r} not found in quality_gate_profiles "
            f"(keys: {list(profiles)})"
        )
    base_prof = profiles[profile_name]
    if not isinstance(base_prof, dict):
        raise ValueError(f"quality_gate_profiles[{profile_name!r}] must be a mapping")
    return dict(base_prof)


def effective_final_quality_gate(cfg: PipelineConfig) -> QualityGateConfig:
    """accept 直前の ``run_quality_gate`` に使う閾値。``nfa_gate.enabled`` 時は ``nfa_gate.quality_gate_*`` をマージ。"""
    if not cfg.nfa_gate.enabled:
        return cfg.quality_gate
    ng = cfg.nfa_gate
    if not ng.quality_gate_overrides and not ng.quality_gate_profile:
        return cfg.quality_gate
    d = cfg.quality_gate.model_dump()
    prof = ng.quality_gate_profile
    if prof:
        d = {**_resolve_quality_gate_profile_dict(cfg, prof, context="nfa_gate.quality_gate_profile"), **d}
    d.update(ng.quality_gate_overrides)
    return QualityGateConfig.model_validate(d)


def _mora_gates_needed(
    lang: str,
    cfg: PipelineConfig,
    align_prefilter_qg: QualityGateConfig | None,
) -> tuple[bool, bool, bool]:
    """
    各段でモーラ数が参照されるか。
    戻り値は (early_audio_gate, align_prefilter, final_quality_gate)。失敗時の reject 理由が異なる。
    """
    ja = lang.split("-")[0] == "ja"
    early = (
        ja
        and cfg.early_audio_gate.enabled
        and cfg.early_audio_gate.check_mora_duration
        and cfg.quality_gate.min_sec_per_mora is not None
    )
    pref = ja and align_prefilter_qg is not None and align_prefilter_qg.min_sec_per_mora is not None
    final_qg = effective_final_quality_gate(cfg)
    final = ja and final_qg.min_sec_per_mora is not None
    return early, pref, final


def _compute_clip_mora_count_once(
    text_norm: str,
    *,
    need_early: bool,
    need_pref: bool,
    need_final: bool,
    prefilter_mora_fail_reason: str | None = None,
) -> tuple[int | None, str | None]:
    """
    OpenJTalk ベースのモーラ数を高々 1 回だけ算出する。
    戻り値 ``(mora_count, reject_reason)``。``reject_reason`` が非 None なら当該クリップを拒否する。
    """
    if not (need_early or need_pref or need_final):
        return None, None
    try:
        from cv_preprocess.text.mora_estimate import mora_count_for_text

        return mora_count_for_text(text_norm), None
    except Exception:
        if need_pref:
            return None, prefilter_mora_fail_reason or "mfa_prefilter_mora_estimate_failed"
        if need_final:
            return None, "mora_estimate_failed"
        return None, None


def infer_release(root: Path) -> str:
    if re.fullmatch(r"[a-z]{2}(-[A-Za-z]+)?", root.name):
        return root.parent.name
    return root.name


def _maybe_prefilter_final_gate_reuse_pair(
    gate_pf: GateResult,
    prefilter_qg: QualityGateConfig,
    cfg: PipelineConfig,
    y: np.ndarray,
    sr: int,
    text_len: int,
    mora_pf: int | None,
    mora_fin: bool,
    clip_mora_count: int | None,
) -> tuple[GateResult, str] | None:
    """
    Aligner prefilter の ``run_quality_gate`` を本番と同一条件で再利用できるとき ``(result, fingerprint)`` を返す。
    ゲートやモーラ参照が本番と異なる場合は ``None``（呼び出し側は直前の候補を維持してよい）。
    """
    mora_n_final = clip_mora_count if mora_fin else None
    if mora_pf != mora_n_final:
        return None
    final_qg = effective_final_quality_gate(cfg)
    if not quality_gate_configs_equivalent(prefilter_qg, final_qg):
        return None
    fp = quality_gate_run_fingerprint(
        y,
        sr,
        text_len,
        gate=final_qg,
        snr_cfg=cfg.snr,
        mora_count=mora_pf,
    )
    return (gate_pf, fp)
