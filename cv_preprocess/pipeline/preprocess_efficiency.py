"""preprocess の効率化: MFA 並列の解決、二段 denoise、早期ゲート用ヘルパー。"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from cv_preprocess.audio.denoise import apply_denoise
from cv_preprocess.config import AudioPipelineConfig, DenoiseStep, MfaGateConfig, PipelineConfig


def resolve_mfa_parallelism(mg: MfaGateConfig) -> tuple[int, int]:
    """``(num_jobs, batch_size)`` の実効値。``auto_*`` は YAML の固定値を上書きする。"""
    nj = int(mg.num_jobs)
    if mg.auto_num_jobs:
        nj = max(1, os.cpu_count() or 1)
    bs = int(mg.batch_size)
    if mg.auto_scale_batch_size:
        bs = max(bs, min(int(mg.batch_size_max), nj * int(mg.auto_batch_jobs_multiplier)))
    return nj, bs


def exclusive_single_sgmse_denoise_for_two_pass_batch(steps: list[Any]) -> DenoiseStep | None:
    """二段 denoise で **SGMSE だけ** が有効ならそのステップを返す（マイクロバッチ適用可）。"""
    active: list[DenoiseStep] = []
    for st in steps:
        if isinstance(st, DenoiseStep):
            m = str(st.method).strip().lower()
            if m in ("none", "", "skip"):
                continue
            active.append(st)
    if len(active) != 1:
        return None
    if str(active[0].method).strip().lower() != "sgmse":
        return None
    return active[0]


def two_pass_uses_split_pipelines(cfg: PipelineConfig) -> bool:
    """``two_pass_denoise`` + ``audio_pipeline_align`` / ``audio_pipeline_enhance`` の完全分離モード。"""
    return bool(
        cfg.two_pass_denoise.enabled
        and cfg.audio_pipeline_align is not None
        and cfg.audio_pipeline_enhance is not None
    )


def effective_audio_catalog_for_preprocess(cfg: PipelineConfig) -> AudioPipelineConfig:
    """メタの ``audio_pipeline_id``・SGMSE ウォームアップ・出力 WAV ビット深度参照に使うカタログ。"""
    if two_pass_uses_split_pipelines(cfg):
        return cfg.audio_pipeline_enhance
    return cfg.audio_pipeline


def resolve_preprocess_pass1_pipeline(cfg: PipelineConfig) -> AudioPipelineConfig:
    """MFA/NFA までに ``run_steps_on_array`` へ渡すパイプライン。"""
    if two_pass_uses_split_pipelines(cfg):
        assert cfg.audio_pipeline_align is not None
        return cfg.audio_pipeline_align
    if cfg.two_pass_denoise.enabled:
        return audio_pipeline_skip_denoise(cfg.audio_pipeline)
    return cfg.audio_pipeline


def audio_pipeline_skip_denoise(cfg: AudioPipelineConfig) -> AudioPipelineConfig:
    """全 ``denoise`` ステップを ``method: none`` にしたパイプライン（第 1 パス用）。"""
    new_steps: list[Any] = []
    for st in cfg.steps:
        if isinstance(st, DenoiseStep):
            new_steps.append(st.model_copy(update={"method": "none"}))
        else:
            new_steps.append(st)
    return cfg.model_copy(update={"steps": new_steps})


def apply_denoise_steps_only(
    y: np.ndarray,
    sr: int,
    steps: list[Any],
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """``steps`` のうち非 none の ``denoise`` だけを順に適用する（第 2 パス用）。"""
    meta: dict[str, Any] = {"denoise_trace": []}
    y_out = np.asarray(y, dtype=np.float32)
    for st in steps:
        if isinstance(st, DenoiseStep):
            m = str(st.method).strip().lower()
            if m in ("none", "", "skip"):
                continue
            y_out = apply_denoise(y_out, sr, st)
            meta["denoise_trace"].append({"method": st.method})
    return y_out, sr, meta
