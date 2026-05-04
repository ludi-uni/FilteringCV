from __future__ import annotations

from dataclasses import replace

from cv_preprocess.audio.pipeline import run_steps_on_array
from cv_preprocess.audio.quality_gate import sidon_rescue_after_enhance_split_waveform
from cv_preprocess.config import DenoiseStep, PipelineConfig
from cv_preprocess.pipeline.preprocess.types import PendingClip
from cv_preprocess.pipeline.preprocess_efficiency import (
    apply_denoise_steps_only,
    two_pass_uses_split_pipelines,
)


def finalize_two_pass_denoise(p: PendingClip, cfg: PipelineConfig) -> PendingClip:
    if not cfg.two_pass_denoise.enabled:
        return p
    merged: dict[str, object] = dict(p.ameta)
    if two_pass_uses_split_pipelines(cfg):
        assert cfg.audio_pipeline_enhance is not None
        y2, sr2, enh_meta = run_steps_on_array(p.y, p.sr, cfg.audio_pipeline_enhance)
        for k, v in enh_meta.items():
            if k != "steps_trace":
                merged[k] = v
        y2, sr2, sidon_meta = sidon_rescue_after_enhance_split_waveform(
            y2,
            sr2,
            text_len=len(p.text_norm),
            mora_count=p.mora_count,
            gate=cfg.quality_gate,
            snr_cfg=cfg.snr,
        )
        for k, v in sidon_meta.items():
            merged[k] = v
        merged["two_pass_denoise"] = {
            "mode": "split_enhance",
            "steps_trace": enh_meta.get("steps_trace", []),
        }
    else:
        y2, sr2, dmeta = apply_denoise_steps_only(p.y, p.sr, cfg.audio_pipeline.steps)
        merged["two_pass_denoise"] = dmeta
    return replace(p, y=y2, sr=sr2, ameta=merged)


def finalize_two_pass_sgmse_microbatch(
    pendings: list[PendingClip],
    cfg: PipelineConfig,
    step: DenoiseStep,
) -> list[PendingClip]:
    from cv_preprocess.audio.sgmse_dereverb import apply_sgmse_dereverb_batch

    pairs = [(p.y, p.sr) for p in pendings]
    ys = apply_sgmse_dereverb_batch(pairs, step)
    dmeta: dict[str, object] = {
        "denoise_trace": [{"method": "sgmse"}],
        "sgmse_micro_batch": len(pendings),
    }
    return [replace(p, y=y2, ameta={**p.ameta, "two_pass_denoise": dmeta}) for p, y2 in zip(pendings, ys)]
