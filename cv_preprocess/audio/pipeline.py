from __future__ import annotations

from typing import Any

import numpy as np

from cv_preprocess.audio.diff_click_repair import apply_diff_click_repair
from cv_preprocess.audio.denoise import apply_denoise
from cv_preprocess.audio.edge_ui_click import apply_edge_ui_click
from cv_preprocess.audio.filters import butter_highpass, butter_lowpass
from cv_preprocess.audio.lip_noise import apply_lip_noise_suppress
from cv_preprocess.audio.lip_noise_repair import apply_lip_noise_repair
from cv_preprocess.audio.normalize_audio import normalize_loudness, normalize_peak
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.audio.trim_silence import trim_silence
from cv_preprocess.config import (
    AudioPipelineConfig,
    BandwidthExtensionStep,
    DecodeStep,
    DenoiseStep,
    DiffClickRepairStep,
    EdgeUiClickStep,
    HighpassStep,
    LipNoiseRepairStep,
    LipNoiseSuppressStep,
    LowpassStep,
    NormalizeAudioStep,
    ResampleStep,
    SaveWavStep,
    SidonRestoreStep,
    TrimSilenceStep,
)


def _mono_1d_float32(y: np.ndarray) -> np.ndarray:
    """バッチ次元付き (1, T) 等を単一モノラル 1 次元に揃える。``np.pad`` は非 1 次元で軸ごとに幅を複製する。"""
    a = np.asarray(y, dtype=np.float32)
    if a.ndim > 1:
        a = np.mean(a, axis=0)
    return np.ascontiguousarray(a.reshape(-1))


# ``post_spectral: true`` の edge を、これらのブロックの「直後」に遅延実行するための分類。
_SPECTRAL_STEPS_FOR_DEFERRED_EDGE: tuple[type[Any], ...] = (
    DenoiseStep,
    LipNoiseRepairStep,
    LipNoiseSuppressStep,
    BandwidthExtensionStep,
    DiffClickRepairStep,
    SidonRestoreStep,
)


def _is_spectral_step_for_deferred_edge(step: Any) -> bool:
    return isinstance(step, _SPECTRAL_STEPS_FOR_DEFERRED_EDGE)


def _merge_edge_meta_into_pipeline(meta: dict[str, Any], edge_meta: dict[str, Any]) -> None:
    meta["edge_removed_leading_ms"] = float(meta.get("edge_removed_leading_ms", 0.0)) + float(
        edge_meta["edge_removed_leading_ms"]
    )
    meta["edge_removed_trailing_ms"] = float(meta.get("edge_removed_trailing_ms", 0.0)) + float(
        edge_meta["edge_removed_trailing_ms"]
    )
    prev = meta.get("edge_click_confidence")
    conf = edge_meta["edge_click_confidence"]
    if not prev:
        meta["edge_click_confidence"] = conf
    else:
        meta["edge_click_confidence"] = f"{prev}; {conf}"


def apply_edge_ui_click_step(
    y: np.ndarray,
    sr: int,
    step: EdgeUiClickStep,
) -> tuple[np.ndarray, dict[str, Any]]:
    """``EdgeUiClickStep`` 相当の ``apply_edge_ui_click`` を 1 回だけ適用する。

    ``run_steps_on_array`` と同一パラメータで呼び出す。戻り値のメタは
    ``edge_removed_leading_ms`` / ``edge_removed_trailing_ms`` / ``edge_click_confidence``。
    """
    er = apply_edge_ui_click(
        y,
        sr,
        lead_scan_ms=step.lead_scan_ms,
        trail_scan_ms=step.trail_scan_ms,
        max_transient_ms=step.max_transient_ms,
        peak_above_noise_db=step.peak_above_noise_db,
        removal=step.removal,
        trail_click_requires_silence_ms=step.trail_click_requires_silence_ms,
        lead_click_requires_pre_silence_ms=step.lead_click_requires_pre_silence_ms,
        lead_post_click_top_db=step.lead_post_click_top_db,
        diff_mad_k=step.diff_mad_k,
        diff_median_smooth_samples=step.diff_median_smooth_samples,
        short_energy_ms=step.short_energy_ms,
        neighbor_energy_ms=step.neighbor_energy_ms,
        energy_spike_ratio=step.energy_spike_ratio,
        speech_vad_frame_ms=step.speech_vad_frame_ms,
        speech_vad_hop_ms=step.speech_vad_hop_ms,
        speech_rms_mad_k=step.speech_rms_mad_k,
        min_speech_run_ms=step.min_speech_run_ms,
        mute_edge_fade_ms=step.mute_edge_fade_ms,
    )
    meta = {
        "edge_removed_leading_ms": er.removed_leading_ms,
        "edge_removed_trailing_ms": er.removed_trailing_ms,
        "edge_click_confidence": f"L:{er.confidence_leading}/T:{er.confidence_trailing}",
    }
    return er.y, meta


def run_spectral_processing_then_edge_ui_click(
    y: np.ndarray,
    sr: int,
    spectral_cfg: AudioPipelineConfig,
    edge_step: EdgeUiClickStep,
    *,
    omit_edge_ui_clicks_from_spectral: bool = True,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """スペクトル系処理のあとに ``edge_ui_click`` を 1 回だけ適用する（プログラム用）。

    パイプライン YAML では ``EdgeUiClickStep.post_spectral: true`` を使うと
    ``run_steps_on_array`` 内で同等の遅延が行われる。本関数は既存スクリプト向けに残す。

    ``spectral_cfg`` には通常 ``audio_pipeline_enhance`` の前半（denoise / lip 等）を渡す。
    既定では ``spectral_cfg.steps`` 内の ``edge_ui_click`` は実行せず捨て、
    最後に ``edge_step`` だけを適用する。

    Parameters
    ----------
    omit_edge_ui_clicks_from_spectral
        True のとき ``spectral_cfg`` から ``EdgeUiClickStep`` を除いた部分だけを
        ``run_steps_on_array`` に渡す。False のときは設定どおり先に edge も走る。
    """
    if omit_edge_ui_clicks_from_spectral:
        kept = [s for s in spectral_cfg.steps if not isinstance(s, EdgeUiClickStep)]
        spectral_cfg = spectral_cfg.model_copy(update={"steps": kept})
    y, sr, meta = run_steps_on_array(y, sr, spectral_cfg)
    y, edge_meta = apply_edge_ui_click_step(y, sr, edge_step)
    meta["post_spectral_edge_removed_leading_ms"] = edge_meta["edge_removed_leading_ms"]
    meta["post_spectral_edge_removed_trailing_ms"] = edge_meta["edge_removed_trailing_ms"]
    meta["post_spectral_edge_click_confidence"] = edge_meta["edge_click_confidence"]
    trace = list(meta.get("steps_trace", []))
    trace.append({"type": "edge_ui_click", "phase": "post_spectral"})
    meta["steps_trace"] = trace
    return y, sr, meta


def run_steps_on_array(
    y: np.ndarray,
    sr: int,
    cfg: AudioPipelineConfig,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Apply configured steps to in-memory audio (decode/load done outside).

    ``EdgeUiClickStep`` で ``post_spectral: true`` のときは、少なくとも 1 回スペクトル系
    （denoise / lip_noise_repair / lip_noise_suppress / diff_click_repair / bandwidth_extension / sidon_restore）
    を実行したあとで、**次の非スペクトル系ステップの直前**に適用する。
    スペクトル系がまだ無いときはその場で即時に適用する（align 専用パイプライン互換）。
    """
    meta: dict[str, Any] = {"steps_trace": []}
    target_sr = cfg.target_sample_rate
    y = _mono_1d_float32(y)
    pending_post_edge: list[EdgeUiClickStep] = []
    spectral_seen = False

    def flush_pending_post_edge() -> None:
        nonlocal y, sr
        if not pending_post_edge:
            return
        for e in pending_post_edge:
            y, edge_meta = apply_edge_ui_click_step(y, sr, e)
            _merge_edge_meta_into_pipeline(meta, edge_meta)
            meta["steps_trace"].append({"type": "edge_ui_click", "phase": "post_spectral"})
        pending_post_edge.clear()

    def try_flush_pending_before(next_step: Any) -> None:
        nonlocal y, sr
        if not pending_post_edge or not spectral_seen:
            return
        if isinstance(next_step, EdgeUiClickStep) and next_step.post_spectral:
            return
        if _is_spectral_step_for_deferred_edge(next_step):
            return
        flush_pending_post_edge()

    for step in cfg.steps:
        if isinstance(step, DecodeStep):
            continue
        if isinstance(step, SaveWavStep):
            flush_pending_post_edge()
            meta["save_wav"] = {"bit_depth": step.bit_depth}
            continue

        try_flush_pending_before(step)

        if isinstance(step, ResampleStep):
            out_sr = step.sr or target_sr
            y = resample_audio(y, sr, out_sr)
            sr = out_sr
            meta["steps_trace"].append({"type": "resample", "sr": sr})
        elif isinstance(step, EdgeUiClickStep):
            if step.post_spectral and spectral_seen:
                pending_post_edge.append(step)
            else:
                y, edge_meta = apply_edge_ui_click_step(y, sr, step)
                _merge_edge_meta_into_pipeline(meta, edge_meta)
                phase = "post_spectral_inline" if step.post_spectral else "in_place"
                meta["steps_trace"].append({"type": "edge_ui_click", "phase": phase})
        elif isinstance(step, DenoiseStep):
            y = apply_denoise(y, sr, step)
            spectral_seen = True
            meta["steps_trace"].append({"type": "denoise", "method": step.method})
        elif isinstance(step, NormalizeAudioStep):
            if step.method == "loudness":
                y = normalize_loudness(y, sr, integrated_lufs=step.integrated_lufs)
            else:
                y = normalize_peak(y, peak_dbfs=step.peak_dbfs)
            meta["steps_trace"].append({"type": "normalize", "method": step.method})
        elif isinstance(step, TrimSilenceStep):
            trim_track: dict[str, bool] = {}
            y = trim_silence(
                y,
                sr,
                top_db=step.head_tail_db,
                max_keep_sec=step.max_keep_sec,
                frame_length=step.trim_frame_length,
                hop_length=step.trim_hop_length,
                trim_sides=step.trim_sides,
                max_trailing_spike_frames=step.max_trailing_spike_frames,
                track_metadata=trim_track,
            )
            if trim_track.get("exceeded_max_keep_sec") and step.reject_if_truncated:
                meta["trim_exceeds_max_keep_sec"] = True
            ph = int(round(float(step.pad_head_ms) * float(sr) / 1000.0))
            pt = int(round(float(step.pad_tail_ms) * float(sr) / 1000.0))
            if ph > 0 or pt > 0:
                y = np.pad(_mono_1d_float32(y), (ph, pt), mode="constant", constant_values=0.0)
            meta["steps_trace"].append(
                {
                    "type": "trim_silence",
                    "trim_sides": step.trim_sides,
                    "max_trailing_spike_frames": step.max_trailing_spike_frames,
                    "pad_head_ms": step.pad_head_ms,
                    "pad_tail_ms": step.pad_tail_ms,
                    "exceeded_max_keep_sec": bool(trim_track.get("exceeded_max_keep_sec")),
                }
            )
        elif isinstance(step, LowpassStep):
            y = butter_lowpass(y, sr, step.cutoff_hz, order=step.order)
            meta["steps_trace"].append({"type": "lowpass"})
        elif isinstance(step, HighpassStep):
            y = butter_highpass(y, sr, step.cutoff_hz, order=step.order)
            meta["steps_trace"].append({"type": "highpass"})
        elif isinstance(step, LipNoiseRepairStep):
            y, rmeta = apply_lip_noise_repair(
                y,
                sr,
                frame_ms=step.frame_ms,
                hop_ms=step.hop_ms,
                median_kernel_ms=step.median_kernel_ms,
                rms_ratio_threshold=step.rms_ratio_threshold,
                zcr_ratio_threshold=step.zcr_ratio_threshold,
                crest_factor_threshold=step.crest_factor_threshold,
                flux_ratio_threshold=step.flux_ratio_threshold,
                max_event_ms=step.max_event_ms,
                merge_gap_ms=step.merge_gap_ms,
                repair_pad_ms=step.repair_pad_ms,
                max_repair_ms=step.max_repair_ms,
                interpolation=step.interpolation,
                max_repairs_per_clip=step.max_repairs_per_clip,
                fft_bins=step.fft_bins,
            )
            spectral_seen = True
            meta.update(rmeta)
            meta["steps_trace"].append({"type": "lip_noise_repair"})
        elif isinstance(step, DiffClickRepairStep):
            y, rmeta = apply_diff_click_repair(
                y,
                sr,
                mad_k=step.mad_k,
                min_abs_jump=step.min_abs_jump,
                use_second_diff=step.use_second_diff,
                second_diff_mad_k=step.second_diff_mad_k,
                require_both=step.require_both,
                merge_gap_ms=step.merge_gap_ms,
                repair_pad_ms=step.repair_pad_ms,
                max_repair_ms=step.max_repair_ms,
                max_repairs_per_clip=step.max_repairs_per_clip,
                interpolation=step.interpolation,
            )
            spectral_seen = True
            meta.update(rmeta)
            meta["steps_trace"].append({"type": "diff_click_repair"})
        elif isinstance(step, LipNoiseSuppressStep):
            y = apply_lip_noise_suppress(
                y,
                sr,
                n_fft=step.n_fft,
                hop_length=step.hop_length,
                band_low_hz=step.band_low_hz,
                band_high_hz=step.band_high_hz,
                spike_ratio=step.spike_ratio,
                max_burst_frames=step.max_burst_frames,
                mag_gain=step.mag_gain,
                median_kernel_frames=step.median_kernel_frames,
                temporal_smooth_frames=step.temporal_smooth_frames,
            )
            spectral_seen = True
            meta["steps_trace"].append({"type": "lip_noise_suppress"})
        elif isinstance(step, BandwidthExtensionStep):
            from cv_preprocess.audio.hifigan_bwe import apply_hifigan_bwe

            cfg_json = step.config_json or (step.generator_checkpoint.parent / "config.json")
            y, sr = apply_hifigan_bwe(
                y,
                sr,
                config_json=cfg_json,
                generator_checkpoint=step.generator_checkpoint,
                device=step.device,
            )
            spectral_seen = True
            meta["steps_trace"].append(
                {
                    "type": "bandwidth_extension",
                    "config_json": str(cfg_json),
                    "generator_checkpoint": str(step.generator_checkpoint),
                }
            )
        elif isinstance(step, SidonRestoreStep):
            from cv_preprocess.audio.sidon_restore import apply_sidon_restore

            if step.enabled:
                y = apply_sidon_restore(y, sr, step)
            spectral_seen = True
            meta["steps_trace"].append({"type": "sidon_restore", "skipped": not step.enabled})
        else:
            raise TypeError(f"Unsupported step: {type(step)}")
    flush_pending_post_edge()
    return y, sr, meta
