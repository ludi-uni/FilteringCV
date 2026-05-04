from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cv_preprocess.audio.snr import estimate_snr_db
from cv_preprocess.config import EarlyAudioGateConfig, QualityGateConfig, SnrEstimatorConfig

QualityTier = Literal["A", "B", "C"]


def _as_mono_1d(y: np.ndarray) -> np.ndarray:
    """品質ゲート用に (T,) の float32 へ。``(C,T)`` / ``(T,C)`` はチャネル平均、それ以上は末尾軸を時間とみなして平均。"""
    a = np.asarray(y, dtype=np.float32)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        if a.shape[0] <= a.shape[1]:
            return np.mean(a, axis=0).astype(np.float32)
        return np.mean(a, axis=1).astype(np.float32)
    return np.mean(a.reshape(-1, a.shape[-1]), axis=0).astype(np.float32)


def quality_gate_configs_equivalent(a: QualityGateConfig, b: QualityGateConfig) -> bool:
    """``quality_gate`` と aligner prefilter 用マージ後ゲートが同一閾値か（再利用可否の前提）。"""
    return a.model_dump(mode="json") == b.model_dump(mode="json")


def quality_gate_run_fingerprint(
    y: np.ndarray,
    sr: int,
    text_len: int,
    *,
    gate: QualityGateConfig,
    snr_cfg: SnrEstimatorConfig,
    mora_count: int | None,
) -> str:
    """
    同一波形・同一ゲート・同一 SNR 設定・同一テキスト長・同一モーラ数での ``run_quality_gate`` 短絡用キー。
    ``two_pass_denoise`` 後など波形が変わった場合は別フィンガープリントになる。
    """
    y_a = np.asarray(y, dtype=np.float32).ravel(order="C")
    h = hashlib.sha256()
    h.update(y_a.tobytes())
    h.update(int(sr).to_bytes(4, "little", signed=True))
    h.update(int(text_len).to_bytes(4, "little", signed=False))
    if mora_count is None:
        h.update(b"\xff\xff\xff\xff")
    else:
        h.update(int(mora_count).to_bytes(4, "little", signed=True))
    cfg_blob = json.dumps(
        {"g": gate.model_dump(mode="json"), "s": snr_cfg.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    h.update(cfg_blob)
    return h.hexdigest()


@dataclass
class GateResult:
    ok: bool
    reason: str | None
    duration_sec: float
    silence_ratio: float
    estimated_snr_db: float | None
    clipping_ratio: float
    dc_offset: float
    #: ``silence_ratio`` と同じフレーム／閾値で、**末尾から連続**する無音の長さ（秒）。
    trailing_silence_sec: float = 0.0
    mora_count: int | None = None
    min_required_duration_sec: float | None = None
    #: ``quality_tier_mode`` が ``annotate`` / ティア reject のときのみ。``off`` では常に ``None``。
    quality_tier: QualityTier | None = None
    quality_score: float | None = None


def _quality_score(
    snr_val: float | None,
    silence_ratio: float,
    clipping_ratio: float,
    gate: QualityGateConfig,
) -> float:
    """0〜100 の粗い総合指標（SNR・無音率・クリッピングを合成）。同一ティア内の並べ替え用。"""
    if snr_val is not None:
        sn_part = 50.0 * max(0.0, min(1.0, (float(snr_val) - 3.0) / 22.0))
    else:
        sn_part = 22.0
    span = max(float(gate.max_silence_ratio), 1e-6)
    sil_part = 35.0 * max(0.0, min(1.0, (span - float(silence_ratio)) / span))
    cspan = max(float(gate.max_clipping_ratio), 1e-9)
    cl_part = 15.0 * max(0.0, min(1.0, (cspan - float(clipping_ratio)) / cspan))
    return float(round(sn_part + sil_part + cl_part, 2))


def _assign_quality_tier(
    snr_val: float | None,
    silence_ratio: float,
    clipping_ratio: float,
    gate: QualityGateConfig,
) -> tuple[QualityTier, float]:
    score = _quality_score(snr_val, silence_ratio, clipping_ratio, gate)
    if snr_val is not None:
        if (
            float(snr_val) >= float(gate.quality_tier_a_min_snr_db)
            and float(silence_ratio) <= float(gate.quality_tier_a_max_silence_ratio)
            and float(clipping_ratio) <= float(gate.quality_tier_a_max_clipping_ratio)
        ):
            return "A", score
        if float(snr_val) >= float(gate.quality_tier_b_min_snr_db) and float(
            silence_ratio
        ) <= float(gate.quality_tier_b_max_silence_ratio):
            return "B", score
        return "C", score

    if float(clipping_ratio) <= float(gate.quality_tier_a_max_clipping_ratio) and float(
        silence_ratio
    ) <= float(gate.quality_tier_unknown_snr_silence_a):
        return "A", score
    if float(silence_ratio) <= float(gate.quality_tier_unknown_snr_silence_b):
        return "B", score
    return "C", score


def _frame_rms_peak_series(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
) -> np.ndarray:
    """短時間フレームの RMS 列（``silence_ratio_frames`` / ``measure_trailing_silence_sec`` と同一窓・同一 hop）。

    従来どおり ``range(0, y.size - frame, hop)`` と同一の窓集合（**最後の完全窓は含めない**）。
    """
    y = _as_mono_1d(y)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    lim = int(y.size) - frame
    if lim <= 0:
        return np.array([], dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    vw = sliding_window_view(y64, frame)[:lim:hop, :]
    return np.sqrt(np.mean(vw * vw, axis=1) + 1e-18).astype(np.float64)


def _silence_ratio_from_rms_peaks(
    p: np.ndarray,
    *,
    rms_floor: float,
    ref_percentile: float,
) -> float:
    if p.size == 0:
        return 0.0
    if ref_percentile >= 100.0:
        ref = float(np.max(p) + 1e-12)
    else:
        ref = float(np.percentile(p, ref_percentile) + 1e-12)
    thr = float(rms_floor) * ref
    return float(np.mean(p < thr))


def _trailing_silence_sec_from_rms_peaks(
    p: np.ndarray,
    sr: int,
    hop_ms: float,
    *,
    rms_floor: float,
    ref_percentile: float,
) -> float:
    if p.size == 0:
        return 0.0
    hop = max(1, int(sr * hop_ms / 1000.0))
    if ref_percentile >= 100.0:
        ref = float(np.max(p) + 1e-12)
    else:
        ref = float(np.percentile(p, ref_percentile) + 1e-12)
    thr = float(rms_floor) * ref
    silent = p < thr
    n = int(silent.size)
    if n == 0 or not bool(silent[-1]):
        return 0.0
    rev = silent[::-1]
    inv = ~rev
    trailing_frames = int(rev.size if not np.any(inv) else np.argmax(inv))
    return float(trailing_frames * hop / float(sr))


def _silence_ratio_and_trailing_sec(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float,
    hop_ms: float,
    rms_floor: float,
    ref_percentile: float,
) -> tuple[float, float]:
    """
    無音率と末尾無音を RMS フレーム列 1 本で算出する（従来 2 関数と同じ定義）。
    ``measure_trailing_silence_sec`` は ``y.size < frame + hop`` のとき 0 を返す挙動を維持する。
    """
    y = _as_mono_1d(y)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    p = _frame_rms_peak_series(y, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    sil = _silence_ratio_from_rms_peaks(p, rms_floor=rms_floor, ref_percentile=ref_percentile)
    if y.size < frame + hop:
        return sil, 0.0
    trail = _trailing_silence_sec_from_rms_peaks(
        p, sr, hop_ms, rms_floor=rms_floor, ref_percentile=ref_percentile
    )
    return sil, trail


def measure_trailing_silence_sec(
    y: np.ndarray,
    sr: int,
    *,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    rms_floor: float = 0.05,
    ref_percentile: float = 100.0,
) -> float:
    """末尾から連続して無音と判定される区間の長さ（秒）。``silence_ratio_frames`` と同一閾値。"""
    y = _as_mono_1d(y)
    frame = max(1, int(sr * frame_ms / 1000.0))
    hop = max(1, int(sr * hop_ms / 1000.0))
    if y.size < frame + hop:
        return 0.0
    p = _frame_rms_peak_series(y, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return _trailing_silence_sec_from_rms_peaks(
        p, sr, hop_ms, rms_floor=rms_floor, ref_percentile=ref_percentile
    )


def _tier_cap_for_trailing(
    tier: QualityTier,
    score: float,
    snr_val: float | None,
    silence_ratio: float,
    clipping_ratio: float,
    trailing_sec: float,
    gate: QualityGateConfig,
) -> tuple[QualityTier, float]:
    cap = gate.quality_tier_a_max_trailing_silence_sec
    if cap is None or tier != "A" or trailing_sec <= float(cap) + 1e-9:
        return tier, score
    if snr_val is not None:
        if float(snr_val) >= float(gate.quality_tier_b_min_snr_db) and float(
            silence_ratio
        ) <= float(gate.quality_tier_b_max_silence_ratio):
            return "B", score
        return "C", score
    if float(silence_ratio) <= float(gate.quality_tier_unknown_snr_silence_b):
        return "B", score
    return "C", score


def silence_ratio_frames(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    *,
    rms_floor: float = 0.05,
    ref_percentile: float = 100.0,
) -> float:
    p = _frame_rms_peak_series(y, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return _silence_ratio_from_rms_peaks(p, rms_floor=rms_floor, ref_percentile=ref_percentile)


def run_early_audio_gate(
    y: np.ndarray,
    sr: int,
    *,
    text_len: int,
    mora_count: int | None,
    main_gate: QualityGateConfig,
    snr_cfg: SnrEstimatorConfig,
    early: EarlyAudioGateConfig,
) -> GateResult:
    """``main_gate`` / ``snr_cfg`` の閾値で、有効化した指標だけを検査する（ティア付与は行わない）。"""
    y = _as_mono_1d(y)
    duration_sec = float(y.size) / float(sr) if sr > 0 else 0.0
    silence_ratio, trailing_silence_sec = _silence_ratio_and_trailing_sec(
        y,
        sr,
        frame_ms=snr_cfg.frame_ms,
        hop_ms=snr_cfg.hop_ms,
        rms_floor=main_gate.silence_ratio_rms_floor,
        ref_percentile=main_gate.silence_ratio_ref_percentile,
    )
    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    dc_offset = float(abs(float(np.mean(y))))
    snr_val = estimate_snr_db(
        y,
        sr,
        frame_ms=snr_cfg.frame_ms,
        hop_ms=snr_cfg.hop_ms,
        noise_percentile=snr_cfg.noise_percentile,
        signal_percentile=snr_cfg.signal_percentile,
        min_frames=snr_cfg.min_frames,
    )

    def fail(reason: str) -> GateResult:
        return GateResult(
            False,
            reason,
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
            mora_count=mora_count,
        )

    if early.check_duration:
        if duration_sec < main_gate.min_duration_sec or duration_sec > main_gate.max_duration_sec:
            return fail("early_gate_duration")
    if early.check_silence_ratio:
        if silence_ratio > main_gate.max_silence_ratio:
            return fail("early_gate_silence")
    if early.check_snr and main_gate.min_estimated_snr_db is not None:
        # 推定不能（短尺・エネルギー分布が平坦等）は閾値比較しない（quality_gate のコメントと同趣旨）
        if snr_val is not None and snr_val < main_gate.min_estimated_snr_db:
            return fail("early_gate_snr")
    if early.check_clipping:
        if clipping_ratio > main_gate.max_clipping_ratio:
            return fail("early_gate_clipping")
    if early.check_dc_offset:
        if dc_offset > main_gate.max_abs_dc_offset:
            return fail("early_gate_dc")
    if early.check_chars_per_sec:
        cps = text_len / duration_sec if duration_sec > 0 else 0.0
        if main_gate.min_chars_per_sec is not None and cps < main_gate.min_chars_per_sec:
            return fail("early_gate_text_audio_low")
        if main_gate.max_chars_per_sec is not None and cps > main_gate.max_chars_per_sec:
            return fail("early_gate_text_audio_high")
    min_req_out: float | None = None
    if early.check_mora_duration:
        if (
            main_gate.min_sec_per_mora is not None
            and mora_count is not None
            and mora_count > 0
        ):
            min_req_out = mora_count * float(main_gate.min_sec_per_mora) * float(main_gate.mora_gate_relax)
            if duration_sec + 1e-9 < min_req_out:
                return fail("early_gate_text_audio_mora")

    return GateResult(
        True,
        None,
        duration_sec,
        silence_ratio,
        snr_val,
        clipping_ratio,
        dc_offset,
        trailing_silence_sec=trailing_silence_sec,
        mora_count=mora_count,
        min_required_duration_sec=min_req_out,
    )


def run_quality_gate(
    y: np.ndarray,
    sr: int,
    *,
    text_len: int,
    gate: QualityGateConfig,
    snr_cfg: SnrEstimatorConfig,
    mora_count: int | None = None,
) -> GateResult:
    y = _as_mono_1d(y)
    duration_sec = float(y.size) / float(sr) if sr > 0 else 0.0
    silence_ratio, trailing_silence_sec = _silence_ratio_and_trailing_sec(
        y,
        sr,
        frame_ms=snr_cfg.frame_ms,
        hop_ms=snr_cfg.hop_ms,
        rms_floor=gate.silence_ratio_rms_floor,
        ref_percentile=gate.silence_ratio_ref_percentile,
    )
    clipping_ratio = float(np.mean(np.abs(y) >= 0.999))
    dc_offset = float(abs(float(np.mean(y))))

    snr_val = estimate_snr_db(
        y,
        sr,
        frame_ms=snr_cfg.frame_ms,
        hop_ms=snr_cfg.hop_ms,
        noise_percentile=snr_cfg.noise_percentile,
        signal_percentile=snr_cfg.signal_percentile,
        min_frames=snr_cfg.min_frames,
    )

    if duration_sec < gate.min_duration_sec or duration_sec > gate.max_duration_sec:
        return GateResult(
            False,
            "gate_duration",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )
    if silence_ratio > gate.max_silence_ratio:
        return GateResult(
            False,
            "gate_silence",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )
    if gate.max_trailing_silence_sec is not None:
        if trailing_silence_sec > float(gate.max_trailing_silence_sec) + 1e-9:
            return GateResult(
                False,
                "gate_trailing_silence",
                duration_sec,
                silence_ratio,
                snr_val,
                clipping_ratio,
                dc_offset,
                trailing_silence_sec=trailing_silence_sec,
            )
    if gate.min_estimated_snr_db is not None:
        if snr_val is not None and snr_val < gate.min_estimated_snr_db:
            return GateResult(
                False,
                "gate_snr",
                duration_sec,
                silence_ratio,
                snr_val,
                clipping_ratio,
                dc_offset,
                trailing_silence_sec=trailing_silence_sec,
            )
    if clipping_ratio > gate.max_clipping_ratio:
        return GateResult(
            False,
            "gate_clipping",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )
    if dc_offset > gate.max_abs_dc_offset:
        return GateResult(
            False,
            "gate_dc",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )
    cps = text_len / duration_sec if duration_sec > 0 else 0.0
    if gate.min_chars_per_sec is not None and cps < gate.min_chars_per_sec:
        return GateResult(
            False,
            "gate_text_audio_low",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )
    if gate.max_chars_per_sec is not None and cps > gate.max_chars_per_sec:
        return GateResult(
            False,
            "gate_text_audio_high",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
        )

    min_req_out: float | None = None
    if (
        gate.min_sec_per_mora is not None
        and mora_count is not None
        and mora_count > 0
    ):
        min_req_out = mora_count * float(gate.min_sec_per_mora) * float(gate.mora_gate_relax)
        if duration_sec + 1e-9 < min_req_out:
            return GateResult(
                False,
                "gate_text_audio_mora",
                duration_sec,
                silence_ratio,
                snr_val,
                clipping_ratio,
                dc_offset,
                trailing_silence_sec=trailing_silence_sec,
                mora_count=mora_count,
                min_required_duration_sec=min_req_out,
            )

    mode = str(gate.quality_tier_mode).strip().lower()
    if mode == "off":
        return GateResult(
            True,
            None,
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
            mora_count=mora_count,
            min_required_duration_sec=min_req_out,
        )

    tier, score = _assign_quality_tier(snr_val, silence_ratio, clipping_ratio, gate)
    tier, score = _tier_cap_for_trailing(
        tier,
        score,
        snr_val,
        silence_ratio,
        clipping_ratio,
        trailing_silence_sec,
        gate,
    )

    if mode == "annotate":
        return GateResult(
            True,
            None,
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
            mora_count=mora_count,
            min_required_duration_sec=min_req_out,
            quality_tier=tier,
            quality_score=score,
        )
    if mode == "reject_c" and tier == "C":
        return GateResult(
            False,
            "gate_quality_tier_c",
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
            mora_count=mora_count,
            min_required_duration_sec=min_req_out,
            quality_tier=tier,
            quality_score=score,
        )
    if mode == "reject_b" and tier in ("B", "C"):
        reason = "gate_quality_tier_b" if tier == "B" else "gate_quality_tier_c"
        return GateResult(
            False,
            reason,
            duration_sec,
            silence_ratio,
            snr_val,
            clipping_ratio,
            dc_offset,
            trailing_silence_sec=trailing_silence_sec,
            mora_count=mora_count,
            min_required_duration_sec=min_req_out,
            quality_tier=tier,
            quality_score=score,
        )

    return GateResult(
        True,
        None,
        duration_sec,
        silence_ratio,
        snr_val,
        clipping_ratio,
        dc_offset,
        trailing_silence_sec=trailing_silence_sec,
        mora_count=mora_count,
        min_required_duration_sec=min_req_out,
        quality_tier=tier,
        quality_score=score,
    )


def sidon_rescue_after_enhance_split_waveform(
    y: np.ndarray,
    sr: int,
    *,
    text_len: int,
    mora_count: int | None,
    gate: QualityGateConfig,
    snr_cfg: SnrEstimatorConfig,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """split ``audio_pipeline_enhance`` 直後: ``run_quality_gate`` と同一基準で ``ok`` ならそのまま。
    ``ok=False`` かつ ``gate.sidon_after_enhance_split.enabled`` のとき Sidon を適用し、**再度**
    ``run_quality_gate`` を実行してメタに記録する。
    """
    sidon_cfg = gate.sidon_after_enhance_split
    meta: dict[str, Any] = {"sidon_after_enhance_split": {"enabled": bool(sidon_cfg.enabled)}}
    if not sidon_cfg.enabled:
        return y, sr, meta

    g_pre = run_quality_gate(
        y,
        sr,
        text_len=text_len,
        gate=gate,
        snr_cfg=snr_cfg,
        mora_count=mora_count,
    )
    block = meta["sidon_after_enhance_split"]
    block["pre_gate_ok"] = g_pre.ok
    block["pre_gate_reason"] = g_pre.reason
    block["pre_quality_tier"] = g_pre.quality_tier
    block["pre_estimated_snr_db"] = g_pre.estimated_snr_db
    if g_pre.ok:
        return y, sr, meta

    try:
        from cv_preprocess.audio.sidon_restore import apply_sidon_restore

        y_sidon = apply_sidon_restore(y, sr, sidon_cfg)
        block["applied"] = True
    except Exception as e:
        block["applied"] = False
        block["error"] = f"{type(e).__name__}: {e}"
        return y, sr, meta

    g_post = run_quality_gate(
        y_sidon,
        sr,
        text_len=text_len,
        gate=gate,
        snr_cfg=snr_cfg,
        mora_count=mora_count,
    )
    block["post_gate_ok"] = g_post.ok
    block["post_gate_reason"] = g_post.reason
    block["post_quality_tier"] = g_post.quality_tier
    block["post_estimated_snr_db"] = g_post.estimated_snr_db
    return y_sidon, sr, meta
