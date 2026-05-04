from __future__ import annotations

import numpy as np

from cv_preprocess.config import DenoiseStep


def _ms_to_stft_frames(ms: float, sr: int, hop_length: int) -> int:
    if hop_length <= 0 or sr <= 0:
        return 0
    return max(0, int((ms / 1000.0) * sr / hop_length))


def _spectral_subtract(
    y: np.ndarray,
    sr: int,
    cfg: DenoiseStep,
) -> np.ndarray:
    import librosa

    y = np.asarray(y, dtype=np.float32).copy()
    n_fft = int(cfg.n_fft)
    hop = int(cfg.hop_length)
    if y.size < hop + 16:
        return y.astype(np.float32)

    D = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window="hann",
        center=True,
        dtype=np.complex64,
    )
    mag = np.abs(D).astype(np.float64)
    phase = np.angle(D)
    n_bins, n_frames = mag.shape
    if n_frames < 2:
        return y.astype(np.float32)

    lead = min(_ms_to_stft_frames(cfg.noise_lead_ms, sr, hop), n_frames)
    trail = min(_ms_to_stft_frames(cfg.noise_trail_ms, sr, hop), n_frames)
    idx: list[int] = []
    idx.extend(range(lead))
    idx.extend(range(max(0, n_frames - trail), n_frames))

    frame_rms = np.sqrt(np.mean(mag**2, axis=0) + 1e-20)
    thr = float(np.percentile(frame_rms, cfg.noise_quiet_rms_percentile))
    quiet = np.flatnonzero(frame_rms <= thr).tolist()
    idx.extend(quiet)
    idx = sorted(set(idx))[: max(1, int(cfg.max_noise_frames))]

    if not idx:
        idx = list(range(min(2, n_frames)))

    noise_mag = np.mean(mag[:, idx], axis=1, keepdims=True) + 1e-10

    # 末尾帯の「低エネルギー」STFT フレームから反響・ルームトーン寄りのスペクトルを推定し、
    # ノイズ見積りを引き上げて減算を強める（完全なディリバーブではないが主観的には効きやすい）。
    mix = float(cfg.reverb_tail_mix)
    if mix > 0.0 and cfg.reverb_tail_ms > 0.0:
        tail_f = _ms_to_stft_frames(cfg.reverb_tail_ms, sr, hop)
        tail_f = max(2, min(tail_f, n_frames))
        tail_region = mag[:, n_frames - tail_f : n_frames]
        frame_e = np.sqrt(np.mean(tail_region**2, axis=0) + 1e-20)
        thr_t = float(np.percentile(frame_e, cfg.reverb_tail_frame_percentile))
        sel = frame_e <= thr_t
        if np.any(sel):
            tail_est = np.mean(tail_region[:, sel], axis=1, keepdims=True) + 1e-10
            noise_mag = np.maximum(noise_mag, mix * tail_est)

    mag_sq = mag**2
    noise_sq = noise_mag**2
    mag_out_sq = np.maximum(mag_sq - float(cfg.subtract_alpha) * noise_sq, 0.0)
    mag_out = np.sqrt(mag_out_sq)
    mag_out = np.maximum(mag_out, float(cfg.spectral_floor) * mag)

    D_out = (mag_out * np.exp(1j * phase)).astype(np.complex64)
    y_out = librosa.istft(
        D_out,
        hop_length=hop,
        n_fft=n_fft,
        win_length=n_fft,
        window="hann",
        center=True,
        length=y.size,
    )
    if y_out.size != y.size:
        y_out = np.resize(y_out, y.shape)
    return y_out.astype(np.float32)


def apply_denoise(y: np.ndarray, sr: int, step: DenoiseStep) -> np.ndarray:
    m = step.method.strip().lower()
    if m in ("none", "", "skip"):
        return np.asarray(y, dtype=np.float32)
    if m == "spectral_subtract":
        return _spectral_subtract(y, sr, step)
    if m == "dasheng":
        from cv_preprocess.audio.dasheng_denoise import apply_dasheng_denoise

        return apply_dasheng_denoise(y, sr, step)
    if m == "sgmse":
        from cv_preprocess.audio.sgmse_dereverb import apply_sgmse_dereverb

        return apply_sgmse_dereverb(y, sr, step)
    if m == "wpe_deepfilternet":
        from cv_preprocess.audio.wpe_deepfilternet_denoise import apply_wpe_deepfilternet

        return apply_wpe_deepfilternet(y, sr, step)
    raise ValueError(f"Unknown denoise method: {step.method!r}")
