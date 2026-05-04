from __future__ import annotations

from typing import Literal

import numpy as np

TrimSides = Literal["both", "leading", "trailing"]


def _last_nonsilent_frame_skip_terminal_spikes(non: np.ndarray, max_spike_frames: int) -> int:
    """``non[i]`` が非無音。末尾から見て、直前が無音に挟まれた短い True 連なりを切り捨てた最後のフレーム。"""
    non = np.asarray(non, dtype=bool).reshape(-1)
    if non.size == 0:
        return -1
    i = int(non.shape[0]) - 1
    while i >= 0:
        if not non[i]:
            i -= 1
            continue
        burst_end = i
        burst_start = i
        while burst_start - 1 >= 0 and non[burst_start - 1]:
            burst_start -= 1
        burst_len = burst_end - burst_start + 1
        left = burst_start - 1
        if (
            max_spike_frames > 0
            and burst_len <= max_spike_frames
            and left >= 0
            and not non[left]
        ):
            i = left - 1
            continue
        return burst_end
    return -1


def _end_sample_trailing_spike_aware(
    y: np.ndarray,
    *,
    top_db: float,
    frame_length: int,
    hop_length: int,
    max_trailing_spike_frames: int,
) -> int:
    """librosa trim と同じ RMS→dB 判定で、末尾の短いスパイク島を無視した排他 end サンプル。"""
    import librosa
    from librosa import core

    y = np.asarray(y, dtype=np.float32)
    fl = int(frame_length)
    hl = int(hop_length)
    mse = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)
    db = np.squeeze(np.asarray(librosa.amplitude_to_db(mse[0], ref=np.max, top_db=None), dtype=np.float64))
    non = (db > -float(top_db)).astype(bool)
    last_f = _last_nonsilent_frame_skip_terminal_spikes(non, max_trailing_spike_frames)
    if last_f < 0:
        return 0
    return int(min(y.shape[-1], int(core.frames_to_samples(last_f + 1, hop_length=hl))))


def trim_silence(
    y: np.ndarray,
    sr: int,
    *,
    top_db: float = 40.0,
    max_keep_sec: float | None = None,
    frame_length: int = 2048,
    hop_length: int = 512,
    trim_sides: TrimSides = "both",
    max_trailing_spike_frames: int = 0,
) -> np.ndarray:
    """``librosa.effects.trim`` 相当。``trim_sides`` で先頭・末尾のどちらを削るか選べる。

    ``trailing`` は **右端の無音相当だけ** を落とし、先頭はそのまま（帯域補完後の尾部ノイズ対策向け）。
    ``max_trailing_spike_frames`` は ``trim_sides=trailing`` のとき、末尾から数えてこの長さ以下の
    非無音島（その直前が無音）をスパイクとして無視し、それより左を保持する。

    NaN/Inf が 1 サンプルでも混ざると ``librosa.effects.trim`` が ``[0, 0]`` を返し空配列になり、
    後段のパッドだけが残って無音 WAV になるため、trim 前に有限値へ正規化し、空結果はフォールバックする。
    """
    import librosa

    y_work = np.asarray(y, dtype=np.float32).copy()
    if y_work.size == 0:
        return y_work
    if not np.isfinite(y_work).all():
        np.nan_to_num(y_work, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    fl = int(frame_length)
    hl = int(hop_length)
    _, idx = librosa.effects.trim(y_work, top_db=top_db, frame_length=fl, hop_length=hl)
    start, end = int(idx[0]), int(idx[1])
    if start >= end:
        y_t = y_work
    elif trim_sides == "both":
        y_t = y_work[start:end]
    elif trim_sides == "leading":
        y_t = y_work[start:]
    else:
        end_lib = int(idx[1])
        if max_trailing_spike_frames > 0:
            end_spike = _end_sample_trailing_spike_aware(
                y_work,
                top_db=top_db,
                frame_length=fl,
                hop_length=hl,
                max_trailing_spike_frames=max_trailing_spike_frames,
            )
            # spike 探索が非無音島を見つけられないと 0 になり y[:0] が空になる（実データでは BWE 後の
            # 弱い末尾などで再現しうる）。その場合は librosa の右境界にフォールバックする。
            if end_spike <= 0:
                end = end_lib
            else:
                end = min(end_lib, end_spike)
        else:
            end = end_lib
        y_t = y_work[:end]

    if y_t.size == 0 and y_work.size > 0:
        y_t = y_work
    elif y_work.size > 0:
        peak_in = float(np.max(np.abs(y_work)))
        peak_out = float(np.max(np.abs(y_t)))
        # trim 後に信号が完全に消えたのに入力に有意な振幅があった → 誤 trim とみなして入力を返す
        if peak_in > 1e-6 and peak_out < 1e-12:
            y_t = y_work

    if max_keep_sec is not None:
        max_n = int(sr * max_keep_sec)
        if y_t.size > max_n:
            y_t = y_t[:max_n]
    return y_t.astype(np.float32)
