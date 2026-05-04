"""NARA-WPE（反響軽減）＋ DeepFilterNet（学習済みノイズ抑制）の軽量チェーン。

SGMSE（拡散系）より高速な構成向け。処理順は **WPE → DeepFilterNet**（先に遅延反響を抑え、その後神経ノイズ除去）。

**DeepFilterNet 事前学習モデル**（``DeepFilterNet`` / ``DeepFilterNet2`` / ``DeepFilterNet3``）は
`Rikorose/DeepFilterNet <https://github.com/Rikorose/DeepFilterNet>`_ の ``models/*.zip`` 由来で、
リポジトリ同様 **MIT または Apache-2.0**（デュアルライセンス）の下で利用可能（商用可。利用条件は各ライセンス文面に従うこと）。

**依存**: ``uv sync --extra wpe_dfn``。``deepfilterlib`` は初回に Rust（``cargo``）でビルドされる場合がある。

**torchaudio**: DeepFilterNet 0.5.x は旧 API 前提のため、:mod:`cv_preprocess.audio.deepfilternet_torchaudio_shim` で補う。
"""

from __future__ import annotations

import importlib
import sys
import threading
from typing import Any

import numpy as np
from tqdm import tqdm

from cv_preprocess.audio.deepfilternet_torchaudio_shim import ensure_deepfilternet_torchaudio_shim
from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import AudioPipelineConfig, DenoiseStep

_WPE_DFN_LOCK = threading.Lock()
# (model, df_state, df_sr) — df_sr は ModelParams を毎クリップ構築しないため初回だけ保持
_DF_BUNDLE: dict[tuple[str, str, bool, str], tuple[Any, Any, int]] = {}
_ENHANCE_FN: Any | None = None


def _df_enhance_module() -> Any:
    """``df`` パッケージが ``from .enhance import enhance`` で ``df.enhance`` を関数にしているため、
    ``import df.enhance`` はモジュールではなく関数を返す。``get_device`` パッチには本物のモジュールが必要。
    """
    return importlib.import_module("df.enhance")


def _get_enhance_fn() -> Any:
    global _ENHANCE_FN
    if _ENHANCE_FN is None:
        ensure_deepfilternet_torchaudio_shim()
        _ENHANCE_FN = _df_enhance_module().enhance
    return _ENHANCE_FN


def maybe_warmup_wpe_deepfilternet(audio: AudioPipelineConfig) -> None:
    """パイプラインに ``wpe_deepfilternet`` があるとき、DFN 重みの取得だけ先に済ませる。"""
    for st in audio.steps:
        if isinstance(st, DenoiseStep) and str(st.method).strip().lower() == "wpe_deepfilternet":
            _get_deepfilternet_model_and_state(st)
            return


def _mono_float(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = np.mean(y, axis=0).astype(np.float32)
    elif y.ndim > 2:
        y = np.mean(y.reshape(-1, y.shape[-1]), axis=0).astype(np.float32)
    return y


def _apply_nara_wpe(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int,
    hop_length: int,
    taps: int,
    delay: int,
    iterations: int,
    psd_context: int,
    statistics_mode: str,
) -> np.ndarray:
    import librosa
    from nara_wpe.wpe import wpe_v8

    y = np.asarray(y, dtype=np.float32).copy()
    if y.size < n_fft + hop_length * 4:
        return y
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
        dtype=np.complex64,
    )
    # (F, T) → (F, D=1, T)
    obs = stft[:, np.newaxis, :]
    min_t = int(delay + taps + 3)
    if obs.shape[-1] < min_t:
        return y
    try:
        dereverb = wpe_v8(
            obs,
            taps=int(taps),
            delay=int(delay),
            iterations=int(iterations),
            psd_context=int(psd_context),
            statistics_mode=statistics_mode,
        )
    except (ValueError, np.linalg.LinAlgError):
        return y
    out_stft = dereverb[:, 0, :]
    y_out = librosa.istft(
        out_stft.astype(np.complex64),
        hop_length=hop_length,
        n_fft=n_fft,
        win_length=n_fft,
        window="hann",
        center=True,
        length=y.size,
    )
    if y_out.size != y.size:
        y_out = np.resize(y_out, y.shape)
    return y_out.astype(np.float32)


def _patch_df_get_device(device: Any) -> tuple[Any, Any]:
    df_enhance = _df_enhance_module()
    import df.modules as df_modules

    orig_m = df_modules.get_device
    orig_e = df_enhance.get_device

    def _fixed() -> Any:
        return device

    df_modules.get_device = _fixed  # type: ignore[assignment]
    df_enhance.get_device = _fixed  # type: ignore[assignment]
    return orig_m, orig_e


def _restore_df_get_device(orig_m: Any, orig_e: Any) -> None:
    df_enhance = _df_enhance_module()
    import df.modules as df_modules

    df_modules.get_device = orig_m  # type: ignore[assignment]
    df_enhance.get_device = orig_e  # type: ignore[assignment]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "wpe_deepfilternet needs torch. Install with: uv sync --extra wpe_dfn"
        ) from e
    return torch


def _resolve_torch_device(step: DenoiseStep) -> Any:
    torch = _import_torch()
    d = str(step.deepfilternet_device).strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("deepfilternet_device=cuda but CUDA is not available")
        return torch.device("cuda:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _get_deepfilternet_model_and_state(step: DenoiseStep) -> tuple[Any, Any, int]:
    ensure_deepfilternet_torchaudio_shim()
    init_df = _df_enhance_module().init_df

    model_id = str(step.deepfilternet_model).strip()
    post = bool(step.deepfilternet_post_filter)
    dev = _resolve_torch_device(step)
    dev_key = str(dev)
    key = (model_id, dev_key, post, str(step.deepfilternet_log_level))
    if key in _DF_BUNDLE:
        return _DF_BUNDLE[key]

    with _WPE_DFN_LOCK:
        if key in _DF_BUNDLE:
            return _DF_BUNDLE[key]
        orig_m, orig_e = _patch_df_get_device(dev)
        try:
            tqdm.write(
                "[cv-preprocess] DeepFilterNet: モデル読込（初回は GitHub から zip 取得のため数十秒かかることがあります）…",
                file=sys.stderr,
            )
            model, df_state, _suffix = init_df(
                model_base_dir=model_id,
                post_filter=post,
                log_level=str(step.deepfilternet_log_level).strip(),
                log_file=None,
                config_allow_defaults=True,
                epoch="best",
                mask_only=False,
            )
            # 推論ループでは ``get_device`` の差し替えを毎回しないよう、設定に固定する。
            from df.config import config as df_config

            df_config.set("DEVICE", str(dev), str, "train")
            from df.model import ModelParams

            df_sr = int(ModelParams().sr)
        finally:
            _restore_df_get_device(orig_m, orig_e)
        _DF_BUNDLE[key] = (model, df_state, df_sr)
        return model, df_state, df_sr


def _deepfilternet_enhance(
    y: np.ndarray,
    sr: int,
    step: DenoiseStep,
    *,
    model: Any,
    df_state: Any,
    df_sr: int,
) -> np.ndarray:
    torch = _import_torch()
    enhance = _get_enhance_fn()
    y_in = _mono_float(y)
    y_df = resample_audio(y_in, sr, df_sr) if sr != df_sr else y_in.copy()
    n = int(y_df.shape[0])
    if n < 512:
        return np.asarray(y, dtype=np.float32)

    # ``df.enhance.df_features`` が ``audio.numpy()`` を呼ぶため入力は CPU のまま（特徴は内部で GPU へ）。
    wav = torch.from_numpy(y_df).float().unsqueeze(0)
    enhanced = enhance(
        model,
        df_state,
        wav,
        pad=True,
        atten_lim_db=step.deepfilternet_atten_lim_db,
    )
    out = enhanced.float().squeeze(0).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if out.shape[0] != n:
        if out.shape[0] > n:
            out = out[:n]
        else:
            out = np.pad(out, (0, n - out.shape[0]))
    if sr != df_sr:
        out = resample_audio(out, df_sr, sr)
    return out.astype(np.float32)


def apply_wpe_deepfilternet(y: np.ndarray, sr: int, step: DenoiseStep) -> np.ndarray:
    y0 = np.asarray(y, dtype=np.float32)
    if not step.wpe_deepfilternet_run_wpe:
        y1 = y0
    else:
        y1 = _apply_nara_wpe(
            y0,
            sr,
            n_fft=int(step.wpe_n_fft),
            hop_length=int(step.wpe_hop_length),
            taps=int(step.wpe_taps),
            delay=int(step.wpe_delay),
            iterations=int(step.wpe_iterations),
            psd_context=int(step.wpe_psd_context),
            statistics_mode=str(step.wpe_statistics_mode).strip().lower(),
        )
    model, df_state, df_sr = _get_deepfilternet_model_and_state(step)
    return _deepfilternet_enhance(
        y1, sr, step, model=model, df_state=df_state, df_sr=df_sr
    )
