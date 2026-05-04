"""Sidon 音声復元（split enhance 後の救済用）。

公式リポジトリ: https://github.com/sarulab-speech/Sidon （MIT）
推論用 TorchScript: https://huggingface.co/sarulab-speech/sidon-v0.1

``uv sync --extra sidon`` で torch / torchaudio / transformers / huggingface_hub を入れた環境でのみ import 可能。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config.gates_quality import QualityGateSidonAfterEnhanceSplitConfig

_bundle_lock = threading.Lock()
_bundle: dict[str, Any] | None = None


def _resolve_torch_device(cfg: QualityGateSidonAfterEnhanceSplitConfig) -> str:
    d = str(cfg.device).strip().lower()
    if d == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return d


def _ensure_sidon_bundle(cfg: QualityGateSidonAfterEnhanceSplitConfig) -> dict[str, Any]:
    global _bundle
    with _bundle_lock:
        if _bundle is not None:
            return _bundle
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import SeamlessM4TFeatureExtractor

        device_str = _resolve_torch_device(cfg)
        use_cuda = device_str == "cuda"
        fe_name = cfg.feature_extractor_filename_cuda if use_cuda else cfg.feature_extractor_filename_cpu
        dec_name = cfg.decoder_filename_cuda if use_cuda else cfg.decoder_filename_cpu
        fe_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=fe_name)
        dec_path = hf_hub_download(repo_id=cfg.hf_repo_id, filename=dec_name)
        preprocessor = SeamlessM4TFeatureExtractor.from_pretrained(cfg.ssl_model_id)
        dev = torch.device(device_str)
        fe = torch.jit.load(fe_path, map_location=dev).to(dev).eval()
        decoder = torch.jit.load(dec_path, map_location=dev).to(dev).eval()
        _bundle = {
            "device_str": device_str,
            "device": dev,
            "preprocessor": preprocessor,
            "fe": fe,
            "decoder": decoder,
        }
        return _bundle


def apply_sidon_restore(
    y: np.ndarray,
    sr: int,
    cfg: QualityGateSidonAfterEnhanceSplitConfig,
) -> np.ndarray:
    """単声道 float 波形を Sidon で復元し、入力 ``sr`` にリサンプルして返す。"""
    import torch
    import torchaudio

    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size < 512:
        return y.astype(np.float32, copy=False)

    b = _ensure_sidon_bundle(cfg)
    device: torch.device = b["device"]
    preprocessor = b["preprocessor"]
    fe = b["fe"]
    decoder = b["decoder"]

    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        return y.astype(np.float32, copy=False)
    wav = torch.from_numpy(y).float().view(1, -1)
    wav = 0.9 * (wav / peak)
    target_n_samples = int(48_000 / float(sr) * float(wav.shape[1]))
    wav = torchaudio.functional.highpass_biquad(wav, sr, 50.0)
    wav_16k = torchaudio.functional.resample(wav, sr, 16_000)
    wav_16k = torch.nn.functional.pad(wav_16k, (0, 24_000))

    restored_parts: list[torch.Tensor] = []
    feature_cache: torch.Tensor | None = None
    chunk_samples = 16_000 * 96
    for chunk in wav_16k.view(-1).split(chunk_samples):
        padded = torch.nn.functional.pad(chunk, (160, 160))
        inputs = preprocessor(padded.cpu().numpy(), sampling_rate=16_000, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        with torch.inference_mode():
            out = fe(input_features)
            feature = out["last_hidden_state"] if isinstance(out, dict) else out
            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
            dec_in = feature.transpose(1, 2)
            restored = decoder(dec_in).view(-1)[:-960]
            restored_parts.append(restored)
            feature_cache = feature[:, -1:, :]

    if not restored_parts:
        return y.astype(np.float32, copy=False)
    restored_wav = torch.cat(restored_parts, dim=0)
    n_take = min(int(restored_wav.shape[0]), max(target_n_samples, 1))
    restored_wav = restored_wav[:n_take].detach().float().cpu().numpy().astype(np.float32)
    if not np.isfinite(restored_wav).all():
        restored_wav = np.nan_to_num(restored_wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if sr != 48_000:
        restored_wav = resample_audio(restored_wav, 48_000, sr)
    n_orig = int(y.shape[0])
    if restored_wav.shape[0] > n_orig:
        restored_wav = restored_wav[:n_orig]
    elif restored_wav.shape[0] < n_orig:
        restored_wav = np.pad(restored_wav, (0, n_orig - restored_wav.shape[0])).astype(np.float32)
    return restored_wav.astype(np.float32, copy=False)
