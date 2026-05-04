"""Optional neural denoiser: mispeech/dasheng-denoiser (16 kHz). Install: ``uv sync --extra dasheng``."""

from __future__ import annotations

from typing import Any

import numpy as np

from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import DenoiseStep

_DASHENG_MODELS: dict[tuple[str, str], Any] = {}

DASHENG_SR = 16000


def _import_torch_stack() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModel
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Dasheng denoiser needs torch, torchaudio, transformers, einops. "
            "Install with: uv sync --extra dasheng"
        ) from e
    return torch, AutoModel


def _resolve_device(step: DenoiseStep) -> Any:
    torch, _ = _import_torch_stack()
    d = step.dasheng_device.strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("dasheng_device=cuda but CUDA is not available")
        return torch.device("cuda")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_model(model_id: str, device: Any) -> Any:
    torch, AutoModel = _import_torch_stack()
    key = (model_id, str(device))
    if key not in _DASHENG_MODELS:
        m = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        m = m.to(device)
        m.eval()
        _DASHENG_MODELS[key] = m
    return _DASHENG_MODELS[key]


def apply_dasheng_denoise(y: np.ndarray, sr: int, step: DenoiseStep) -> np.ndarray:
    torch, _ = _import_torch_stack()
    y_in = np.asarray(y, dtype=np.float32)
    if y_in.ndim == 2:
        y_in = np.mean(y_in, axis=0).astype(np.float32)
    elif y_in.ndim > 2:
        y_in = np.mean(y_in.reshape(-1, y_in.shape[-1]), axis=0).astype(np.float32)
    else:
        y_in = y_in.copy()

    y16 = resample_audio(y_in, sr, DASHENG_SR) if sr != DASHENG_SR else y_in.copy()
    n = int(y16.shape[0])
    if n < 256:
        return np.asarray(y, dtype=np.float32)

    device = _resolve_device(step)
    model = _get_model(step.dasheng_model_id.strip(), device)
    wav = torch.from_numpy(y16).to(device).unsqueeze(0)

    with torch.inference_mode():
        if device.type == "cuda" and step.dasheng_cuda_autocast_fp16:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(wav)
        else:
            out = model(wav)

    out_np = out.float().squeeze(0).detach().cpu().numpy().astype(np.float32)
    if out_np.shape[0] != n:
        if out_np.shape[0] > n:
            out_np = out_np[:n]
        else:
            out_np = np.pad(out_np, (0, n - out_np.shape[0]))

    if sr != DASHENG_SR:
        out_np = resample_audio(out_np, DASHENG_SR, sr)
    return out_np.astype(np.float32)
