# SPDX-License-Identifier: MIT
# HiFi-GAN Generator / メル処理は jik876/hifi-gan（MIT）の構成に準拠。
# https://github.com/jik876/hifi-gan
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# --- mel + 前処理（meldataset.py 相当） ---------------------------------
def _dynamic_range_compression_torch(x: Any, C: float = 1.0, clip_val: float = 1e-5) -> Any:
    import torch

    return torch.log(torch.clamp(x, min=clip_val) * C)


def _spectral_normalize_torch(magnitudes: Any) -> Any:
    return _dynamic_range_compression_torch(magnitudes)


_mel_basis: dict[str, Any] = {}
_hann_window: dict[str, Any] = {}


def mel_spectrogram_hifigan(
    y: Any,
    *,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float,
    center: bool = False,
) -> Any:
    import librosa
    import torch
    import torch.nn.functional as F

    if torch.min(y) < -1.0:
        pass
    if torch.max(y) > 1.0:
        pass

    global _mel_basis, _hann_window
    key_basis = f"{fmax}_{n_fft}_{num_mels}_{fmin}_{sampling_rate}_{y.device}"
    if key_basis not in _mel_basis:
        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        _mel_basis[key_basis] = torch.from_numpy(mel).float().to(y.device)
    win_key = f"{win_size}_{y.device}"
    if win_key not in _hann_window:
        _hann_window[win_key] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=_hann_window[win_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1e-9)
    spec = torch.matmul(_mel_basis[key_basis], spec)
    spec = _spectral_normalize_torch(spec)
    return spec


# --- Generator（models.py の Generator / ResBlock のみ） -----------------
LRELU_SLOPE = 0.1


def _init_weights(m: Any, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class _ResBlock1(nn.Module):  # noqa: N801
    def __init__(self, h: Any, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3, 5)):
        from torch.nn import Conv1d
        from torch.nn.utils import weight_norm

        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=_get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(3)
            ]
        )
        self.convs1.apply(_init_weights)
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=_get_padding(kernel_size, 1),
                    )
                )
                for _ in range(3)
            ]
        )
        self.convs2.apply(_init_weights)

    def forward(self, x: Any) -> Any:
        import torch.nn.functional as F

        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        from torch.nn.utils import remove_weight_norm

        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class _ResBlock2(nn.Module):  # noqa: N801
    def __init__(self, h: Any, channels: int, kernel_size: int = 3, dilation: tuple[int, ...] = (1, 3)):
        from torch.nn import Conv1d
        from torch.nn.utils import weight_norm

        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[i],
                        padding=_get_padding(kernel_size, dilation[i]),
                    )
                )
                for i in range(2)
            ]
        )
        self.convs.apply(_init_weights)

    def forward(self, x: Any) -> Any:
        import torch.nn.functional as F

        for conv in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        from torch.nn.utils import remove_weight_norm

        for layer in self.convs:
            remove_weight_norm(layer)


class _HifiGenerator(nn.Module):  # noqa: N801
    def __init__(self, h: Any):
        from torch.nn import Conv1d, ConvTranspose1d
        from torch.nn.utils import weight_norm

        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(int(h.num_mels), h.upsample_initial_channel, 7, 1, padding=3))
        resblock = _ResBlock1 if h.resblock == "1" else _ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, x: Any) -> Any:
        import torch.nn.functional as F

        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        from torch.nn.utils import remove_weight_norm

        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def _hparams_from_json(path: Path) -> SimpleNamespace:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("HiFi-GAN config must be a JSON object")
    return SimpleNamespace(**{k: v for k, v in raw.items() if not isinstance(v, dict)})


@dataclass
class HifiGanEngine:
    h: SimpleNamespace
    generator: Any
    device: Any

    @classmethod
    def load(
        cls,
        *,
        config_json: Path,
        generator_checkpoint: Path,
        device_s: str,
    ) -> HifiGanEngine:
        import torch

        if device_s == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_s)

        h = _hparams_from_json(config_json)
        gen = _HifiGenerator(h).to(device)
        try:
            ckpt = torch.load(generator_checkpoint, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(generator_checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "generator" in ckpt:
            state = ckpt["generator"]
        else:
            state = ckpt
        if isinstance(state, dict) and any(str(k).startswith("module.") for k in state):
            state = {str(k).replace("module.", "", 1): v for k, v in state.items()}
        gen.load_state_dict(state, strict=True)
        gen.eval()
        gen.remove_weight_norm()
        return cls(h=h, generator=gen, device=device)

    def infer_waveform(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
        import torch

        from cv_preprocess.audio.resample import resample_audio

        target_sr = int(self.h.sampling_rate)
        y = np.asarray(y, dtype=np.float32)
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        if sr != target_sr:
            y = resample_audio(y, sr, target_sr)
        wav = torch.from_numpy(y).float().to(self.device)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = torch.clamp(wav, -1.0, 1.0)
        mel = mel_spectrogram_hifigan(
            wav,
            n_fft=int(self.h.n_fft),
            num_mels=int(self.h.num_mels),
            sampling_rate=int(self.h.sampling_rate),
            hop_size=int(self.h.hop_size),
            win_size=int(self.h.win_size),
            fmin=float(self.h.fmin),
            fmax=float(self.h.fmax),
            center=False,
        )
        with torch.no_grad():
            y_hat = self.generator(mel)
        # 次元 1 をすべて落とし 1 次元にする。``(1, T)`` のままだと後段 ``np.pad`` が各軸にパッドを乗せて破綻する。
        out = np.asarray(y_hat.detach().cpu().float().squeeze().numpy(), dtype=np.float32).reshape(-1)
        if not np.isfinite(out).all():
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        peak = float(np.max(np.abs(out)) + 1e-12)
        if peak > 1.0:
            out = (out / peak * 0.99).astype(np.float32)
        return out, target_sr


_ENGINE_CACHE: dict[tuple[str, str, str], HifiGanEngine] = {}


def get_hifigan_engine(
    *,
    config_json: Path,
    generator_checkpoint: Path,
    device: str,
) -> HifiGanEngine:
    key = (str(config_json.resolve()), str(generator_checkpoint.resolve()), device)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = HifiGanEngine.load(
            config_json=config_json,
            generator_checkpoint=generator_checkpoint,
            device_s=device,
        )
    return _ENGINE_CACHE[key]


def apply_hifigan_bwe(
    y: np.ndarray,
    sr: int,
    *,
    config_json: Path,
    generator_checkpoint: Path,
    device: str,
) -> tuple[np.ndarray, int]:
    eng = get_hifigan_engine(
        config_json=config_json,
        generator_checkpoint=generator_checkpoint,
        device=device,
    )
    return eng.infer_waveform(y, sr)
