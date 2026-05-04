"""Optional SGMSE dereverb / enhancement: speechbrain/sgmse-voicebank (16 kHz).

``DenoiseStep.sgmse_return_sample_rate`` を ``16000`` 等にすると、モデル出力をその SR のまま返し
入力 SR への再リサンプルを省略できる（パイプライン後段で ``resample`` して 48 kHz 出力する構成向け）。

**依存**: ``uv sync --extra sgmse`` で torch / torchaudio / speechbrain / torch-ema / ninja と、
``pyproject.toml`` の ``[tool.uv.sources]`` から **GitHub 版 ``sgmse``**（``sp-uhh/sgmse``）を取得する。

**ソース（.cpp / .cu）**はその Git 版に含まれる（PyPI の wheel だけだと欠けることがある）。
JIT ビルドには次が **システム**に必要::

- **g++**（例: ``build-essential``）
- **Python.h**（例: Debian/Ubuntu の ``python3-dev``）
- **nvcc**（CUDA **devel** イメージ / Toolkit。``*-runtime-*`` Docker イメージには無い）
- **ninja**（上記 extra で入る）
"""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from tqdm import tqdm

from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import AudioPipelineConfig, DenoiseStep

_SGMSE_MODELS: dict[tuple[str, str, str], Any] = {}
_SGMSE_LOAD_LOCK = threading.Lock()


@contextmanager
def _suppress_hf_hub_progress_bars() -> Iterator[None]:
    """SpeechBrain の ``from_hparams`` は複数ファイルを順に取得し、Hub の tqdm が乱れて見えることがある。"""
    try:
        from huggingface_hub.utils import (
            are_progress_bars_disabled,
            disable_progress_bars,
            enable_progress_bars,
        )
    except ImportError:
        yield
        return
    if are_progress_bars_disabled():
        yield
        return
    disable_progress_bars()
    try:
        yield
    finally:
        enable_progress_bars()

SGMSE_SR = 16000
# MFA/NFA flush 後にまとめて推論する SGMSE マイクロバッチの最大本数（VRAM・パディング長のトレードオフ）
SGMSE_MICRO_BATCH_MAX = 8


def _sgmse_mono_input(y: np.ndarray) -> np.ndarray:
    y_in = np.asarray(y, dtype=np.float32)
    if y_in.ndim == 2:
        return np.mean(y_in, axis=0).astype(np.float32)
    if y_in.ndim > 2:
        return np.mean(y_in.reshape(-1, y_in.shape[-1]), axis=0).astype(np.float32)
    return y_in.copy()


def maybe_warmup_sgmse(audio: AudioPipelineConfig) -> None:
    """音声パイプラインに SGMSE があるとき、Hub 取得をメイン tqdm の前に済ませる。

    初回取得は数分かかることがあり、その間 ``for row in tqdm(...)`` が進まないため
    「preprocess が止まった」ように見えるのを避ける。
    """
    for st in audio.steps:
        if isinstance(st, DenoiseStep) and str(st.method).strip().lower() == "sgmse":
            _get_enhancer(st)
            return


def _import_torch_stack() -> Any:
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SGMSE needs torch and torchaudio. Install with: uv sync --extra sgmse"
        ) from e
    return torch


def _default_savedir() -> Path:
    return Path.home() / ".cache" / "cv_preprocess" / "sgmse-voicebank"


def _resolve_savedir(step: DenoiseStep) -> Path:
    raw = step.sgmse_savedir
    if raw is None or not str(raw).strip():
        return _default_savedir()
    p = Path(str(raw).strip()).expanduser()
    return p


def _resolve_device(step: DenoiseStep) -> Any:
    torch = _import_torch_stack()
    d = step.sgmse_device.strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("sgmse_device=cuda but CUDA is not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _speechbrain_device_str(device: Any) -> str:
    """SpeechBrain ``run_opts`` は ``cuda`` ではなく ``cuda:0`` 形式を期待する箇所がある。"""
    torch = _import_torch_stack()
    if getattr(device, "type", None) != "cuda":
        return str(device)
    idx = getattr(device, "index", None)
    if idx is None:
        idx = int(torch.cuda.current_device())
    return f"cuda:{idx}"


def _get_enhancer(step: DenoiseStep) -> Any:
    try:
        import sgmse  # noqa: F401  # SpeechBrain SGMSE が参照するパッケージ
    except ModuleNotFoundError as e:
        raise ImportError(
            "``sgmse`` が見つかりません。``uv sync --extra sgmse``（``pyproject.toml`` の Git 参照）を実行してください。"
        ) from e

    try:
        from speechbrain.inference.enhancement import SGMSEEnhancement
    except ModuleNotFoundError as e:
        miss = getattr(e, "name", "") or ""
        if miss == "speechbrain" or str(e).startswith("No module named 'speechbrain'"):
            raise ImportError(
                "SpeechBrain がインストールされていません。SGMSE ステップを使うには次を実行してください: "
                "uv sync --extra sgmse"
            ) from e
        raise

    source = step.sgmse_model_source.strip()
    savedir = str(_resolve_savedir(step))
    device = _resolve_device(step)
    dev_str = _speechbrain_device_str(device)
    key = (source, savedir, dev_str)
    if key in _SGMSE_MODELS:
        return _SGMSE_MODELS[key]

    with _SGMSE_LOAD_LOCK:
        if key in _SGMSE_MODELS:
            return _SGMSE_MODELS[key]
        run_opts = {"device": dev_str}
        try:
            try:
                from huggingface_hub.utils import are_progress_bars_disabled

                _suppress_tqdm = not are_progress_bars_disabled()
            except ImportError:
                _suppress_tqdm = False
            _msg = (
                "[cv-preprocess] SGMSE: Hugging Face Hub から初回取得中（数分かかることがあります"
                + ("；Hub の進捗バーは重複表示を避けるため省略します" if _suppress_tqdm else "")
                + "）…"
            )
            tqdm.write(_msg, file=sys.stderr)
            with _suppress_hf_hub_progress_bars():
                m = SGMSEEnhancement.from_hparams(
                    source=source, savedir=savedir, run_opts=run_opts
                )
        except BaseException as e:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            tb_l = tb.lower()
            # ビルドログにも upfirdn2d.cpp のパスが出るため、「cpp が無い」と誤判定しない順で見る。
            if "c++: not found" in tb or "g++: not found" in tb:
                raise RuntimeError(
                    "sgmse の CUDA 拡張を JIT ビルドするには C++ コンパイラ（g++）が必要です。"
                    "例: Debian/Ubuntu で apt-get install -y build-essential。Dev Container では CUDA **devel** 系イメージの利用を推奨します。"
                ) from e
            if "nvcc: not found" in tb or ("/cuda/bin/nvcc" in tb and "not found" in tb_l):
                raise RuntimeError(
                    "sgmse の .cu をコンパイルするには nvcc（CUDA Toolkit）が PATH 上に必要です。"
                    "`nvidia/cuda:*-runtime-*` イメージには nvcc が含まれません。**devel** イメージに切り替えるか、"
                    "ホストに CUDA Toolkit を入れてください。"
                ) from e
            if "ninja is required" in tb_l or "pip install ninja" in tb_l:
                raise RuntimeError(
                    "CUDA 拡張の JIT ビルドに Ninja が必要です: uv sync --extra sgmse または uv pip install ninja"
                ) from e
            if "python.h" in tb_l and ("fatal error" in tb_l or "no such file" in tb_l):
                raise RuntimeError(
                    "PyTorch / sgmse の C++ 拡張ビルドに Python.h が必要です。"
                    "例: apt-get install -y python3-dev（Debian/Ubuntu）。Dev Container の Dockerfile に含めています。"
                ) from e
            if "error building extension" in tb_l and "upfirdn2d" in tb_l:
                raise RuntimeError(
                    "sgmse の upfirdn2d CUDA 拡張のビルドに失敗しました。"
                    "g++（build-essential）・python3-dev（Python.h）・nvcc（CUDA devel）・ninja を確認し、"
                    "PyTorch の CUDA 版と整合するツールチェーンにしてください。"
                ) from e
            if "no such file or directory" in tb_l and "upfirdn2d.cpp" in tb_l and "failed:" not in tb_l and "error building extension" not in tb_l:
                raise RuntimeError(
                    "sgmse パッケージ内に upfirdn2d.cpp が見つかりません（古い PyPI wheel の残り等）。"
                    "``uv sync --extra sgmse`` で Git 版に入れ直すか、``uv lock`` 更新後に再同期してください。"
                ) from e
            raise
        m.eval()
        _SGMSE_MODELS[key] = m
        return _SGMSE_MODELS[key]


def _enhance_batch_sgmse(
    enhancer: Any,
    wav: Any,
    *,
    step: DenoiseStep,
    device: Any,
    torch: Any,
) -> Any:
    """``enhance_batch`` を実行。``autocast(fp16)`` が SpeechBrain/sgmse 内部で dtype 不整合を起こす環境では float32 にフォールバック。"""
    with torch.inference_mode():
        if device.type != "cuda" or not step.sgmse_cuda_autocast_fp16:
            return enhancer.enhance_batch(wav)
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return enhancer.enhance_batch(wav)
        except RuntimeError as e:
            msg = str(e).lower()
            if "half" in msg and "float" in msg:
                return enhancer.enhance_batch(wav)
            raise


def apply_sgmse_dereverb(y: np.ndarray, sr: int, step: DenoiseStep) -> np.ndarray:
    torch = _import_torch_stack()
    y_in = _sgmse_mono_input(y)

    y16 = resample_audio(y_in, sr, SGMSE_SR) if sr != SGMSE_SR else y_in.copy()
    n = int(y16.shape[0])
    if n < 512:
        return np.asarray(y, dtype=np.float32)

    device = _resolve_device(step)
    enhancer = _get_enhancer(step)
    wav = torch.from_numpy(y16).float().to(device).unsqueeze(0)

    out = _enhance_batch_sgmse(enhancer, wav, step=step, device=device, torch=torch)

    out_np = out.float().squeeze(0).detach().cpu().numpy().astype(np.float32)
    if out_np.shape[0] != n:
        if out_np.shape[0] > n:
            out_np = out_np[:n]
        else:
            out_np = np.pad(out_np, (0, n - out_np.shape[0]))

    out_sr = int(step.sgmse_return_sample_rate) if step.sgmse_return_sample_rate is not None else int(sr)
    if out_sr != SGMSE_SR:
        out_np = resample_audio(out_np, SGMSE_SR, out_sr)
    return out_np.astype(np.float32)


def apply_sgmse_dereverb_batch(
    items: list[tuple[np.ndarray, int]],
    step: DenoiseStep,
) -> list[np.ndarray]:
    """複数クリップを 1 回の ``enhance_batch`` に載せる（同一 ``DenoiseStep``・可変長はゼロパディング）。"""
    if not items:
        return []
    if len(items) == 1:
        y0, sr0 = items[0]
        return [apply_sgmse_dereverb(y0, sr0, step)]

    out_list: list[np.ndarray | None] = [None] * len(items)
    work_idx: list[int] = []
    work_sr: list[int] = []
    work_y16: list[np.ndarray] = []
    work_len: list[int] = []

    for i, (y, sr) in enumerate(items):
        y_in = _sgmse_mono_input(y)
        y16 = resample_audio(y_in, sr, SGMSE_SR) if sr != SGMSE_SR else y_in.copy()
        n = int(y16.shape[0])
        if n < 512:
            out_list[i] = np.asarray(y, dtype=np.float32)
            continue
        work_idx.append(i)
        work_sr.append(int(sr))
        work_y16.append(y16)
        work_len.append(n)

    if not work_idx:
        return [out_list[i] for i in range(len(items))]  # type: ignore[misc]

    out_sr_override = (
        int(step.sgmse_return_sample_rate) if step.sgmse_return_sample_rate is not None else None
    )

    torch = _import_torch_stack()
    device = _resolve_device(step)
    enhancer = _get_enhancer(step)
    t_max = max(work_len)
    bsz = len(work_idx)
    batch_np = np.zeros((bsz, t_max), dtype=np.float32)
    for b, y16 in enumerate(work_y16):
        ln = work_len[b]
        batch_np[b, :ln] = y16
    wav = torch.from_numpy(batch_np).float().to(device)

    out = _enhance_batch_sgmse(enhancer, wav, step=step, device=device, torch=torch)

    out_np = out.float().detach().cpu().numpy().astype(np.float32)
    for b, list_i in enumerate(work_idx):
        n = work_len[b]
        sr_i = work_sr[b]
        out_sr = out_sr_override if out_sr_override is not None else sr_i
        seg = np.asarray(out_np[b, :n], dtype=np.float32)
        if seg.shape[0] != n:
            if seg.shape[0] > n:
                seg = seg[:n]
            else:
                seg = np.pad(seg, (0, n - seg.shape[0]))
        if out_sr != SGMSE_SR:
            seg = resample_audio(seg, SGMSE_SR, out_sr)
        out_list[list_i] = seg.astype(np.float32)

    if any(x is None for x in out_list):
        raise RuntimeError("apply_sgmse_dereverb_batch: internal slot unfilled")
    return out_list  # type: ignore[return-value]
