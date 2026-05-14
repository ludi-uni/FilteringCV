"""ASR ゲート: バッチ推論（mock / NeMo 常駐 subprocess）と CER・PER による足切り。"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config.asr_gate import AsrGateConfig, resolve_asr_python


_asr_worker_lock = threading.Lock()
_asr_worker_proc: subprocess.Popen | None = None
_asr_worker_init_sig: tuple[Any, ...] | None = None
_asr_atexit_registered = False


def _asr_worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "asr_transcribe_worker.py"


def _read_worker_json_ack(stdout: Any, proc: subprocess.Popen, *, timeout_sec: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    out = stdout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ASR worker exited early (code={proc.returncode})")
        line = out.readline()
        if not line:
            raise RuntimeError("ASR worker stdout closed before JSON ack")
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise TimeoutError(f"ASR worker JSON ack timed out after {timeout_sec}s")


def _close_asr_worker_unlocked(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.stdin:
            proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
            proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def close_asr_worker() -> None:
    """常駐 ASR ワーカー subprocess を終了する。"""
    global _asr_worker_proc, _asr_worker_init_sig
    with _asr_worker_lock:
        proc = _asr_worker_proc
        _asr_worker_proc = None
        _asr_worker_init_sig = None
    _close_asr_worker_unlocked(proc)


def _worker_script_digest() -> str:
    p = _asr_worker_script_path()
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:24]
    except OSError:
        return "missing"


def _worker_init_signature(ag: AsrGateConfig) -> tuple[Any, ...]:
    return (
        ag.backend,
        ag.pretrained_name,
        str(ag.model_path.resolve()) if ag.model_path is not None else None,
        bool(ag.use_local_attention),
        _worker_script_digest(),
    )


def _ensure_asr_worker(ag: AsrGateConfig) -> subprocess.Popen:
    global _asr_worker_proc, _asr_worker_init_sig, _asr_atexit_registered
    sig = _worker_init_signature(ag)
    worker_py = _asr_worker_script_path()
    if not worker_py.is_file():
        raise FileNotFoundError(f"ASR persistent worker script not found: {worker_py}")

    py = resolve_asr_python(ag)
    with _asr_worker_lock:
        if _asr_worker_proc is not None:
            if _asr_worker_proc.poll() is not None:
                _close_asr_worker_unlocked(_asr_worker_proc)
                _asr_worker_proc = None
                _asr_worker_init_sig = None
            elif _asr_worker_init_sig != sig:
                _close_asr_worker_unlocked(_asr_worker_proc)
                _asr_worker_proc = None
                _asr_worker_init_sig = None

        if _asr_worker_proc is None:
            proc = subprocess.Popen(
                [py, str(worker_py)],
                cwd=str(tempfile.gettempdir()),
                env=os.environ.copy(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            init_msg: dict[str, Any] = {
                "op": "init",
                "pretrained_name": ag.pretrained_name,
                "model_path": str(ag.model_path.resolve()) if ag.model_path is not None else None,
                "use_local_attention": ag.use_local_attention,
            }
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(init_msg, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            try:
                ack = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(ag.timeout_sec))
            except Exception as e:
                proc.kill()
                raise RuntimeError(f"ASR worker init ack failed: {e}") from e
            if not ack.get("ok"):
                proc.kill()
                raise RuntimeError(f"ASR worker init failed: {ack.get('detail', ack)}")
            _asr_worker_proc = proc
            _asr_worker_init_sig = sig
            if not _asr_atexit_registered:
                atexit.register(close_asr_worker)
                _asr_atexit_registered = True

        return _asr_worker_proc


def _transcribe_nemo_one_shot(ag: AsrGateConfig, paths: list[str]) -> list[str]:
    """常駐を使わず 1 プロセスで init→transcribe→shutdown（テスト・フォールバック用）。"""
    py = resolve_asr_python(ag)
    worker_py = _asr_worker_script_path()
    proc = subprocess.Popen(
        [py, str(worker_py)],
        cwd=str(tempfile.gettempdir()),
        env=os.environ.copy(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None
    init_msg = {
        "op": "init",
        "pretrained_name": ag.pretrained_name,
        "model_path": str(ag.model_path.resolve()) if ag.model_path is not None else None,
        "use_local_attention": ag.use_local_attention,
    }
    proc.stdin.write(json.dumps(init_msg, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    ack = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(ag.timeout_sec))
    if not ack.get("ok"):
        proc.kill()
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"ASR one-shot init failed: {ack.get('detail', ack)} stderr={err!r}")
    proc.stdin.write(json.dumps({"op": "transcribe", "paths": paths, "batch_size": len(paths)}) + "\n")
    proc.stdin.flush()
    out = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(ag.timeout_sec))
    try:
        proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
        proc.stdin.flush()
    except Exception:
        pass
    proc.wait(timeout=60)
    if not out.get("ok"):
        raise RuntimeError(f"ASR transcribe failed: {out.get('detail', out)}")
    texts = out.get("texts")
    if not isinstance(texts, list):
        raise RuntimeError("ASR transcribe response missing texts list")
    return [str(t) if t is not None else "" for t in texts]


def transcribe_batch_paths(ag: AsrGateConfig, paths: list[str]) -> list[str]:
    """絶対パスの WAV 列に対応する仮説テキスト列を返す（長さは paths と一致）。"""
    if not paths:
        return []
    if ag.backend == "mock":
        raise ValueError("transcribe_batch_paths is for nemo_transcribe backend")
    force_sub = os.environ.get("CV_PREPROCESS_ASR_SUBPROCESS", "").strip() in ("1", "true", "yes")
    if ag.persistent_worker and not force_sub:
        proc = _ensure_asr_worker(ag)
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(
            json.dumps({"op": "transcribe", "paths": paths, "batch_size": len(paths)}, ensure_ascii=False)
            + "\n"
        )
        proc.stdin.flush()
        out = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(ag.timeout_sec))
        if not out.get("ok"):
            raise RuntimeError(f"ASR transcribe failed: {out.get('detail', out)}")
        texts = out.get("texts")
        if not isinstance(texts, list) or len(texts) != len(paths):
            raise RuntimeError("ASR transcribe response texts length mismatch")
        return [str(t) if t is not None else "" for t in texts]
    return _transcribe_nemo_one_shot(ag, paths)


def write_temp_wavs_for_asr_batch(
    clips: list[tuple[Any, int]],
    ag: AsrGateConfig,
    work_dir: Path,
) -> list[str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, (y, sr) in enumerate(clips):
        y = np.asarray(y, dtype=np.float32)
        sr_i = int(sr)
        target = int(ag.sample_rate_hz)
        if sr_i != target:
            y = resample_audio(y, sr_i, target)
        wav_path = work_dir / f"{i:05d}.wav"
        sf.write(str(wav_path), y, target, subtype="PCM_16")
        paths.append(str(wav_path.resolve()))
    return paths


def mock_asr_hypothesis(text_norm: str, ag: AsrGateConfig) -> str:
    mode = ag.mock_mode
    if mode == "echo":
        return text_norm
    if mode == "empty":
        return ""
    if mode == "mismatch_char":
        return (text_norm + "＠") if text_norm else "＠"
    return text_norm
