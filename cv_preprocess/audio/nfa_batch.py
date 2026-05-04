"""NeMo Forced Aligner（``tools/nemo_forced_aligner/align.py``）のバッチ呼び出し。

既定では **モデル常駐ワーカー subprocess**（`nfa_align_worker.py`）を 1 プロセス立て、
各バッチは stdin 経由でジョブ化する。従来の **毎バッチ ``align.py`` 起動**は
``nfa_gate.persistent_worker: false`` または環境変数 ``CV_PREPROCESS_NFA_SUBPROCESS=1`` で選択可能。
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from cv_preprocess.audio.resample import resample_audio
from cv_preprocess.config import NfaGateConfig


@dataclass
class NfaClipResult:
    ok: bool
    token_string: str | None
    detail: str | None = None
    #: ``align_using_pred_text`` 時に NeMo 出力マニフェストから取った認識文（照合用）
    pred_text: str | None = None


def _safe_manifest_text(text_norm: str, *, strip_spaces: bool) -> str:
    s = text_norm.replace("\r", " ").replace("\n", " ").strip()
    if strip_spaces:
        s = re.sub(r"\s+", "", s)
    return s


def _resolve_nfa_python(cfg: NfaGateConfig) -> str:
    if cfg.nfa_python:
        return cfg.nfa_python
    return os.environ.get("NFA_PYTHON", "python3")


def _resolve_nfa_align_dir(cfg: NfaGateConfig) -> Path | None:
    if cfg.nfa_align_dir is not None:
        return cfg.nfa_align_dir
    env = os.environ.get("NFA_ALIGN_DIR")
    return Path(env) if env else None


def _nfa_worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "nfa_align_worker.py"


_nfa_worker_lock = threading.Lock()
_nfa_worker_proc: subprocess.Popen | None = None
_nfa_worker_init_sig: tuple[Any, ...] | None = None
_nfa_atexit_registered = False


def _close_nfa_worker_unlocked(proc: subprocess.Popen | None) -> None:
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


def close_nfa_worker() -> None:
    """常駐 NFA ワーカー subprocess を終了する（preprocess 終了時・atexit 用）。"""
    global _nfa_worker_proc, _nfa_worker_init_sig
    with _nfa_worker_lock:
        proc = _nfa_worker_proc
        _nfa_worker_proc = None
        _nfa_worker_init_sig = None
    _close_nfa_worker_unlocked(proc)


def _worker_init_signature(cfg: NfaGateConfig) -> tuple[Any, ...]:
    return (
        cfg.pretrained_name,
        str(cfg.model_path.resolve()) if cfg.model_path is not None else None,
        bool(cfg.use_local_attention),
        bool(cfg.align_using_pred_text),
        tuple(cfg.extra_align_args),
    )


def _read_worker_json_ack(stdout: Any, proc: subprocess.Popen, *, timeout_sec: float) -> dict[str, Any]:
    """NeMo 等が stdout にログを混ぜるため、先頭が ``{`` の 1 行 JSON まで読み飛ばす。"""
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    out = stdout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"NFA worker exited early (code={proc.returncode})")
        line = out.readline()
        if not line:
            raise RuntimeError("NFA worker stdout closed before JSON ack")
        s = line.strip()
        if not s or not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise TimeoutError(f"NFA worker JSON ack timed out after {timeout_sec}s")


def _parse_bool_extra(extra: list[str], key: str, default: bool) -> bool:
    prefix = f"{key}="
    for s in extra:
        if s.startswith(prefix):
            v = s[len(prefix) :].strip().lower()
            return v in ("true", "1", "yes")
    return default


def _ensure_nfa_worker(cfg: NfaGateConfig, align_dir: Path, nfa_py: str) -> subprocess.Popen:
    """モデル読み込み済みのワーカーを返す（同一シグネチャなら再利用）。"""
    global _nfa_worker_proc, _nfa_worker_init_sig, _nfa_atexit_registered
    sig = _worker_init_signature(cfg)
    worker_py = _nfa_worker_script_path()
    if not worker_py.is_file():
        raise FileNotFoundError(f"NFA persistent worker script not found: {worker_py}")

    with _nfa_worker_lock:
        if _nfa_worker_proc is not None:
            if _nfa_worker_proc.poll() is not None:
                _close_nfa_worker_unlocked(_nfa_worker_proc)
                _nfa_worker_proc = None
                _nfa_worker_init_sig = None
            elif _nfa_worker_init_sig != sig:
                _close_nfa_worker_unlocked(_nfa_worker_proc)
                _nfa_worker_proc = None
                _nfa_worker_init_sig = None

        if _nfa_worker_proc is None:
            env = os.environ.copy()
            env["CV_NFA_ALIGN_DIR"] = str(align_dir.resolve())
            proc = subprocess.Popen(
                [nfa_py, str(worker_py)],
                cwd=str(align_dir.resolve()),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            init_msg: dict[str, Any] = {
                "op": "init",
                "pretrained_name": cfg.pretrained_name,
                "model_path": str(cfg.model_path.resolve()) if cfg.model_path is not None else None,
                "use_local_attention": cfg.use_local_attention,
                "use_buffered_chunked_streaming": _parse_bool_extra(
                    cfg.extra_align_args, "use_buffered_chunked_streaming", False
                ),
                "align_using_pred_text": bool(cfg.align_using_pred_text)
                or _parse_bool_extra(cfg.extra_align_args, "align_using_pred_text", False),
                "simulate_cache_aware_streaming": _parse_bool_extra(
                    cfg.extra_align_args, "simulate_cache_aware_streaming", False
                ),
                "chunk_batch_size": int(cfg.worker_chunk_batch_size),
                "save_output_file_formats": ["ctm"],
            }
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(init_msg, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            try:
                ack = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(cfg.timeout_sec))
            except Exception as e:
                proc.kill()
                raise RuntimeError(f"NFA worker init ack failed: {e}") from e
            if not ack.get("ok"):
                proc.kill()
                raise RuntimeError(f"NFA worker init failed: {ack.get('detail', ack)}")
            _nfa_worker_proc = proc
            _nfa_worker_init_sig = sig
            if not _nfa_atexit_registered:
                atexit.register(close_nfa_worker)
                _nfa_atexit_registered = True

        return _nfa_worker_proc


def _attach_pred_texts_if_needed(
    cfg: NfaGateConfig,
    align_out: Path,
    manifest_path: Path,
    results: list[NfaClipResult],
) -> None:
    if not cfg.align_using_pred_text:
        return
    out_m = align_out / f"{manifest_path.stem}_with_output_file_paths.json"
    if not out_m.is_file():
        return
    preds: list[str | None] = []
    for line in out_m.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = o.get("pred_text")
        preds.append(None if v is None else str(v).strip())
    n = min(len(results), len(preds))
    if len(preds) != len(results):
        print(
            f"[cv-preprocess] NFA pred_text 行数 ({len(preds)}) がバッチ件数 ({len(results)}) と一致しません。"
            f" pred_text の付与を先頭 {n} 件に限定します。",
            file=sys.stderr,
            flush=True,
        )
    for i in range(n):
        if results[i].ok:
            results[i].pred_text = preds[i]


def parse_token_ctm_file(path: Path) -> list[str]:
    """
    NeMo NFA が書く token-level CTM（1 行: utt channel start dur token conf type speaker）から
    トークン列を順に取り出す。空トークン・空白のみは除外。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    tokens: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        tok = parts[4]
        if not tok or tok.isspace():
            continue
        tokens.append(tok)
    return tokens


def tokens_to_space_string(tokens: list[str]) -> str:
    return " ".join(tokens)


def _read_ctm_results(align_out: Path, items: list[tuple[str, np.ndarray, int, str]]) -> list[NfaClipResult]:
    ctm_root = align_out / "ctm" / "tokens"
    results: list[NfaClipResult] = []
    for utt_id, _, _, _ in items:
        ctm = ctm_root / f"{utt_id}.ctm"
        if not ctm.is_file():
            results.append(NfaClipResult(ok=False, token_string=None, detail="token_ctm_missing"))
            continue
        try:
            toks = parse_token_ctm_file(ctm)
            results.append(NfaClipResult(ok=True, token_string=tokens_to_space_string(toks) if toks else ""))
        except OSError as e:
            results.append(NfaClipResult(ok=False, token_string=None, detail=f"ctm_read:{e}"))
    return results


def _run_nfa_align_subprocess(
    cfg: NfaGateConfig,
    items: list[tuple[str, np.ndarray, int, str]],
    *,
    align_dir: Path,
    nfa_py: str,
    corpus_dir: Path,
    align_out: Path,
    manifest_path: Path,
) -> list[NfaClipResult]:
    hydra_args: list[str] = [
        f"manifest_filepath={manifest_path.resolve()}",
        f"output_dir={align_out.resolve()}",
        f"batch_size={int(cfg.batch_size)}",
        f"use_local_attention={'true' if cfg.use_local_attention else 'false'}",
        f"align_using_pred_text={'true' if cfg.align_using_pred_text else 'false'}",
    ]
    if cfg.pretrained_name:
        hydra_args.append(f"pretrained_name={cfg.pretrained_name}")
    if cfg.model_path is not None:
        hydra_args.append(f"model_path={cfg.model_path.resolve()}")
    hydra_args.extend(cfg.extra_align_args)

    cmd = [nfa_py, str(align_dir / "align.py"), *hydra_args]
    proc = subprocess.run(
        cmd,
        cwd=str(align_dir),
        capture_output=True,
        text=True,
        timeout=float(cfg.timeout_sec),
        check=False,
        env=os.environ.copy(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "")[:4000]
        return [NfaClipResult(ok=False, token_string=None, detail=f"nfa_exit_{proc.returncode}: {err}") for _ in items]
    return _read_ctm_results(align_out, items)


def _run_nfa_align_persistent_worker(
    cfg: NfaGateConfig,
    items: list[tuple[str, np.ndarray, int, str]],
    *,
    align_dir: Path,
    nfa_py: str,
    corpus_dir: Path,
    align_out: Path,
    manifest_path: Path,
) -> list[NfaClipResult]:
    proc = _ensure_nfa_worker(cfg, align_dir, nfa_py)
    job = {
        "op": "align",
        "manifest_filepath": str(manifest_path.resolve()),
        "output_dir": str(align_out.resolve()),
        "batch_size": int(cfg.batch_size),
    }
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(job, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    try:
        ack = _read_worker_json_ack(proc.stdout, proc, timeout_sec=float(cfg.timeout_sec))
    except Exception as e:
        close_nfa_worker()
        return [NfaClipResult(ok=False, token_string=None, detail=f"nfa_worker_align_ack:{e}")] * len(items)
    if not ack.get("ok"):
        detail = str(ack.get("detail", ack))[:4000]
        return [NfaClipResult(ok=False, token_string=None, detail=f"nfa_worker_align:{detail}")] * len(items)
    return _read_ctm_results(align_out, items)


def run_nfa_align_batch(
    cfg: NfaGateConfig,
    items: list[tuple[str, np.ndarray, int, str]],
    *,
    work_parent: Path,
) -> list[NfaClipResult]:
    """
    ``items``: (utterance_id, waveform, sample_rate, text_norm)

    ``persistent_worker``（既定 ``true``）のときは ``nfa_align_worker.py`` を 1 プロセス常駐させ、
    各バッチは JSON 1 行でワーカーに渡す（モデルは初回のみロード）。
    それ以外は従来どおり ``align.py`` をバッチごとに ``subprocess.run`` する。
    """
    if not items:
        return []

    align_dir = _resolve_nfa_align_dir(cfg)
    if align_dir is None or not align_dir.is_dir():
        msg = "nfa_align_dir が未設定、または NFA_ALIGN_DIR 環境変数が無効です"
        return [NfaClipResult(ok=False, token_string=None, detail=msg) for _ in items]

    align_py = align_dir / "align.py"
    if not align_py.is_file():
        msg = f"align.py が見つかりません: {align_py}"
        return [NfaClipResult(ok=False, token_string=None, detail=msg) for _ in items]

    nfa_py = _resolve_nfa_python(cfg)
    batch_root = cfg.work_dir if cfg.work_dir is not None else work_parent
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex[:12]
    corpus_dir = batch_root / f"nfa_corpus_{batch_id}"
    align_out = batch_root / f"nfa_aligned_{batch_id}"
    manifest_path = corpus_dir / "manifest.jsonl"

    force_subprocess = os.environ.get("CV_PREPROCESS_NFA_SUBPROCESS", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    use_worker = bool(cfg.persistent_worker) and not force_subprocess

    try:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        tgt_sr = int(cfg.model_sample_rate_hz)

        manifest_lines: list[dict[str, str]] = []
        for utt_id, y, sr, text_norm in items:
            wav_path = corpus_dir / f"{utt_id}.wav"
            y_f = np.asarray(y, dtype=np.float32).reshape(-1)
            if int(sr) != tgt_sr:
                y_f = resample_audio(y_f, int(sr), tgt_sr)
            y_i16 = np.clip(y_f, -1.0, 1.0) * 32767.0
            sf.write(str(wav_path), y_i16.astype(np.int16), tgt_sr, subtype="PCM_16")
            text_m = _safe_manifest_text(text_norm, strip_spaces=cfg.manifest_strip_spaces)
            manifest_lines.append(
                {
                    "audio_filepath": str(wav_path.resolve()),
                    "text": text_m,
                }
            )

        manifest_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in manifest_lines) + "\n",
            encoding="utf-8",
        )

        if use_worker:
            try:
                results = _run_nfa_align_persistent_worker(
                    cfg,
                    items,
                    align_dir=align_dir,
                    nfa_py=nfa_py,
                    corpus_dir=corpus_dir,
                    align_out=align_out,
                    manifest_path=manifest_path,
                )
            except Exception as e:
                err_one = f"{type(e).__name__}: {e}".replace("\n", " ")[:800]
                print(
                    f"[cv-preprocess] NFA persistent_worker 失敗、従来の align.py 起動にフォールバック: {err_one}",
                    file=sys.stderr,
                    flush=True,
                )
                close_nfa_worker()
                results = _run_nfa_align_subprocess(
                    cfg,
                    items,
                    align_dir=align_dir,
                    nfa_py=nfa_py,
                    corpus_dir=corpus_dir,
                    align_out=align_out,
                    manifest_path=manifest_path,
                )
        else:
            results = _run_nfa_align_subprocess(
                cfg,
                items,
                align_dir=align_dir,
                nfa_py=nfa_py,
                corpus_dir=corpus_dir,
                align_out=align_out,
                manifest_path=manifest_path,
            )
        _attach_pred_texts_if_needed(cfg, align_out, manifest_path, results)
        return results
    finally:
        if cfg.clean_workdir:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            shutil.rmtree(align_out, ignore_errors=True)
