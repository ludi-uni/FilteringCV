"""ノイズ除去後 WAV に対する Montreal Forced Aligner（``mfa align``）バッチ呼び出し。"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from cv_preprocess.config import MfaGateConfig


@dataclass
class MfaClipResult:
    ok: bool
    phone_string: str | None
    detail: str | None = None


def mfa_cli_available(mfa_executable: str) -> bool:
    """``mfa_executable`` が既存ファイル、または ``PATH`` 上のコマンドとして解決できるか。"""
    p = Path(mfa_executable)
    if p.is_file():
        return True
    return shutil.which(mfa_executable) is not None


def _safe_lab_text(text_norm: str, *, strip_spaces: bool) -> str:
    s = text_norm.replace("\r", " ").replace("\n", " ").strip()
    if strip_spaces:
        s = re.sub(r"\s+", "", s)
    return s


def _write_batch_corpus(
    work: Path,
    items: list[tuple[str, np.ndarray, int, str]],
    *,
    lab_strip_spaces: bool,
) -> None:
    """items: (utterance_id, y, sr, text_norm)"""
    work.mkdir(parents=True, exist_ok=True)
    for utt_id, y, sr, text_norm in items:
        wav_path = work / f"{utt_id}.wav"
        lab_path = work / f"{utt_id}.lab"
        y_f = np.asarray(y, dtype=np.float32).reshape(-1)
        y_i16 = np.clip(y_f, -1.0, 1.0) * 32767.0
        sf.write(str(wav_path), y_i16.astype(np.int16), int(sr), subtype="PCM_16")
        lab_path.write_text(_safe_lab_text(text_norm, strip_spaces=lab_strip_spaces) + "\n", encoding="utf-8")


def _find_textgrid_for_utt(align_out: Path, utt_id: str) -> Path | None:
    """MFA の出力はフラットでもサブディレクトリでもあり得る。"""
    direct = align_out / f"{utt_id}.TextGrid"
    if direct.is_file():
        return direct
    for p in align_out.rglob(f"{utt_id}.TextGrid"):
        if p.is_file():
            return p
    return None


def run_mfa_align_batch(
    cfg: MfaGateConfig,
    items: list[tuple[str, np.ndarray, int, str]],
    *,
    work_parent: Path,
) -> list[MfaClipResult]:
    """
    ``items``: (utterance_id, waveform, sample_rate, text_norm)

    ``mfa align CORPUS DICT ACOUSTIC OUT`` を1回実行し、各 utt の TextGrid を解析する。
    """
    if not items:
        return []

    from cv_preprocess.audio.textgrid_phones import extract_phone_tokens_from_textgrid, phones_to_space_string

    batch_root = cfg.work_dir if cfg.work_dir is not None else work_parent
    batch_root.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex[:12]
    corpus_dir = batch_root / f"corpus_{batch_id}"
    align_out = batch_root / f"aligned_{batch_id}"

    try:
        _write_batch_corpus(
            corpus_dir,
            items,
            lab_strip_spaces=cfg.lab_strip_spaces,
        )

        cmd: list[str] = [
            cfg.mfa_executable,
            "align",
            str(corpus_dir),
            cfg.dictionary,
            cfg.acoustic_model,
            str(align_out),
            "--num_jobs",
            str(cfg.num_jobs),
        ]
        if cfg.single_speaker:
            cmd.append("--single_speaker")
        if cfg.clean:
            cmd.append("--clean")
        if cfg.beam is not None:
            cmd.extend(["--beam", str(cfg.beam)])
        if cfg.retry_beam is not None:
            cmd.extend(["--retry_beam", str(cfg.retry_beam)])
        if cfg.g2p_model_path:
            cmd.extend(["--g2p_model_path", cfg.g2p_model_path])
        cmd.extend(cfg.extra_align_args)

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=cfg.timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[:4000]
            return [
                MfaClipResult(ok=False, phone_string=None, detail=f"mfa_exit_{proc.returncode}: {err}")
                for _ in items
            ]

        results: list[MfaClipResult] = []
        for utt_id, _, _, _ in items:
            tg = _find_textgrid_for_utt(align_out, utt_id)
            if tg is None:
                results.append(
                    MfaClipResult(ok=False, phone_string=None, detail="textgrid_missing"),
                )
                continue
            try:
                toks = extract_phone_tokens_from_textgrid(tg)
                results.append(
                    MfaClipResult(ok=True, phone_string=phones_to_space_string(toks) if toks else ""),
                )
            except OSError as e:
                results.append(
                    MfaClipResult(ok=False, phone_string=None, detail=f"textgrid_read:{e}"),
                )
        return results
    finally:
        if cfg.clean_workdir:
            shutil.rmtree(corpus_dir, ignore_errors=True)
            shutil.rmtree(align_out, ignore_errors=True)
