from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from cv_preprocess.audio.nfa_batch import NfaClipResult
from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline import nfa_g2p_map_suggest as nfa_suggest_mod
from cv_preprocess.pipeline.mfa_g2p_map_suggest import _g2p_tokens
from cv_preprocess.pipeline.nfa_g2p_map_suggest import run_nfa_g2p_map_suggest


def test_run_nfa_g2p_map_suggest_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corp = tmp_path / "ja"
    clips = corp / "clips"
    clips.mkdir(parents=True)
    sr = 22050
    t = np.linspace(0, 0.4, int(sr * 0.4), dtype=np.float32)
    y = (0.08 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(clips / "one.wav", y, sr, subtype="PCM_16")
    (corp / "validated.tsv").write_text(
        "client_id\tpath\tsentence\tlocale\n"
        "c1\tone.wav\tこんにちは\tja\n",
        encoding="utf-8",
    )

    map_stub = tmp_path / "existing.yaml"
    map_stub.write_text("{}", encoding="utf-8")

    def fake_nfa(_cfg, items, work_parent):
        out: list[NfaClipResult] = []
        for _utt, _y, _sr, text_norm in items:
            n = len(_g2p_tokens(text_norm, kana=False))
            toks = " ".join(f"NF{i}" for i in range(n))
            out.append(NfaClipResult(ok=True, token_string=toks))
        return out

    monkeypatch.setattr(nfa_suggest_mod, "run_nfa_align_batch", fake_nfa)

    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": corp, "locale_expected": "ja"},
            "output": {"root": tmp_path / "out"},
            "text": {
                "phonemize": True,
                "require_japanese": True,
                "phoneme_alignment_check": {"enabled": False},
            },
            "early_audio_gate": {"enabled": False},
            "two_pass_denoise": {"enabled": False},
            "quality_gate": {
                "min_duration_sec": 0.05,
                "max_duration_sec": 60.0,
                "max_silence_ratio": 0.99,
                "quality_tier_mode": "off",
                "min_estimated_snr_db": None,
            },
            "audio_pipeline": {
                "target_sample_rate": 22050,
                "steps": [
                    {"type": "resample", "sr": 22050},
                    {"type": "save_wav", "bit_depth": 16},
                ],
            },
            "nfa_gate": {
                "enabled": True,
                "pretrained_name": "nvidia/parakeet-tdt_ctc-0.6b-ja",
                "prefilter": {"enabled": False},
                "compare_tokens_to_g2p": True,
                "nfa_to_g2p_token_map_path": map_stub,
            },
        }
    )

    out_yaml = tmp_path / "nfa_map_out.yaml"
    rep = run_nfa_g2p_map_suggest(
        cfg,
        output_yaml=out_yaml,
        min_votes=1,
        min_ratio=0.51,
        show_progress=False,
    )
    assert out_yaml.is_file()
    assert Path(rep["report_path"]).is_file()
    assert rep["counts"]["rows_considered"] >= 1
    body = out_yaml.read_text(encoding="utf-8")
    assert "NF0" in body or "NF1" in body
