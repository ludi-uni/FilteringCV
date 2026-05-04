from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cv_preprocess.config import PipelineConfig
from cv_preprocess.pipeline.phoneme_manifest import run_phoneme_manifest


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "sentence", "client_id", "locale"], delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_phoneme_manifest_g2p_text(tmp_path: Path) -> None:
    corpus = tmp_path / "ja"
    (corpus / "clips").mkdir(parents=True)
    _write_tsv(
        corpus / "validated.tsv",
        [
            {
                "path": "a.mp3",
                "sentence": "あ",
                "client_id": "c1",
                "locale": "ja",
            }
        ],
    )
    out = tmp_path / "aligned.jsonl"
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": corpus},
            "text": {"phonemize": True, "g2p_kana": False},
        }
    )
    rep = run_phoneme_manifest(cfg, output_path=out, source="g2p_text", show_progress=False)
    assert rep["counts"]["rows_written"] == 1
    line = out.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["source_path"] == "a.mp3"
    assert obj["phonemes"] == "a"


def test_phoneme_manifest_mfa_textgrid_with_map(tmp_path: Path) -> None:
    corpus = tmp_path / "ja"
    (corpus / "clips").mkdir(parents=True)
    _write_tsv(
        corpus / "validated.tsv",
        [
            {
                "path": "clip_x.mp3",
                "sentence": "あ",
                "client_id": "c1",
                "locale": "ja",
            }
        ],
    )
    mfa_dir = tmp_path / "mfa"
    mfa_dir.mkdir()
    tg = mfa_dir / "clip_x.TextGrid"
    tg.write_text(
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.5
tiers? <exists>
size = 1
item [1]:
    class = "IntervalTier"
    name = "phones"
    xmin = 0
    xmax = 0.5
    intervals: size = 1
    intervals [1]:
        xmin = 0
        xmax = 0.5
        text = "MFA_A"
""",
        encoding="utf-8",
    )
    map_path = tmp_path / "map.yaml"
    map_path.write_text('MFA_A: "a"\n', encoding="utf-8")
    out = tmp_path / "out.jsonl"
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": corpus},
            "text": {"phonemize": True},
        }
    )
    rep = run_phoneme_manifest(
        cfg,
        output_path=out,
        source="mfa_textgrid",
        mfa_textgrid_root=mfa_dir,
        mfa_token_map_path=map_path,
        show_progress=False,
    )
    assert rep["counts"]["rows_written"] == 1
    obj = json.loads(out.read_text(encoding="utf-8").strip())
    assert obj["phonemes"] == "a"


def test_phoneme_manifest_requires_output() -> None:
    cfg = PipelineConfig.model_validate({"input": {"corpus_root": Path("x")}})
    with pytest.raises(ValueError, match="出力先"):
        run_phoneme_manifest(cfg, output_path=None, source="g2p_text", show_progress=False)


def test_phoneme_manifest_mfa_requires_root(tmp_path: Path) -> None:
    corpus = tmp_path / "ja"
    _write_tsv(corpus / "validated.tsv", [{"path": "a.mp3", "sentence": "あ", "client_id": "c", "locale": "ja"}])
    cfg = PipelineConfig.model_validate({"input": {"corpus_root": corpus}})
    out = tmp_path / "o.jsonl"
    with pytest.raises(ValueError, match="mfa_textgrid_root"):
        run_phoneme_manifest(cfg, output_path=out, source="mfa_textgrid", show_progress=False)
