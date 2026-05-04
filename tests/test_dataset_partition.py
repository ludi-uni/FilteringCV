import json
from pathlib import Path

from cv_preprocess.pipeline.dataset_partition import run_dataset_partition, validate_group_by


def test_validate_group_by() -> None:
    assert validate_group_by("quality_tier") == "quality_tier"
    assert validate_group_by("SPLIT") == "split"
    assert validate_group_by("split-quality-tier") == "split_quality_tier"


def test_partition_by_quality_tier_copies(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "wavs").mkdir(parents=True)
    wav = root / "wavs" / "cv_ja_000001.wav"
    wav.write_bytes(b"RIFFfake")

    meta = root / "metadata.jsonl"
    rec = {
        "utt_id": "cv_ja_000001",
        "audio_path": "wavs/cv_ja_000001.wav",
        "text_norm": "あ",
        "text_raw": "あ",
        "quality_tier": "A",
        "quality_score": 88.5,
        "split": "train",
    }
    meta.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

    out = tmp_path / "partitioned"
    r = run_dataset_partition(
        metadata_path=meta,
        audio_root=root,
        output_root=out,
        group_by="quality_tier",
        use_symlink=False,
    )
    assert r["buckets"] == {"A": 1}
    assert (out / "A" / "wavs" / "cv_ja_000001.wav").read_bytes() == b"RIFFfake"
    lines = (out / "A" / "metadata.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["quality_tier"] == "A"
    tsv = (out / "A" / "validated.tsv").read_text(encoding="utf-8").strip()
    assert tsv.split("\t")[0] == "wavs/cv_ja_000001.wav"


def test_only_tiers_filter(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "wavs").mkdir(parents=True)
    for name, tier in [("a.wav", "A"), ("b.wav", "B")]:
        (root / "wavs" / name).write_bytes(b"x")
    meta = root / "metadata.jsonl"
    rows = [
        {"audio_path": f"wavs/{n}", "text_norm": "t", "text_raw": "t", "quality_tier": t}
        for n, t in [("a.wav", "A"), ("b.wav", "B")]
    ]
    meta.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")

    out = tmp_path / "out"
    r = run_dataset_partition(
        metadata_path=meta,
        audio_root=root,
        output_root=out,
        group_by="quality_tier",
        only_tiers=["A"],
        use_symlink=False,
    )
    assert r["buckets"] == {"A": 1}
    assert r["skipped_by_filter"] == 1
    assert not (out / "B").exists()


def test_dry_run_no_files(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    (root / "wavs").mkdir(parents=True)
    (root / "wavs" / "x.wav").write_bytes(b"x")
    meta = root / "metadata.jsonl"
    meta.write_text(
        json.dumps(
            {
                "audio_path": "wavs/x.wav",
                "text_norm": "a",
                "text_raw": "a",
                "quality_tier": "B",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "empty"
    r = run_dataset_partition(
        metadata_path=meta,
        audio_root=root,
        output_root=out,
        group_by="quality_tier",
        dry_run=True,
    )
    assert r["buckets"] == {"B": 1}
    assert not out.exists() or not any(out.iterdir())
