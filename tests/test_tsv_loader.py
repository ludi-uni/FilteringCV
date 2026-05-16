from __future__ import annotations

import csv
from pathlib import Path

from cv_preprocess.config import PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow, load_validated_tsv, prepare_clip_rows
from cv_preprocess.pipeline.scan import scan_corpus


def test_load_validated_tsv_multiline_quoted_sentence_merges_physical_lines(tmp_path: Path) -> None:
    """CV TSV は文中に \" を含むとクォートされ、改行で物理行が増える。論理行は1件のまま。"""
    tsv = tmp_path / "validated.tsv"
    # 2 論理行: 1 行目は sentence 内に改行を含む（公式の csv で書き出し）
    rows = [
        [
            "client_a",
            "a.mp3",
            "sid_a",
            "line1\nline2",
            "",
            "2",
            "0",
            "",
            "",
            "",
            "",
            "ja",
            "",
        ],
        [
            "client_b",
            "b.mp3",
            "sid_b",
            "単一行",
            "",
            "2",
            "0",
            "",
            "",
            "",
            "",
            "ja",
            "",
        ],
    ]
    with tsv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "client_id",
                "path",
                "sentence_id",
                "sentence",
                "sentence_domain",
                "up_votes",
                "down_votes",
                "age",
                "gender",
                "accents",
                "variant",
                "locale",
                "segment",
            ]
        )
        for row in rows:
            w.writerow(row)

    physical = sum(1 for _ in tsv.open("rb"))
    assert physical == 4, "ヘッダ + 論理2行だが1行目の sentence が2物理行"  # noqa: PLR2004

    loaded, stats = load_validated_tsv(tsv)
    assert stats["rows_ok"] == 2
    assert [r.client_id for r in loaded] == ["client_a", "client_b"]


def test_load_validated_tsv_strips_utf8_bom(tmp_path: Path) -> None:
    tsv = tmp_path / "validated.tsv"
    tsv.write_text(
        "\ufeffclient_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
        "c1\tx.mp3\ts\thello\t\t2\t0\t\t\t\t\tja\t\n",
        encoding="utf-8",
    )
    loaded, stats = load_validated_tsv(tsv)
    assert stats["rows_ok"] == 1
    assert loaded[0].client_id == "c1"


def _write_minimal_tsv(tmp_path: Path, lines: list[str]) -> Path:
    tsv = tmp_path / "validated.tsv"
    header = (
        "client_id\tpath\tsentence_id\tsentence\tsentence_domain\t"
        "up_votes\tdown_votes\tage\tgender\taccents\tvariant\tlocale\tsegment\n"
    )
    tsv.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return tsv


def test_prepare_clip_rows_speaker_filter_without_merge(tmp_path: Path) -> None:
    corp = tmp_path / "ja"
    corp.mkdir()
    _write_minimal_tsv(
        corp,
        [
            "keep\ta.mp3\ts\thello\t\t2\t0\t\t\t\t\tja\t",
            "drop\tb.mp3\ts\tworld\t\t2\t0\t\t\t\t\tja\t",
        ],
    )
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": corp},
            "speakers": {"include_client_ids": ["keep"]},
        }
    )
    rows, _ = load_validated_tsv(corp / "validated.tsv")
    filtered, after_sp, after_meta, after_cap = prepare_clip_rows(
        rows,
        cfg,
        apply_speaker_merge=False,
        sort_by_path=False,
    )
    assert after_sp == 1
    assert after_meta == 1
    assert after_cap == 1
    assert [r.client_id for r in filtered] == ["keep"]
    assert [r.client_id for r in rows] == ["keep", "drop"]


def test_prepare_clip_rows_ignores_max_clips_per_speaker_at_load(tmp_path: Path) -> None:
    """max_clips_per_speaker は preprocess の採用段階でのみ効く（行読み込みでは絞らない）。"""
    rows = [
        ClipRow("s1", "b.mp3", "x", {}, locale=None, sentence_id=None),
        ClipRow("s1", "a.mp3", "y", {}, locale=None, sentence_id=None),
        ClipRow("s2", "c.mp3", "z", {}, locale=None, sentence_id=None),
    ]
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": tmp_path},
            "speakers": {"max_clips_per_speaker": 2},
        }
    )
    filtered, after_sp, after_meta, after_cap = prepare_clip_rows(
        rows,
        cfg,
        apply_speaker_merge=False,
        sort_by_path=True,
    )
    assert after_cap == 3
    assert len(filtered) == 3


def test_scan_corpus_uses_shared_row_filters(tmp_path: Path) -> None:
    corp = tmp_path / "ja"
    (corp / "clips").mkdir(parents=True)
    _write_minimal_tsv(
        corp,
        [
            "c1\ta.mp3\ts\thello\t\t2\t0\t\t\t\t\tja\t",
            "c2\tb.mp3\ts\tworld\t\t2\t0\t\t\t\t\tja\t",
        ],
    )
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": corp},
            "speakers": {"include_client_ids": ["c1"]},
        }
    )
    info = scan_corpus(cfg)
    assert info["rows_after_speaker_filter"] == 1
    assert info["rows_after_clip_metadata_filter"] == 1
    assert info["unique_client_ids"] == 2
