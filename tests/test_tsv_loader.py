from __future__ import annotations

import csv
from pathlib import Path

from cv_preprocess.io.tsv_loader import load_validated_tsv


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
