from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from cv_preprocess.config import ClipMetadataFilters


def _relax_csv_field_limit() -> None:
    """Raise csv field size limit (default 128KiB); long `sentence` cells can exceed it."""
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            break
        except OverflowError:
            max_size = int(max_size / 10)


@dataclass
class ClipRow:
    client_id: str
    path: str
    sentence: str
    raw: dict[str, str]
    locale: str | None = None
    sentence_id: str | None = None


def _normalize_header(name: str) -> str:
    n = name.strip()
    aliases = {
        "accent": "accents",
    }
    return aliases.get(n, n)


def load_validated_tsv(
    tsv_path: Path,
    *,
    skip_on_missing_required: bool = True,
) -> tuple[list[ClipRow], dict[str, int]]:
    """Load Common Voice clip TSV (validated.tsv etc.)."""
    _relax_csv_field_limit()
    stats = {"rows_total": 0, "rows_skipped": 0, "rows_ok": 0}
    rows: list[ClipRow] = []
    with tsv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames:
            return rows, stats
        fieldmap = {_normalize_header(h): h for h in reader.fieldnames}

        def get(row: dict[str, str], key: str) -> str | None:
            orig = fieldmap.get(key)
            if not orig:
                return None
            v = row.get(orig)
            if v is None or v == "":
                return None
            return v

        for raw_row in reader:
            stats["rows_total"] += 1
            path = get(raw_row, "path")
            sentence = get(raw_row, "sentence")
            client_id = get(raw_row, "client_id")
            if not path or not sentence or not client_id:
                stats["rows_skipped"] += 1
                if not skip_on_missing_required:
                    raise ValueError(f"Missing required columns in row {stats['rows_total']}")
                continue
            locale = get(raw_row, "locale")
            sentence_id = get(raw_row, "sentence_id")
            rows.append(
                ClipRow(
                    client_id=client_id,
                    path=path,
                    sentence=sentence,
                    raw={k: raw_row[k] for k in raw_row if raw_row[k] is not None},
                    locale=locale,
                    sentence_id=sentence_id,
                )
            )
            stats["rows_ok"] += 1
    return rows, stats


def filter_by_speakers(rows: list[ClipRow], include: list[str] | None) -> list[ClipRow]:
    if not include:
        return rows
    s = {x.strip() for x in include if x and str(x).strip()}
    return [r for r in rows if r.client_id.strip() in s]


def apply_merge_filtered_speakers_as_one(
    rows: list[ClipRow],
    *,
    enabled: bool,
    merged_client_id: str,
) -> None:
    """``merge_filtered_speakers_as_one`` 用に、残存行の ``client_id`` をすべて ``merged_client_id`` に上書きする。"""
    if not enabled or not rows:
        return
    cid = merged_client_id.strip() or "__cv_merged_speaker__"
    for r in rows:
        r.client_id = cid


def _row_raw_field(row: ClipRow, logical_key: str) -> str | None:
    """TSV 列名を正規化したキー（accents 等）で raw から値を取得。空セルは None。"""
    for k, v in row.raw.items():
        if _normalize_header(k.strip()) == logical_key:
            if v is None or str(v).strip() == "":
                return None
            return str(v).strip()
    return None


def _allowset_with_blank(values: list[str]) -> tuple[set[str], bool]:
    """許容トークンの集合と、明示 ``""`` による「空セル許容」フラグ。"""
    allow_blank = False
    s: set[str] = set()
    for v in values:
        if v is None:
            continue
        t = str(v).strip()
        if t == "":
            allow_blank = True
        else:
            s.add(t.casefold())
    return s, allow_blank


def _field_matches_allowlist(value: str | None, allow: set[str], allow_blank: bool) -> bool:
    if value is None:
        return allow_blank
    return value.casefold() in allow


def _row_int_field(row: ClipRow, logical_key: str) -> int | None:
    v = _row_raw_field(row, logical_key)
    if v is None:
        return None
    try:
        return int(str(v).strip(), 10)
    except ValueError:
        return None


def filter_by_clip_metadata(rows: list[ClipRow], filters: ClipMetadataFilters | None) -> list[ClipRow]:
    """CV の gender / age 等がすべて指定された許容値に含まれる行だけ残す。各軸の許容リストが空ならその軸は見ない。

    許容リストに要素 ``""``（空文字列）が **明示的に** 含まれる場合のみ、その列が TSV で空の行も通す。
    含まれない場合、空セルはその軸で不一致として落とす。
    """
    if filters is None or not filters.is_active():
        return rows

    genders, genders_blank = _allowset_with_blank(filters.genders)
    ages, ages_blank = _allowset_with_blank(filters.ages)
    accents, accents_blank = _allowset_with_blank(filters.accents)
    variants, variants_blank = _allowset_with_blank(filters.variants)
    locales, locales_blank = _allowset_with_blank(filters.locales)
    segments, segments_blank = _allowset_with_blank(filters.segments)

    out: list[ClipRow] = []
    for row in rows:
        if filters.genders:
            v = _row_raw_field(row, "gender")
            if not _field_matches_allowlist(v, genders, genders_blank):
                continue
        if filters.ages:
            v = _row_raw_field(row, "age")
            if not _field_matches_allowlist(v, ages, ages_blank):
                continue
        if filters.accents:
            v = _row_raw_field(row, "accents")
            if not _field_matches_allowlist(v, accents, accents_blank):
                continue
        if filters.variants:
            v = _row_raw_field(row, "variant")
            if not _field_matches_allowlist(v, variants, variants_blank):
                continue
        if filters.locales:
            v = _row_raw_field(row, "locale")
            if not _field_matches_allowlist(v, locales, locales_blank):
                continue
        if filters.segments:
            v = _row_raw_field(row, "segment")
            if not _field_matches_allowlist(v, segments, segments_blank):
                continue
        if filters.up_votes is not None:
            u = _row_int_field(row, "up_votes")
            if u is None or u < filters.up_votes:
                continue
        if filters.down_votes is not None:
            d = _row_int_field(row, "down_votes")
            if d is None or d > filters.down_votes:
                continue
        out.append(row)
    return out


def iter_clip_audio_paths(corpus_root: Path, audio_subdir: str, row: ClipRow) -> Path:
    return corpus_root / audio_subdir / row.path
