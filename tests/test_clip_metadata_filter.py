from __future__ import annotations

import yaml

from cv_preprocess.config import ClipMetadataFilters, PipelineConfig
from cv_preprocess.io.tsv_loader import ClipRow, filter_by_clip_metadata


def _row(**kwargs: str) -> ClipRow:
    raw = {**kwargs}
    return ClipRow(
        client_id=raw.get("client_id", "c1"),
        path=raw.get("path", "a.mp3"),
        sentence=raw.get("sentence", "テスト"),
        raw=raw,
        locale=raw.get("locale"),
    )


def test_filter_gender_allowlist() -> None:
    f = ClipMetadataFilters(genders=["male"])
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", gender="male"),
        _row(client_id="2", path="2.mp3", sentence="b", gender="female"),
        _row(client_id="3", path="3.mp3", sentence="c", gender=""),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert len(out) == 1 and out[0].path == "1.mp3"


def test_filter_gender_allowlist_includes_blank_when_empty_token() -> None:
    f = ClipMetadataFilters(genders=["male", ""])
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", gender="male"),
        _row(client_id="2", path="2.mp3", sentence="b", gender="female"),
        _row(client_id="3", path="3.mp3", sentence="c", gender=""),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert {r.path for r in out} == {"1.mp3", "3.mp3"}


def test_filter_accent_blank_excluded_without_empty_token() -> None:
    f = ClipMetadataFilters(accents=["標準語"])
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", accents="標準語"),
        _row(client_id="2", path="2.mp3", sentence="b", accents=""),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert len(out) == 1 and out[0].path == "1.mp3"


def test_filter_accent_blank_included_with_empty_token() -> None:
    f = ClipMetadataFilters(accents=["標準語", ""])
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", accents="標準語"),
        _row(client_id="2", path="2.mp3", sentence="b", accents=""),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert {r.path for r in out} == {"1.mp3", "2.mp3"}


def test_filter_age_case_insensitive() -> None:
    f = ClipMetadataFilters(ages=["twenties"])
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", age="Twenties"),
        _row(client_id="2", path="2.mp3", sentence="b", age="thirties"),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert len(out) == 1


def test_filter_multi_axis_and() -> None:
    f = ClipMetadataFilters(genders=["female"], ages=["thirties"])
    rows = [
        _row(
            client_id="1",
            path="1.mp3",
            sentence="a",
            gender="female",
            age="thirties",
        ),
        _row(
            client_id="2",
            path="2.mp3",
            sentence="b",
            gender="female",
            age="twenties",
        ),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert len(out) == 1 and out[0].path == "1.mp3"


def test_filter_empty_means_no_filter() -> None:
    f = ClipMetadataFilters()
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a"),
    ]
    assert filter_by_clip_metadata(rows, f) == rows


def test_filter_up_votes_min_and_down_votes_max() -> None:
    f = ClipMetadataFilters(up_votes=2, down_votes=1)
    rows = [
        _row(
            client_id="1",
            path="1.mp3",
            sentence="a",
            up_votes="2",
            down_votes="0",
        ),
        _row(
            client_id="2",
            path="2.mp3",
            sentence="b",
            up_votes="1",
            down_votes="0",
        ),
        _row(
            client_id="3",
            path="3.mp3",
            sentence="c",
            up_votes="5",
            down_votes="2",
        ),
    ]
    out = filter_by_clip_metadata(rows, f)
    assert [r.path for r in out] == ["1.mp3"]


def test_filter_votes_missing_cell_excluded_when_threshold_set() -> None:
    f = ClipMetadataFilters(up_votes=0)
    rows = [
        _row(client_id="1", path="1.mp3", sentence="a", down_votes="0"),
    ]
    assert filter_by_clip_metadata(rows, f) == []


def test_config_yaml_accents_preserves_empty_string_token(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "speakers": {
                    "include_client_ids": [],
                    "clip_metadata_filters": {"accents": ["標準語", ""]},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.speakers.clip_metadata_filters.accents == ["標準語", ""]


def test_config_yaml_speakers_clip_metadata_filters(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        yaml.dump(
            {
                "input": {"corpus_root": str(tmp_path / "ja")},
                "speakers": {
                    "include_client_ids": [],
                    "clip_metadata_filters": {
                        "genders": ["male", "female"],
                        "ages": ["twenties"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.speakers.clip_metadata_filters.genders == ["male", "female"]
    assert cfg.speakers.clip_metadata_filters.ages == ["twenties"]
