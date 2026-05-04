"""speakers.merge_filtered_speakers_as_one の行単位 client_id 統合。"""

from pathlib import Path

from cv_preprocess.config import ClipMetadataFilters, PipelineConfig, load_config
from cv_preprocess.io.tsv_loader import (
    ClipRow,
    apply_merge_filtered_speakers_as_one,
    filter_by_clip_metadata,
    filter_by_speakers,
)


def _row(
    client_id: str = "c1",
    path: str = "a.mp3",
    sentence: str = "あ",
    *,
    gender: str | None = None,
    **extra: str,
) -> ClipRow:
    raw: dict[str, str] = {k: str(v) for k, v in extra.items()}
    if gender is not None:
        raw["gender"] = gender
    return ClipRow(client_id=client_id, path=path, sentence=sentence, raw=raw)


def test_apply_merge_noop_when_disabled() -> None:
    rows = [_row(client_id="a"), _row(client_id="b")]
    apply_merge_filtered_speakers_as_one(rows, enabled=False, merged_client_id="x")
    assert [r.client_id for r in rows] == ["a", "b"]


def test_apply_merge_overwrites_when_enabled() -> None:
    rows = [_row(client_id="a"), _row(client_id="b")]
    apply_merge_filtered_speakers_as_one(rows, enabled=True, merged_client_id="merged_one")
    assert [r.client_id for r in rows] == ["merged_one", "merged_one"]


def test_apply_merge_empty_rows() -> None:
    apply_merge_filtered_speakers_as_one([], enabled=True, merged_client_id="x")


def test_speakers_config_resolved_merged_id() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "speakers": {"merge_filtered_speakers_as_one": True, "merged_speaker_client_id": "  myspk  "},
        }
    )
    assert cfg.speakers.merged_speaker_client_id == "myspk"
    assert cfg.speakers.resolved_merged_speaker_client_id() == "myspk"


def test_speakers_config_default_merged_id() -> None:
    cfg = PipelineConfig.model_validate(
        {
            "input": {"corpus_root": Path("CommonVoice")},
            "speakers": {"merge_filtered_speakers_as_one": True},
        }
    )
    assert cfg.speakers.resolved_merged_speaker_client_id() == "__cv_merged_speaker__"


def test_filter_then_merge_integration() -> None:
    rows = [
        _row(client_id="1", path="1.mp3", gender="female"),
        _row(client_id="2", path="2.mp3", gender="male"),
    ]
    f = ClipMetadataFilters(genders=["female"])
    out = filter_by_clip_metadata(filter_by_speakers(rows, None), f)
    apply_merge_filtered_speakers_as_one(out, enabled=True, merged_client_id="solo")
    assert len(out) == 1
    assert out[0].client_id == "solo"


def test_default_yaml_loads_speaker_merge_fields() -> None:
    p = Path(__file__).resolve().parents[1] / "config" / "default.yaml"
    cfg = load_config(p)
    assert isinstance(cfg.speakers.merge_filtered_speakers_as_one, bool)
    assert cfg.speakers.merged_speaker_client_id is None or isinstance(
        cfg.speakers.merged_speaker_client_id, str
    )
