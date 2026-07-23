from __future__ import annotations

from cv_preprocess.catalog import ClipDisposition, stable_clip_id


def test_stable_clip_id_is_deterministic() -> None:
    kwargs = {
        "source_release": "cv-corpus-25.0",
        "normalized_relative_source_path": "clips/foo.mp3",
        "source_row_index": 42,
        "audio_sha256": "a" * 64,
        "text_raw": "今日はいい天気です。",
    }
    first = stable_clip_id(**kwargs)
    second = stable_clip_id(**kwargs)
    assert first == second
    assert len(first) == 64


def test_stable_clip_id_changes_when_inputs_change() -> None:
    base = {
        "source_release": "cv-corpus-25.0",
        "normalized_relative_source_path": "clips/foo.mp3",
        "source_row_index": 1,
        "audio_sha256": "b" * 64,
        "text_raw": "hello",
    }
    baseline = stable_clip_id(**base)

    path_variant = stable_clip_id(**{**base, "normalized_relative_source_path": "clips/bar.mp3"})
    text_variant = stable_clip_id(**{**base, "text_raw": "world"})
    hash_variant = stable_clip_id(**{**base, "audio_sha256": "c" * 64})

    assert path_variant != baseline
    assert text_variant != baseline
    assert hash_variant != baseline


def test_stable_clip_id_order_independent() -> None:
    kwargs_a = {
        "source_release": "release-a",
        "normalized_relative_source_path": "clips/a.mp3",
        "source_row_index": 10,
        "audio_sha256": "d" * 64,
        "text_raw": "alpha",
    }
    kwargs_b = {
        "source_release": "release-b",
        "normalized_relative_source_path": "clips/b.mp3",
        "source_row_index": 20,
        "audio_sha256": "e" * 64,
        "text_raw": "beta",
    }

    first_pass = [stable_clip_id(**kwargs_a), stable_clip_id(**kwargs_b)]
    second_pass = [stable_clip_id(**kwargs_b), stable_clip_id(**kwargs_a)]

    assert first_pass[0] == second_pass[1]
    assert first_pass[1] == second_pass[0]


def test_clip_disposition_values() -> None:
    assert ClipDisposition.HARD_REJECTED.value == "hard_rejected"
    assert ClipDisposition.ELIGIBLE.value == "eligible"
    assert ClipDisposition.SELECTED.value == "selected"
    assert ClipDisposition.RESERVE.value == "reserve"
