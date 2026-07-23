from __future__ import annotations

import random
from pathlib import Path

from cv_preprocess.application.common import SplitPlan
from cv_preprocess.application.select import select_dataset
from cv_preprocess.selection.python_backend import greedy_local_search

from tests.dataset_builder.conftest_helpers import make_synthetic_catalog, selection_pipeline_config


def _coverage_rows() -> list[dict]:
    return [
        {
            "clip_id": "head_common_1",
            "speaker_id": "spk1",
            "phonemes": "a a a",
            "text_norm": "あああ",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "head_common_2",
            "speaker_id": "spk2",
            "phonemes": "a a a",
            "text_norm": "あああ",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "rare_x",
            "speaker_id": "spk3",
            "phonemes": "x y z",
            "text_norm": "きつね",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "rare_q",
            "speaker_id": "spk4",
            "phonemes": "q r s",
            "text_norm": "くま",
            "duration_sec": 1.0,
        },
        {
            "clip_id": "rare_w",
            "speaker_id": "spk5",
            "phonemes": "w t u",
            "text_norm": "うさぎ",
            "duration_sec": 1.0,
        },
    ]


def _phone_coverage(clip_ids: list[str], rows: list[dict]) -> int:
    phones: set[str] = set()
    by_id = {row["clip_id"]: row for row in rows}
    for clip_id in clip_ids:
        phonemes = by_id[clip_id]["phonemes"].split()
        phones.update(phonemes)
    return len(phones)


def test_greedy_beats_head_on_phone_coverage(tmp_path: Path) -> None:
    rows = _coverage_rows()
    config = selection_pipeline_config(
        tmp_path,
        target_duration_hours=3 / 3600,
        extra={
            "dataset_builder": {
                "selection": {
                    "feature_weights": {"phone": 1.0, "speaker_diversity": 0.0, "quality": 0.0},
                    "local_search": {"enabled": False},
                }
            }
        },
    )
    catalog = make_synthetic_catalog(tmp_path, rows)
    result = select_dataset(config, catalog, SplitPlan(catalog=catalog, protocol="unseen_speaker"))

    head_ids = [row["clip_id"] for row in rows[:3]]
    greedy_cov = _phone_coverage(result.selected_clip_ids, rows)
    head_cov = _phone_coverage(head_ids, rows)
    assert greedy_cov > head_cov


def test_greedy_at_least_random_with_margin(tmp_path: Path) -> None:
    rows = _coverage_rows()
    config = selection_pipeline_config(
        tmp_path,
        target_duration_hours=3 / 3600,
        seed=7,
        extra={
            "dataset_builder": {
                "selection": {
                    "feature_weights": {"phone": 1.0, "speaker_diversity": 0.0, "quality": 0.0},
                    "local_search": {"enabled": False},
                }
            }
        },
    )
    catalog = make_synthetic_catalog(tmp_path, rows)
    greedy = select_dataset(config, catalog, SplitPlan(catalog=catalog, protocol="unseen_speaker"))

    from cv_preprocess.application.select import _clip_features_from_catalog
    from cv_preprocess.catalog.reader import read_clips

    candidates = _clip_features_from_catalog(read_clips(catalog.resolved_clips_path()), config, {}, None)
    rng = random.Random(config.dataset_builder.random_seed)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    random_pick = greedy_local_search(
        shuffled,
        config=config.dataset_builder,
        target_duration_sec=0.003 * 3600.0,
        tolerance_ratio=0.5,
        seed=config.dataset_builder.random_seed,
    )
    # emulate random by taking first feasible clips until duration target
    random_ids: list[str] = []
    duration = 0.0
    target = 3 / 3600 * 3600.0
    for clip in shuffled:
        if duration >= target:
            break
        random_ids.append(clip.clip_id)
        duration += clip.duration_sec

    greedy_cov = _phone_coverage(greedy.selected_clip_ids, rows)
    random_cov = _phone_coverage(random_ids, rows)
    assert greedy_cov + 0 >= random_cov - 1
