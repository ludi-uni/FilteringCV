from __future__ import annotations


import polars as pl

from cv_preprocess.config.dataset_builder import DatasetBuilderSplitConfig
from cv_preprocess.split.unseen_speaker import plan_unseen_speaker_splits


def _clips_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_unseen_speaker_assignments_have_no_speaker_overlap() -> None:
    rows = []
    for speaker_idx in range(6):
        speaker = f"spk-{speaker_idx}"
        for clip_idx in range(3):
            rows.append(
                {
                    "clip_id": f"{speaker}-{clip_idx}",
                    "speaker_id": speaker,
                    "duration_sec": 10.0 + clip_idx,
                    "disposition": "eligible",
                    "audio_sha256": f"sha-{speaker}-{clip_idx}",
                    "sentence_id": f"sent-{speaker}-{clip_idx}",
                    "text_norm": f"text-{speaker}-{clip_idx}",
                }
            )

    config = DatasetBuilderSplitConfig.model_validate(
        {
            "protocol": "unseen_speaker",
            "train": 0.6,
            "val": 0.2,
            "test": 0.2,
            "seed": 7,
        }
    )
    assignments, warnings = plan_unseen_speaker_splits(_clips_frame(rows), config)
    assert not warnings or all("overlap" not in warning for warning in warnings)

    split_to_speakers: dict[str, set[str]] = {}
    for speaker, split_name in assignments.items():
        split_to_speakers.setdefault(split_name, set()).add(speaker)

    seen: set[str] = set()
    for speakers in split_to_speakers.values():
        assert seen.isdisjoint(speakers)
        seen |= speakers

    assert len(assignments) == 6
    assert set(assignments.values()) <= {"train", "val", "test"}
