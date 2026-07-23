from __future__ import annotations

from cv_preprocess.config.dataset_builder import DatasetBuilderSplitConfig, PreserveTrainConfig
from cv_preprocess.split.leakage import ClipSplitRecord
from cv_preprocess.split.seen_speaker import assign_clip_splits
from cv_preprocess.split.protocol import SplitProtocol


def _clip(
    clip_id: str,
    *,
    speaker: str,
    phone: str,
    duration_sec: float = 1.0,
) -> ClipSplitRecord:
    return ClipSplitRecord(
        clip_id=clip_id,
        speaker_id=speaker,
        audio_hash=f"audio-{clip_id}",
        sentence_id=f"sent-{clip_id}",
        normalized_text=f"text-{clip_id}",
        duration_sec=duration_sec,
        features_by_family={"phone": [phone]},
    )


def test_critical_feature_keeps_min_train_occurrences() -> None:
    clips = [
        _clip("c1", speaker="spk-a", phone="rare"),
        _clip("c2", speaker="spk-b", phone="rare"),
        _clip("c3", speaker="spk-c", phone="common"),
        _clip("c4", speaker="spk-d", phone="common"),
        _clip("c5", speaker="spk-e", phone="common"),
        _clip("c6", speaker="spk-f", phone="common"),
    ]
    config = DatasetBuilderSplitConfig.model_validate(
        {
            "protocol": "seen_speaker",
            "train": 0.5,
            "val": 0.25,
            "test": 0.25,
            "seed": 1,
            "preserve_train": PreserveTrainConfig(
                enabled=True,
                critical_feature_max_speakers=2,
                min_train_occurrences=1,
            ),
            "leakage_policy": {
                "speaker": "allow",
                "audio_hash": "forbid",
                "sentence_id": "forbid",
                "normalized_text": "forbid",
            },
        }
    )

    assignments, _warnings = assign_clip_splits(
        clips,
        config,
        protocol=SplitProtocol.SEEN_SPEAKER,
        feature_speaker_support={("phone", "rare"): 2, ("phone", "common"): 4},
    )

    train_clips = {clip_id for clip_id, split_name in assignments.items() if split_name == "train"}
    rare_in_train = [clip for clip in clips if clip.clip_id in train_clips and "rare" in clip.features_by_family["phone"]]
    assert len(rare_in_train) >= config.resolved_preserve_train().min_train_occurrences
