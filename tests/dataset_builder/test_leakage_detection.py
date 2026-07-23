from __future__ import annotations

from cv_preprocess.config.dataset_builder import LeakagePolicyConfig
from cv_preprocess.split.leakage import ClipSplitRecord, detect_leakage


def _clip(
    clip_id: str,
    *,
    speaker: str = "spk1",
    audio_hash: str = "audio-a",
    sentence_id: str = "sent-a",
    text: str = "text-a",
) -> ClipSplitRecord:
    return ClipSplitRecord(
        clip_id=clip_id,
        speaker_id=speaker,
        audio_hash=audio_hash,
        sentence_id=sentence_id,
        normalized_text=text,
    )


def test_detect_leakage_forbids_speaker_across_splits() -> None:
    by_split = {
        "train": [_clip("c1", speaker="spk1")],
        "test": [_clip("c2", speaker="spk1")],
    }
    violations = detect_leakage(by_split, LeakagePolicyConfig())
    assert any(v.dimension == "speaker" and v.key == "spk1" for v in violations)


def test_detect_leakage_forbid_for_test_allows_train_val_overlap() -> None:
    policy = LeakagePolicyConfig(sentence_id="forbid_for_test")
    by_split = {
        "train": [_clip("c1", sentence_id="shared")],
        "val": [_clip("c2", sentence_id="shared")],
    }
    violations = detect_leakage(by_split, policy)
    assert not any(v.dimension == "sentence_id" for v in violations)


def test_detect_leakage_forbid_for_test_blocks_test_overlap() -> None:
    policy = LeakagePolicyConfig(sentence_id="forbid_for_test")
    by_split = {
        "train": [_clip("c1", sentence_id="shared")],
        "test": [_clip("c2", sentence_id="shared")],
    }
    violations = detect_leakage(by_split, policy)
    assert any(v.dimension == "sentence_id" and v.key == "shared" for v in violations)


def test_detect_leakage_allow_disables_checks() -> None:
    by_split = {
        "train": [_clip("c1", audio_hash="same")],
        "test": [_clip("c2", audio_hash="same")],
    }
    violations = detect_leakage(by_split, "off")
    assert violations == []
