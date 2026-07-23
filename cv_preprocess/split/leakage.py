from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cv_preprocess.config.dataset_builder import LeakagePolicyConfig

LeakagePolicyAction = Literal["forbid", "forbid_for_test", "allow"]

LEAKAGE_DIMENSIONS: tuple[str, ...] = (
    "speaker",
    "audio_hash",
    "sentence_id",
    "normalized_text",
)


@dataclass(frozen=True)
class ClipSplitRecord:
    clip_id: str
    speaker_id: str
    audio_hash: str
    sentence_id: str
    normalized_text: str
    duration_sec: float = 0.0
    features_by_family: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class LeakageViolation:
    dimension: str
    key: str
    splits: list[str]
    clip_ids_by_split: dict[str, list[str]]


def _resolve_leakage_policy(
    policy: LeakagePolicyConfig | Literal["strict", "warn", "off"],
) -> LeakagePolicyConfig:
    if isinstance(policy, LeakagePolicyConfig):
        return policy
    if policy == "strict":
        return LeakagePolicyConfig()
    if policy == "warn":
        return LeakagePolicyConfig()
    return LeakagePolicyConfig(
        speaker="allow",
        audio_hash="allow",
        sentence_id="allow",
        normalized_text="allow",
    )


def _dimension_value(clip: ClipSplitRecord, dimension: str) -> str | None:
    if dimension == "speaker":
        return clip.speaker_id or None
    if dimension == "audio_hash":
        return clip.audio_hash or None
    if dimension == "sentence_id":
        return clip.sentence_id or None
    if dimension == "normalized_text":
        return clip.normalized_text or None
    return None


def _is_violation(
    splits: set[str],
    action: LeakagePolicyAction,
) -> bool:
    if action == "allow" or len(splits) < 2:
        return False
    if action == "forbid":
        return True
    return "test" in splits and ("train" in splits or "val" in splits)


def detect_leakage(
    selected_clips_by_split: dict[str, list[ClipSplitRecord]],
    leakage_policy: LeakagePolicyConfig | Literal["strict", "warn", "off"],
) -> list[LeakageViolation]:
    """Return leakage violations across splits for configured dimensions."""
    policy = _resolve_leakage_policy(leakage_policy)
    violations: list[LeakageViolation] = []

    for dimension in LEAKAGE_DIMENSIONS:
        action: LeakagePolicyAction = getattr(policy, dimension)
        if action == "allow":
            continue

        key_to_splits: dict[str, set[str]] = {}
        key_to_clips: dict[str, dict[str, list[str]]] = {}

        for split_name, clips in selected_clips_by_split.items():
            for clip in clips:
                value = _dimension_value(clip, dimension)
                if not value:
                    continue
                key_to_splits.setdefault(value, set()).add(split_name)
                key_to_clips.setdefault(value, {}).setdefault(split_name, []).append(clip.clip_id)

        for key, splits in sorted(key_to_splits.items()):
            if not _is_violation(splits, action):
                continue
            violations.append(
                LeakageViolation(
                    dimension=dimension,
                    key=key,
                    splits=sorted(splits),
                    clip_ids_by_split={
                        split_name: sorted(set(key_to_clips[key].get(split_name, [])))
                        for split_name in sorted(splits)
                    },
                )
            )

    return violations
