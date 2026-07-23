from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ClipFeatures:
    clip_id: str
    speaker_id: str
    duration_sec: float
    quality_score: float | None
    audio_sha256: str
    sentence_id: str
    text_norm: str
    features_by_family: dict[str, list[str]] = field(default_factory=dict)
    duplicate_groups: dict[str, str] = field(default_factory=dict)
    override_action: str | None = None
    split: str | None = None


@dataclass
class SelectionExplanation:
    selection_score: float = 0.0
    rank: int | None = None
    positive_contributions: dict[str, Any] = field(default_factory=dict)
    penalties: dict[str, Any] = field(default_factory=dict)
    selected_reason: str | None = None
    reserve_reason: str | None = None


@dataclass
class SelectionResult:
    selected_ids: list[str]
    reserve_ids: list[str]
    explanations: dict[str, SelectionExplanation] = field(default_factory=dict)


@runtime_checkable
class SelectionBackend(Protocol):
    def select(
        self,
        candidates: list[ClipFeatures],
        *,
        target_duration_sec: float,
        tolerance_ratio: float,
        seed: int,
    ) -> SelectionResult: ...
