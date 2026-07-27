from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Duck-typed; concrete ProgressEvent lives in application.common (avoid circular import).
ProgressSink = Callable[[Any], None]


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
    #: Coverage feature keys (``phoneme:v``, ``mora:…``) aligned with coverage-run.
    coverage_keys: list[str] = field(default_factory=list)
    #: Optional acoustic metrics for lightweight diversity (rms, snr, …).
    acoustic_metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass
class SelectionExplanation:
    selection_score: float = 0.0
    rank: int | None = None
    positive_contributions: dict[str, Any] = field(default_factory=dict)
    penalties: dict[str, Any] = field(default_factory=dict)
    selected_reason: str | None = None
    reserve_reason: str | None = None
    selection_phase: str | None = None
    coverage_contributions: list[str] = field(default_factory=list)


@dataclass
class SelectionResult:
    selected_ids: list[str]
    reserve_ids: list[str]
    explanations: dict[str, SelectionExplanation] = field(default_factory=dict)
    coverage_audit: dict[str, Any] | None = None
    coverage_contributions: list[dict[str, Any]] = field(default_factory=list)
    acoustic_summary: dict[str, Any] | None = None
    missing_features: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage_report_paths: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class SelectionBackend(Protocol):
    def select(
        self,
        candidates: list[ClipFeatures],
        *,
        target_duration_sec: float,
        tolerance_ratio: float,
        seed: int,
        progress: ProgressSink | None = None,
        progress_label: str | None = None,
        initial_selected: list[str] | None = None,
    ) -> SelectionResult: ...
