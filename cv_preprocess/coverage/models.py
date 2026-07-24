"""Coverage automation domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StopReason(str, Enum):
    COMPLETE = "complete"
    CANDIDATE_EXHAUSTED = "candidate_exhausted"
    UNREACHABLE = "unreachable"
    LIKELY_UNREACHABLE = "likely_unreachable"
    ANALYSIS_BUDGET_EXCEEDED = "analysis_budget_exceeded"
    ITERATION_LIMIT_REACHED = "iteration_limit_reached"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    CANCELLED = "cancelled"
    FAILED = "failed"
    RUNNING = "running"
    DRY_RUN = "dry_run"


class CheapQuality(BaseModel):
    decode_ok: bool = False
    peak: float | None = None
    rms: float | None = None
    clipping_ratio: float | None = None
    silence_ratio: float | None = None


class ClipIndexRecord(BaseModel):
    clip_id: str
    source_path: str
    client_id: str
    sentence: str
    normalized_text: str
    duration_sec: float | None = None
    sample_rate: int | None = None
    phonemes: list[str] = Field(default_factory=list)
    unique_phonemes: list[str] = Field(default_factory=list)
    moras: list[str] = Field(default_factory=list)
    biphones: list[str] = Field(default_factory=list)
    positioned_phonemes: list[str] = Field(default_factory=list)
    up_votes: int | None = None
    down_votes: int | None = None
    cheap_quality: CheapQuality = Field(default_factory=CheapQuality)
    index_version: int = 1
    source_row_index: int = 0
    audio_sha256: str = ""
    feature_keys: list[str] = Field(default_factory=list)
    invalidated: bool = False

    def feature_key_set(self) -> set[str]:
        if self.feature_keys:
            return set(self.feature_keys)
        keys: set[str] = set()
        for phone in self.unique_phonemes:
            keys.add(f"phoneme:{phone}")
        for mora in set(self.moras):
            keys.add(f"mora:{mora}")
        for biphone in set(self.biphones):
            keys.add(f"biphone:{biphone}")
        for positioned in set(self.positioned_phonemes):
            keys.add(f"positioned_phoneme:{positioned}")
        return keys


class ClipIndexMeta(BaseModel):
    schema_version: int = 1
    created_at: str
    source_fingerprint: str
    normalizer_version: str
    g2p_version: str
    config_hash: str
    clip_count: int
    incremental: bool = False


class ScoreBreakdown(BaseModel):
    clip_id: str
    score: float
    expected_pass_probability: float
    estimated_cost: float
    coverage_gain: float
    diversity_factor: float
    matched_deficits: list[str] = Field(default_factory=list)
    speaker_penalty: float = 1.0
    rare_rescue: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class FeatureCoverageStatus(str, Enum):
    SATISFIED = "satisfied"
    DEFICIT = "deficit"
    CANDIDATE_EXHAUSTED = "candidate_exhausted"
    LIKELY_UNREACHABLE = "likely_unreachable"
    UNREACHABLE = "unreachable"
    OPTIONAL = "optional"


class FeatureCoverageRow(BaseModel):
    feature: str
    target: int
    accepted_before: int = 0
    accepted_after: int = 0
    deficit: int = 0
    candidate_total: int = 0
    candidate_remaining: int = 0
    estimated_pass_rate: float = 0.0
    expected_final_count: float = 0.0
    status: FeatureCoverageStatus = FeatureCoverageStatus.DEFICIT
    required: bool = True


class SpeakerPassStats(BaseModel):
    attempts: int = 0
    passes: int = 0

    @property
    def rate(self) -> float | None:
        if self.attempts <= 0:
            return None
        return self.passes / self.attempts


class CoverageRunState(BaseModel):
    run_id: str
    iteration: int = 0
    status: StopReason = StopReason.RUNNING
    analyzed_clip_ids: list[str] = Field(default_factory=list)
    accepted_clip_ids: list[str] = Field(default_factory=list)
    rejected_clip_ids: list[str] = Field(default_factory=list)
    current_coverage: dict[str, int] = Field(default_factory=dict)
    remaining_deficits: dict[str, int] = Field(default_factory=dict)
    speaker_pass_stats: dict[str, SpeakerPassStats] = Field(default_factory=dict)
    global_pass_attempts: int = 0
    global_pass_passes: int = 0
    analyzed_audio_sec: float = 0.0
    config_hash: str = ""
    index_fingerprint: str = ""
    started_at: str = ""
    updated_at: str = ""
    stop_detail: str | None = None
    rare_rescue_clip_ids: list[str] = Field(default_factory=list)


class AnalyzedClip(BaseModel):
    clip_id: str
    client_id: str = ""
    duration_sec: float | None = None
    row: dict[str, Any] = Field(default_factory=dict)


class RejectedClip(BaseModel):
    clip_id: str
    client_id: str = ""
    reason: str
    duration_sec: float | None = None
    row: dict[str, Any] = Field(default_factory=dict)


class SkippedClip(BaseModel):
    clip_id: str
    reason: str


class ClipError(BaseModel):
    clip_id: str
    error: str


class AnalysisBatchResult(BaseModel):
    accepted: list[AnalyzedClip] = Field(default_factory=list)
    rejected: list[RejectedClip] = Field(default_factory=list)
    skipped: list[SkippedClip] = Field(default_factory=list)
    errors: list[ClipError] = Field(default_factory=list)
