"""Greedy coverage batch planning with temporary deficit updates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from cv_preprocess.config.coverage import CoverageAutomationConfig
from cv_preprocess.coverage.deficits import compute_deficits, remaining_required_deficits, total_deficit
from cv_preprocess.coverage.models import (
    ClipIndexRecord,
    FeatureCoverageRow,
    FeatureCoverageStatus,
    ScoreBreakdown,
    SpeakerPassStats,
    StopReason,
)
from cv_preprocess.coverage.scorer import score_candidate


@dataclass
class CoveragePlan:
    current_coverage: dict[str, int]
    deficits: dict[str, int]
    required_deficits: dict[str, int]
    selected: list[ClipIndexRecord]
    score_breakdowns: list[ScoreBreakdown]
    feature_rows: list[FeatureCoverageRow]
    estimated_required_analysis: int
    batch_size: int
    stop_hints: list[StopReason]
    unreachable_features: list[str]
    likely_unreachable_features: list[str]
    candidate_pool_size: int
    estimated_pass_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_coverage": self.current_coverage,
            "deficits": self.deficits,
            "required_deficits": self.required_deficits,
            "batch_size": self.batch_size,
            "estimated_required_analysis": self.estimated_required_analysis,
            "candidate_pool_size": self.candidate_pool_size,
            "estimated_pass_rate": self.estimated_pass_rate,
            "stop_hints": [s.value for s in self.stop_hints],
            "unreachable_features": self.unreachable_features,
            "likely_unreachable_features": self.likely_unreachable_features,
            "feature_rows": [row.model_dump(mode="json") for row in self.feature_rows],
            "selected": [
                {
                    **breakdown.model_dump(mode="json"),
                    "client_id": record.client_id,
                    "source_path": record.source_path,
                    "feature_keys": sorted(record.feature_key_set()),
                }
                for record, breakdown in zip(self.selected, self.score_breakdowns)
            ],
        }


def pool_feature_counts(candidates: Sequence[ClipIndexRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in candidates:
        for key in record.feature_key_set():
            counter[key] += 1
    return dict(counter)


def estimate_batch_size(
    *,
    required_deficits: Mapping[str, int],
    estimated_pass_rate: float,
    config: CoverageAutomationConfig,
    candidate_count: int,
) -> int:
    remaining = sum(int(v) for v in required_deficits.values())
    if remaining <= 0 or candidate_count <= 0:
        return 0
    rate = max(estimated_pass_rate, config.pass_probability.min_probability)
    required = math.ceil(remaining / rate * config.batch.safety_factor)
    size = max(config.batch.min_size, min(config.batch.max_size, required))
    return min(size, candidate_count)


def select_batch(
    candidates: Sequence[ClipIndexRecord],
    *,
    deficits: Mapping[str, int],
    targets: Mapping[str, int],
    config: CoverageAutomationConfig,
    speaker_stats: Mapping[str, SpeakerPassStats] | None = None,
    global_attempts: int = 0,
    global_passes: int = 0,
    batch_limit: int | None = None,
    exclude_clip_ids: set[str] | None = None,
    rare_rescue_features: set[str] | None = None,
) -> tuple[list[ClipIndexRecord], list[ScoreBreakdown]]:
    speaker_stats = speaker_stats or {}
    exclude_clip_ids = exclude_clip_ids or set()
    remaining = [c for c in candidates if c.clip_id not in exclude_clip_ids and not c.invalidated]
    pool_counts = pool_feature_counts(remaining)
    temporary = dict(deficits)
    selected: list[ClipIndexRecord] = []
    breakdowns: list[ScoreBreakdown] = []
    limit = batch_limit if batch_limit is not None else config.batch.max_size

    while len(selected) < limit and remaining:
        best: ClipIndexRecord | None = None
        best_score = ScoreBreakdown(
            clip_id="",
            score=0.0,
            expected_pass_probability=0.0,
            estimated_cost=1.0,
            coverage_gain=0.0,
            diversity_factor=0.0,
        )
        for candidate in remaining:
            scored = score_candidate(
                candidate,
                deficits=temporary,
                targets=targets,
                pool_counts=pool_counts,
                selected_batch=selected,
                speaker_stats=speaker_stats,
                global_attempts=global_attempts,
                global_passes=global_passes,
                config=config,
                rare_rescue_features=rare_rescue_features,
            )
            if scored.score > best_score.score:
                best = candidate
                best_score = scored
        if best is None or best_score.score <= 0:
            break
        selected.append(best)
        breakdowns.append(best_score)
        remaining = [c for c in remaining if c.clip_id != best.clip_id]
        for feature in best.feature_key_set():
            if feature in temporary:
                temporary[feature] = max(0, temporary[feature] - 1)
    return selected, breakdowns


def build_feature_rows(
    *,
    targets: Mapping[str, int],
    accepted_counts: Mapping[str, int],
    deficits: Mapping[str, int],
    candidates: Sequence[ClipIndexRecord],
    estimated_pass_rate: float,
    config: CoverageAutomationConfig,
) -> tuple[list[FeatureCoverageRow], list[str], list[str], list[StopReason]]:
    pool_counts = pool_feature_counts(candidates)
    rows: list[FeatureCoverageRow] = []
    unreachable: list[str] = []
    likely: list[str] = []
    hints: list[StopReason] = []

    for feature, target in targets.items():
        accepted = int(accepted_counts.get(feature, 0))
        deficit = int(deficits.get(feature, 0))
        cand_total = int(pool_counts.get(feature, 0))
        expected_final = accepted + cand_total * estimated_pass_rate
        required = config.is_required(feature)
        if deficit <= 0:
            status = FeatureCoverageStatus.SATISFIED
        elif cand_total <= 0:
            status = FeatureCoverageStatus.CANDIDATE_EXHAUSTED
            if required:
                unreachable.append(feature)
                hints.append(StopReason.CANDIDATE_EXHAUSTED)
        elif accepted + cand_total < target:
            status = FeatureCoverageStatus.UNREACHABLE
            if required:
                unreachable.append(feature)
                hints.append(StopReason.UNREACHABLE)
        elif expected_final + 1e-9 < target:
            status = FeatureCoverageStatus.LIKELY_UNREACHABLE
            if required:
                likely.append(feature)
                hints.append(StopReason.LIKELY_UNREACHABLE)
        else:
            status = FeatureCoverageStatus.DEFICIT
        if not required and deficit > 0:
            status = FeatureCoverageStatus.OPTIONAL if status == FeatureCoverageStatus.DEFICIT else status
        rows.append(
            FeatureCoverageRow(
                feature=feature,
                target=int(target),
                accepted_before=accepted,
                accepted_after=accepted,
                deficit=deficit,
                candidate_total=cand_total,
                candidate_remaining=cand_total,
                estimated_pass_rate=estimated_pass_rate,
                expected_final_count=float(expected_final),
                status=status,
                required=required,
            )
        )
    return rows, unreachable, likely, hints


def plan_coverage(
    *,
    config: CoverageAutomationConfig,
    index_records: Sequence[ClipIndexRecord],
    accepted_counts: Mapping[str, int],
    analyzed_clip_ids: set[str] | None = None,
    speaker_stats: Mapping[str, SpeakerPassStats] | None = None,
    global_attempts: int = 0,
    global_passes: int = 0,
    batch_size_override: int | None = None,
) -> CoveragePlan:
    targets = config.iter_active_targets()
    deficits = compute_deficits(targets, accepted_counts)
    required = remaining_required_deficits(config, deficits)
    analyzed_clip_ids = analyzed_clip_ids or set()

    unanalyzed = [
        record
        for record in index_records
        if not record.invalidated and record.clip_id not in analyzed_clip_ids
    ]
    # Keep candidates that can help any remaining deficit (required or optional).
    useful = [
        record
        for record in unanalyzed
        if any(deficits.get(feature, 0) > 0 for feature in record.feature_key_set())
    ]

    if global_attempts > 0:
        estimated_pass_rate = global_passes / global_attempts
    else:
        estimated_pass_rate = config.pass_probability.default

    feature_rows, unreachable, likely, hints = build_feature_rows(
        targets=targets,
        accepted_counts=accepted_counts,
        deficits=deficits,
        candidates=useful,
        estimated_pass_rate=estimated_pass_rate,
        config=config,
    )

    if total_deficit(required) <= 0:
        return CoveragePlan(
            current_coverage=dict(accepted_counts),
            deficits=deficits,
            required_deficits=required,
            selected=[],
            score_breakdowns=[],
            feature_rows=feature_rows,
            estimated_required_analysis=0,
            batch_size=0,
            stop_hints=[StopReason.COMPLETE],
            unreachable_features=unreachable,
            likely_unreachable_features=likely,
            candidate_pool_size=len(useful),
            estimated_pass_rate=estimated_pass_rate,
        )

    rare_features = set(config.rare_rescue.target_features) if config.rare_rescue.enabled else set()
    if config.rare_rescue.enabled and not rare_features:
        rare_features = set(required.keys())

    estimated_needed = estimate_batch_size(
        required_deficits=required,
        estimated_pass_rate=estimated_pass_rate,
        config=config,
        candidate_count=len(useful),
    )
    limit = batch_size_override if batch_size_override is not None else estimated_needed
    if batch_size_override is None:
        limit = estimated_needed
    else:
        limit = max(1, min(batch_size_override, len(useful)))

    selected, breakdowns = select_batch(
        useful,
        deficits=deficits,
        targets=targets,
        config=config,
        speaker_stats=speaker_stats,
        global_attempts=global_attempts,
        global_passes=global_passes,
        batch_limit=limit,
        exclude_clip_ids=analyzed_clip_ids,
        rare_rescue_features=rare_features,
    )

    stop_hints = list(dict.fromkeys(hints))
    if not useful and total_deficit(required) > 0:
        stop_hints.append(StopReason.CANDIDATE_EXHAUSTED)
    if unreachable:
        stop_hints.append(StopReason.UNREACHABLE)

    return CoveragePlan(
        current_coverage=dict(accepted_counts),
        deficits=deficits,
        required_deficits=required,
        selected=selected,
        score_breakdowns=breakdowns,
        feature_rows=feature_rows,
        estimated_required_analysis=estimated_needed,
        batch_size=len(selected),
        stop_hints=stop_hints,
        unreachable_features=unreachable,
        likely_unreachable_features=likely,
        candidate_pool_size=len(useful),
        estimated_pass_rate=estimated_pass_rate,
    )
