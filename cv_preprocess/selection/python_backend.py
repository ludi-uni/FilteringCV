from __future__ import annotations

import heapq
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from cv_preprocess.application.common import ProgressEvent, ProgressSink
from cv_preprocess.config.coverage import CoverageAutomationConfig
from cv_preprocess.config.dataset_builder import (
    DatasetBuilderConfig,
    DuplicatesConfig,
    FeatureSupportConfig,
    SelectionConfig,
    SpeakerConstraintsConfig,
)
from cv_preprocess.selection.acoustic_diversity import (
    AcousticDiversityState,
    build_acoustic_diversity_state,
    resolve_acoustic_weight,
    summarize_acoustic_diversity,
)
from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    ConstraintState,
    add_clip_to_state,
    can_add_clip,
)
from cv_preprocess.selection.coverage_audit import (
    audit_coverage,
    build_missing_feature_requests,
    write_coverage_audit_reports,
)
from cv_preprocess.selection.coverage_reservation import reserve_coverage_clips
from cv_preprocess.selection.coverage_targets import (
    SelectionCoverageConstraints,
    build_selection_coverage_constraints,
)
from cv_preprocess.selection.local_search import local_search_improve
from cv_preprocess.selection.protocol import (
    ClipFeatures,
    SelectionExplanation,
    SelectionResult,
)
from cv_preprocess.selection.scoring import (
    dedupe_clip_features,
    pool_counts_for_family,
    precompute_family_score_tables,
    score_clip_fast,
    speaker_counts_for_family,
    total_selection_score,
)


@dataclass
class PythonSelectionBackend:
    config: DatasetBuilderConfig
    coverage_config: CoverageAutomationConfig | None = None
    index_candidate_counts: dict[str, int] | None = None
    coverage_audit_output: Path | None = None

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
    ) -> SelectionResult:
        return greedy_local_search(
            candidates,
            config=self.config,
            target_duration_sec=target_duration_sec,
            tolerance_ratio=tolerance_ratio,
            seed=seed,
            progress=progress,
            progress_label=progress_label,
            initial_selected=initial_selected,
            coverage_config=self.coverage_config,
            index_candidate_counts=self.index_candidate_counts,
            coverage_audit_output=self.coverage_audit_output,
        )


def _constraint_config(
    selection: SelectionConfig,
    speakers: SpeakerConstraintsConfig,
    duplicates: DuplicatesConfig,
) -> ConstraintConfig:
    low_threshold = selection.quality.preferred_score
    duplicate_limits = {
        "exact_audio": duplicates.exact_audio.max_selected
        if duplicates.exact_audio.enabled
        else None,
        "same_source_path": duplicates.same_source_path.max_selected
        if duplicates.same_source_path.enabled
        else None,
        "same_sentence_id": duplicates.same_sentence_id.max_selected
        if duplicates.same_sentence_id.enabled
        else None,
        "same_normalized_text": duplicates.same_normalized_text.max_selected
        if duplicates.same_normalized_text.enabled
        else None,
        "same_speaker_same_text": duplicates.same_speaker_same_text.max_selected
        if duplicates.same_speaker_same_text.enabled
        else None,
        "near_duplicate_text": duplicates.near_duplicate_text.max_selected
        if duplicates.near_duplicate_text.enabled
        else None,
    }
    return ConstraintConfig(
        max_clips_per_speaker=speakers.max_clips_per_speaker,
        max_duration_sec_per_speaker=speakers.max_duration_sec_per_speaker,
        min_duration_sec_per_speaker=(
            speakers.min_duration_minutes * 60.0
            if speakers.min_duration_minutes is not None
            else None
        ),
        duplicate_max_selected={
            key: value for key, value in duplicate_limits.items() if value is not None
        },
        hard_min_quality=selection.quality.hard_min_score,
        preferred_quality=selection.quality.preferred_score,
        max_low_quality_ratio=selection.quality.max_low_quality_ratio,
        low_quality_threshold=low_threshold,
    )


def _temperatures(config: DatasetBuilderConfig) -> dict[str, float]:
    temp = config.distribution_temperature
    return {
        "phone": temp.phone,
        "biphone": temp.biphone,
        "triphone": temp.triphone,
        "mora": temp.mora,
        "mora_bigram": temp.mora_bigram,
        "full_context": temp.full_context,
        "accent_nucleus": temp.accent_nucleus,
        "accent_phrase_length": temp.accent_phrase_length,
        "pause_boundary": temp.pause_boundary,
        "sentence_length_band": temp.sentence_length_band,
        "speaking_rate_band": temp.speaking_rate_band,
        "interrogative_declarative": temp.interrogative_declarative,
    }


def _support_thresholds(
    feature_support: FeatureSupportConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    return dict(feature_support.min_utterances), dict(feature_support.min_speakers)


def _eligible_candidates(candidates: list[ClipFeatures]) -> list[ClipFeatures]:
    return [clip for clip in candidates if clip.override_action != "hard_reject"]


def _emit_progress(
    progress: ProgressSink | None,
    *,
    phase: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
    fraction: float | None = None,
    label: str | None = None,
    **metadata: object,
) -> None:
    if progress is None:
        return
    meta: dict[str, object] = {"phase": phase, **metadata}
    if label:
        meta["label"] = label
        message = f"[{label}] {message}"
    if fraction is not None:
        fraction = max(0.0, min(1.0, float(fraction)))
    progress(
        ProgressEvent(
            stage="select",
            message=message,
            current=current,
            total=total,
            fraction=fraction,
            metadata=meta,
        )
    )


def greedy_local_search(
    candidates: list[ClipFeatures],
    *,
    config: DatasetBuilderConfig,
    target_duration_sec: float,
    tolerance_ratio: float,
    seed: int,
    progress: ProgressSink | None = None,
    progress_label: str | None = None,
    initial_selected: list[str] | None = None,
    coverage_config: CoverageAutomationConfig | None = None,
    index_candidate_counts: dict[str, int] | None = None,
    coverage_audit_output: Path | None = None,
) -> SelectionResult:
    selection = config.selection
    feature_weights = dict(selection.feature_weights)
    diminishing_tau = dict(selection.diminishing_return_tau)
    for family in feature_weights:
        diminishing_tau.setdefault(family, 1.0)

    temperatures = _temperatures(config)
    min_utterances, min_speakers = _support_thresholds(config.feature_support)
    constraint_config = _constraint_config(selection, config.speaker_constraints, config.duplicates)

    eligible = _eligible_candidates(candidates)
    clips_by_id = {clip.clip_id: clip for clip in eligible}
    target_hours = target_duration_sec / 3600.0

    _emit_progress(
        progress,
        phase="prepare",
        message=f"preparing {len(eligible)} eligible candidates (target {target_hours:.2f}h)",
        current=0,
        total=max(1, len(eligible)),
        fraction=0.0,
        label=progress_label,
        eligible=len(eligible),
        target_duration_sec=target_duration_sec,
    )

    pool_counts_by_family = {
        family: pool_counts_for_family(eligible, family) for family in feature_weights
    }
    utterance_counts_by_family = pool_counts_by_family
    speaker_sets_by_family = {
        family: speaker_counts_for_family(eligible, family) for family in feature_weights
    }
    target_tables = precompute_family_score_tables(
        feature_weights=feature_weights,
        temperatures=temperatures,
        pool_counts_by_family=pool_counts_by_family,
        utterance_counts_by_family=utterance_counts_by_family,
        speaker_sets_by_family=speaker_sets_by_family,
        min_utterances_by_family=min_utterances,
        min_speakers_by_family=min_speakers,
    )

    state = ConstraintState()
    current_counts_by_family: dict[str, Counter[str]] = {
        family: Counter() for family in feature_weights
    }
    selected: list[str] = []
    explanations: dict[str, SelectionExplanation] = {}
    contribution_rows: list[dict] = []
    warnings: list[str] = []
    forced_include = [clip for clip in eligible if clip.override_action == "force_include"]
    forced_exclude = {
        clip.clip_id
        for clip in eligible
        if clip.override_action in {"force_exclude", "return_to_reserve"}
    }

    min_duration = target_duration_sec * max(0.0, 1.0 - tolerance_ratio)
    max_duration = target_duration_sec * (1.0 + tolerance_ratio)
    last_progress_at = 0.0
    greedy_steps = 0
    quality_weight = feature_weights.get("quality", 0.0)
    speaker_diversity_weight = feature_weights.get("speaker_diversity", 0.0)
    # Pre-dedupe tokens once; scoring is O(features) and called millions of times.
    features_cache = {clip.clip_id: dedupe_clip_features(clip) for clip in eligible}

    coverage_constraints: SelectionCoverageConstraints | None = None
    required_coverage_targets: dict[str, int] = {}
    conflict_features: set[str] = set()
    if (
        selection.coverage_constraints.enabled
        and coverage_config is not None
        and coverage_config.features
    ):
        coverage_constraints = build_selection_coverage_constraints(
            coverage_config,
            selection,
            eligible,
            index_candidate_counts=index_candidate_counts,
        )
        required_coverage_targets = coverage_constraints.effective_required_minimums()

    acoustic_weight = resolve_acoustic_weight(feature_weights, selection.acoustic_diversity)
    acoustic_cfg = selection.acoustic_diversity
    # Disable acoustic backend when weight is 0 even if enabled flag is set via feature_weights only
    acoustic_enabled = bool(acoustic_cfg.enabled) or acoustic_weight > 0
    if acoustic_enabled and acoustic_cfg.backend != "disabled" and acoustic_weight > 0:
        # Avoid mutating pydantic model; pass weight override
        acoustic_state = build_acoustic_diversity_state(
            eligible,
            acoustic_cfg.model_copy(update={"enabled": True, "weight": acoustic_weight}),
        )
    else:
        acoustic_state = AcousticDiversityState(
            enabled=False,
            backend_name=acoustic_cfg.backend,
            feature_names=list(acoustic_cfg.features),
            weight=0.0,
        )

    def report_greedy(*, force: bool = False, remaining_n: int = 0) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        if not force and (now - last_progress_at) < 0.5:
            return
        last_progress_at = now
        duration_frac = (
            state.total_duration_sec / target_duration_sec if target_duration_sec > 0 else 0.0
        )
        _emit_progress(
            progress,
            phase="greedy",
            message=(
                f"greedy selected={len(selected)} "
                f"duration={state.total_duration_sec / 3600.0:.3f}h / {target_hours:.2f}h "
                f"remaining={remaining_n}"
            ),
            current=len(selected),
            total=max(len(selected) + 1, len(eligible)),
            fraction=min(0.8, duration_frac * 0.8),
            label=progress_label,
            selected=len(selected),
            duration_sec=state.total_duration_sec,
            target_duration_sec=target_duration_sec,
            greedy_steps=greedy_steps,
        )

    def explain_clip(clip: ClipFeatures) -> tuple[float, dict, dict]:
        return total_selection_score(
            clip,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            temperatures=temperatures,
            current_counts_by_family=current_counts_by_family,
            pool_counts_by_family=pool_counts_by_family,
            utterance_counts_by_family=utterance_counts_by_family,
            speaker_sets_by_family=speaker_sets_by_family,
            min_utterances_by_family=min_utterances,
            min_speakers_by_family=min_speakers,
            selected_speakers=state.selected_speakers,
            quality_weight=quality_weight,
            speaker_diversity_weight=speaker_diversity_weight,
            target_tables=target_tables,
        )

    def score_only(clip: ClipFeatures, *, ignore_acoustic: bool = False) -> float:
        base = score_clip_fast(
            clip,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            target_tables=target_tables,
            current_counts_by_family=current_counts_by_family,
            selected_speakers=state.selected_speakers,
            quality_weight=quality_weight,
            speaker_diversity_weight=speaker_diversity_weight,
            features_by_family=features_cache.get(clip.clip_id),
        )
        if ignore_acoustic or not acoustic_state.enabled:
            return base
        # Lower priority than required coverage: acoustic is a soft penalty only.
        return base - acoustic_state.redundancy_penalty(clip.clip_id)

    def commit_selection(
        clip: ClipFeatures,
        reason: str,
        *,
        phase: str | None = None,
        coverage_contribs: list[str] | None = None,
    ) -> None:
        clip_score, positive, penalties = explain_clip(clip)
        if acoustic_state.enabled:
            penalties = dict(penalties)
            penalties["acoustic_redundancy"] = acoustic_state.redundancy_penalty(clip.clip_id)
            clip_score -= penalties["acoustic_redundancy"]
        selected.append(clip.clip_id)
        add_clip_to_state(clip, state, constraint_config)
        for family, tokens in clip.features_by_family.items():
            for token in tokens:
                current_counts_by_family.setdefault(family, Counter())[token] += 1
        explanations[clip.clip_id] = SelectionExplanation(
            selection_score=clip_score,
            positive_contributions=positive,
            penalties=penalties,
            selected_reason=reason,
            selection_phase=phase,
            coverage_contributions=list(coverage_contribs or []),
        )
        acoustic_state.note_selected(clip.clip_id)

    for clip in sorted(forced_include, key=lambda c: c.clip_id):
        ok, penalties = can_add_clip(clip, state, constraint_config)
        if ok and state.total_duration_sec + clip.duration_sec <= max_duration:
            commit_selection(clip, "force_include", phase="force_include")
        else:
            explanations[clip.clip_id] = SelectionExplanation(
                selection_score=0.0,
                penalties=penalties,
                reserve_reason="force_include_blocked_by_constraints",
            )

    # Seed with caller-provided initial selection (e.g. external reservation).
    for clip_id in initial_selected or []:
        if clip_id in selected or clip_id in forced_exclude:
            continue
        clip = clips_by_id.get(clip_id)
        if clip is None:
            continue
        ok, penalties = can_add_clip(clip, state, constraint_config)
        if ok and state.total_duration_sec + clip.duration_sec <= max_duration:
            commit_selection(clip, "initial_selected", phase="initial_selected")
        else:
            explanations[clip_id] = SelectionExplanation(
                selection_score=0.0,
                penalties=penalties,
                reserve_reason="initial_selected_blocked_by_constraints",
            )

    # Phase A: coverage reservation (required minima). Acoustic penalty ignored.
    if coverage_constraints is not None and coverage_constraints.enabled:
        _emit_progress(
            progress,
            phase="coverage_reservation",
            message="reserving clips for required coverage minima",
            current=len(selected),
            total=max(len(eligible), 1),
            fraction=0.05,
            label=progress_label,
        )
        seed_counts: dict[str, int] = {key: 0 for key in coverage_constraints.targets}
        for clip_id in selected:
            for key in coverage_constraints.clip_coverage_keys.get(clip_id, set()):
                if key in seed_counts:
                    seed_counts[key] += 1
        reservation = reserve_coverage_clips(
            eligible,
            coverage_constraints,
            constraint_config=constraint_config,
            initial_state=state,
            initial_selected_counts=seed_counts,
            max_duration_sec=max_duration,
            forced_exclude=forced_exclude | set(selected),
        )
        conflict_features = set(reservation.conflict_features)
        for record in reservation.records:
            if record.clip_id in selected:
                continue
            clip = clips_by_id[record.clip_id]
            contrib_features = [c.feature for c in record.coverage_contributions]
            commit_selection(
                clip,
                "coverage_reserved",
                phase="coverage_reservation",
                coverage_contribs=contrib_features,
            )
            contribution_rows.append(
                {
                    "clip_id": record.clip_id,
                    "selection_phase": "coverage_reservation",
                    "coverage_contributions": [asdict(c) for c in record.coverage_contributions],
                    "quality_score": record.quality_score,
                    "duration_sec": record.duration_sec,
                    "speaker_id": record.speaker_id,
                }
            )
        if reservation.conflict_features:
            warnings.append(
                "selection_constraint_conflict features: "
                + ", ".join(sorted(reservation.conflict_features))
            )

    remaining_ids = {
        clip.clip_id
        for clip in eligible
        if clip.clip_id not in selected and clip.clip_id not in forced_exclude
    }

    # Lazy greedy heap: (-score, clip_id). Scores can be stale after commits;
    # revalidate top entries instead of rescanning all candidates every step.
    heap: list[tuple[float, str]] = []
    for clip_id in remaining_ids:
        clip = clips_by_id[clip_id]
        heapq.heappush(heap, (-score_only(clip), clip_id))

    report_greedy(force=True, remaining_n=len(remaining_ids))
    rebuild_every = 250 if len(remaining_ids) > 5000 else 100
    steps_since_rebuild = 0

    while state.total_duration_sec < min_duration and remaining_ids and heap:
        best_clip: ClipFeatures | None = None
        best_score = float("-inf")
        checked = 0
        max_rechecks = min(128, max(16, len(remaining_ids) // 500 + 16))
        batch_best: ClipFeatures | None = None
        batch_best_score = float("-inf")

        while heap and checked < max_rechecks:
            neg_score, clip_id = heapq.heappop(heap)
            if clip_id not in remaining_ids:
                continue
            clip = clips_by_id[clip_id]
            if state.total_duration_sec + clip.duration_sec > max_duration:
                remaining_ids.discard(clip_id)
                continue
            ok, penalties = can_add_clip(clip, state, constraint_config)
            if not ok:
                if "max_low_quality_ratio" in penalties:
                    heapq.heappush(heap, (0.0, clip_id))
                else:
                    remaining_ids.discard(clip_id)
                continue
            fresh = score_only(clip)
            checked += 1
            if fresh > batch_best_score or (
                fresh == batch_best_score
                and (batch_best is None or clip.clip_id < batch_best.clip_id)
            ):
                batch_best_score = fresh
                batch_best = clip
            # Stale heap entry: reinsert and keep scanning the batch.
            if fresh + 1e-12 < -neg_score:
                heapq.heappush(heap, (-fresh, clip_id))
                continue
            best_clip = clip
            best_score = fresh
            break

        if best_clip is None:
            # Take the best among revalidated batch instead of O(n) full scan.
            best_clip = batch_best
            best_score = batch_best_score

        if best_clip is None or best_score <= 0.0:
            if not remaining_ids:
                break
            # Rare fallback: one full scan when the heap is empty/exhausted.
            best_clip = None
            best_score = float("-inf")
            for clip_id in list(remaining_ids):
                clip = clips_by_id[clip_id]
                if state.total_duration_sec + clip.duration_sec > max_duration:
                    continue
                ok, _ = can_add_clip(clip, state, constraint_config)
                if not ok:
                    continue
                fresh = score_only(clip)
                if fresh > best_score or (
                    fresh == best_score and (best_clip is None or clip.clip_id < best_clip.clip_id)
                ):
                    best_score = fresh
                    best_clip = clip
            if best_clip is None or best_score <= 0.0:
                break
            # Rebuild heap from remaining after fallback.
            heap = [(-score_only(clips_by_id[cid]), cid) for cid in remaining_ids]
            heapq.heapify(heap)
            steps_since_rebuild = 0

        commit_selection(best_clip, "greedy_marginal_utility", phase="normal_fill")
        contribution_rows.append(
            {
                "clip_id": best_clip.clip_id,
                "selection_phase": "normal_fill",
                "coverage_contributions": [],
                "quality_score": best_clip.quality_score,
                "duration_sec": best_clip.duration_sec,
                "speaker_id": best_clip.speaker_id,
            }
        )
        remaining_ids.discard(best_clip.clip_id)
        greedy_steps += 1
        steps_since_rebuild += 1

        if steps_since_rebuild >= rebuild_every:
            heap = [(-score_only(clips_by_id[cid]), cid) for cid in remaining_ids]
            heapq.heapify(heap)
            steps_since_rebuild = 0

        report_greedy(remaining_n=len(remaining_ids))

    report_greedy(force=True, remaining_n=len(remaining_ids))

    _emit_progress(
        progress,
        phase="reserve",
        message="ranking reserve candidates",
        current=len(selected),
        total=max(len(eligible), 1),
        fraction=0.82,
        label=progress_label,
        selected=len(selected),
    )

    selected_set = set(selected)
    reserve_candidates = [clip for clip in eligible if clip.clip_id not in selected_set]
    scored_reserve = [(score_only(clip), clip.clip_id, clip) for clip in reserve_candidates]
    scored_reserve.sort(key=lambda item: (-item[0], item[1]))
    reserve_count = int(round(len(eligible) * selection.reserve_ratio))
    reserve_ids = [clip_id for _, clip_id, _ in scored_reserve[:reserve_count]]
    reserve_id_set = set(reserve_ids)
    for score, clip_id, clip in scored_reserve[:reserve_count]:
        if clip_id not in explanations:
            # Lightweight explanation for reserve top-k only.
            explanations[clip_id] = SelectionExplanation(
                selection_score=score,
                reserve_reason="top_reserve_rank",
            )

    # Local search only over a bounded reserve pool for large corpora.
    ls_reserve_ids = reserve_ids
    max_ls_reserve = 2000
    if len(ls_reserve_ids) > max_ls_reserve:
        ls_reserve_ids = ls_reserve_ids[:max_ls_reserve]

    ls_required = (
        required_coverage_targets
        if (
            coverage_constraints is not None
            and coverage_constraints.preserve_during_local_search
        )
        else None
    )

    if selection.local_search.enabled and selected and ls_reserve_ids:
        selected, ls_reserve_ids, _ = local_search_improve(
            selected,
            ls_reserve_ids,
            clips_by_id,
            feature_weights=feature_weights,
            diminishing_tau=diminishing_tau,
            temperatures=temperatures,
            candidates=eligible,
            min_utterances_by_family=min_utterances,
            min_speakers_by_family=min_speakers,
            constraint_config=constraint_config,
            swap_patterns=selection.local_search.swap_patterns,
            max_iterations=selection.local_search.max_iterations,
            max_wall_sec=selection.local_search.max_wall_sec,
            progress=progress,
            progress_label=progress_label,
            target_tables=target_tables,
            seed=seed,
            required_coverage_targets=ls_required,
        )
        # Merge local-search reserve head back into full reserve ordering.
        ls_set = set(ls_reserve_ids)
        reserve_ids = ls_reserve_ids + [cid for cid in reserve_ids if cid not in ls_set]

    for rank, clip_id in enumerate(selected, start=1):
        if clip_id in explanations:
            explanations[clip_id].rank = rank

    tail_ids = [
        clip_id
        for _, clip_id, _ in scored_reserve
        if clip_id not in reserve_id_set and clip_id not in set(selected)
    ]
    # After local search, selected may have changed; rebuild reserve excluding selected.
    selected_set = set(selected)
    reserve_ids = [cid for cid in reserve_ids if cid not in selected_set]
    seen_reserve = set(reserve_ids)
    for clip_id in tail_ids:
        if clip_id in selected_set or clip_id in seen_reserve:
            continue
        reserve_ids.append(clip_id)
        seen_reserve.add(clip_id)

    rng = random.Random(seed)
    # Keep top reserve_count stable; shuffle only the long tail beyond that.
    head = reserve_ids[:reserve_count]
    tail = reserve_ids[reserve_count:]
    rng.shuffle(tail)
    reserve_ids = head + tail

    # Ensure contribution rows exist for all selected (including post-LS moves).
    contrib_ids = {row["clip_id"] for row in contribution_rows}
    for clip_id in selected:
        if clip_id in contrib_ids:
            continue
        clip = clips_by_id.get(clip_id)
        explanation = explanations.get(clip_id)
        phase = (
            getattr(explanation, "selection_phase", None)
            or getattr(explanation, "selected_reason", None)
            or "normal_fill"
        )
        contribution_rows.append(
            {
                "clip_id": clip_id,
                "selection_phase": phase,
                "coverage_contributions": [],
                "quality_score": clip.quality_score if clip else None,
                "duration_sec": clip.duration_sec if clip else None,
                "speaker_id": clip.speaker_id if clip else None,
            }
        )

    coverage_audit_payload = None
    missing_payload: list[dict] = []
    acoustic_summary = summarize_acoustic_diversity(acoustic_state, selected)
    report_paths: dict[str, str] = {}

    if coverage_constraints is not None and coverage_constraints.enabled:
        audit = audit_coverage(
            coverage_constraints,
            selected,
            clips_by_id,
            conflict_features=conflict_features,
        )
        warnings.extend(audit.warnings)
        missing = build_missing_feature_requests(audit)
        missing_payload = [asdict(m) for m in missing]
        coverage_audit_payload = {
            "summary": audit.summary,
            "warnings": audit.warnings,
            "features": [asdict(r) for r in audit.rows],
        }
        out_dir = coverage_audit_output
        if out_dir is None:
            out_dir = Path(config.work_dir) / "reports" / "selection"
        paths = write_coverage_audit_reports(
            out_dir,
            audit,
            contributions=contribution_rows,
            acoustic_summary=acoustic_summary,
            missing=missing,
        )
        report_paths = {key: str(path) for key, path in paths.items()}
        if audit.should_fail:
            raise RuntimeError(
                "coverage-aware select failed required coverage minima "
                f"(policy={coverage_constraints.violation_policy}); "
                f"see {out_dir / 'coverage-audit.json'}"
            )

    _emit_progress(
        progress,
        phase="done",
        message=f"selection finished selected={len(selected)} reserve={len(reserve_ids)}",
        current=len(selected),
        total=max(len(eligible), 1),
        fraction=1.0,
        label=progress_label,
        selected=len(selected),
        reserve=len(reserve_ids),
    )

    return SelectionResult(
        selected_ids=selected,
        reserve_ids=reserve_ids,
        explanations=explanations,
        coverage_audit=coverage_audit_payload,
        coverage_contributions=contribution_rows,
        acoustic_summary=acoustic_summary,
        missing_features=missing_payload,
        warnings=warnings,
        coverage_report_paths=report_paths,
    )
