from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from cv_preprocess.config.dataset_builder import (
    DatasetBuilderConfig,
    DuplicatesConfig,
    FeatureSupportConfig,
    SelectionConfig,
    SpeakerConstraintsConfig,
)
from cv_preprocess.selection.constraints import (
    ConstraintConfig,
    ConstraintState,
    add_clip_to_state,
    can_add_clip,
)
from cv_preprocess.selection.local_search import local_search_improve
from cv_preprocess.selection.protocol import (
    ClipFeatures,
    SelectionExplanation,
    SelectionResult,
)
from cv_preprocess.selection.scoring import (
    pool_counts_for_family,
    speaker_counts_for_family,
    total_selection_score,
)


@dataclass
class PythonSelectionBackend:
    config: DatasetBuilderConfig

    def select(
        self,
        candidates: list[ClipFeatures],
        *,
        target_duration_sec: float,
        tolerance_ratio: float,
        seed: int,
    ) -> SelectionResult:
        return greedy_local_search(
            candidates,
            config=self.config,
            target_duration_sec=target_duration_sec,
            tolerance_ratio=tolerance_ratio,
            seed=seed,
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


def greedy_local_search(
    candidates: list[ClipFeatures],
    *,
    config: DatasetBuilderConfig,
    target_duration_sec: float,
    tolerance_ratio: float,
    seed: int,
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

    pool_counts_by_family = {
        family: pool_counts_for_family(eligible, family) for family in feature_weights
    }
    utterance_counts_by_family = pool_counts_by_family
    speaker_sets_by_family = {
        family: speaker_counts_for_family(eligible, family) for family in feature_weights
    }

    state = ConstraintState()
    current_counts_by_family: dict[str, Counter[str]] = {
        family: Counter() for family in feature_weights
    }
    selected: list[str] = []
    explanations: dict[str, SelectionExplanation] = {}
    forced_include = [clip for clip in eligible if clip.override_action == "force_include"]
    forced_exclude = {
        clip.clip_id
        for clip in eligible
        if clip.override_action in {"force_exclude", "return_to_reserve"}
    }

    min_duration = target_duration_sec * max(0.0, 1.0 - tolerance_ratio)
    max_duration = target_duration_sec * (1.0 + tolerance_ratio)

    def score_clip(clip: ClipFeatures) -> tuple[float, dict, dict]:
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
            quality_weight=feature_weights.get("quality", 0.0),
            speaker_diversity_weight=feature_weights.get("speaker_diversity", 0.0),
        )

    def commit_selection(clip: ClipFeatures, reason: str) -> None:
        clip_score, positive, penalties = score_clip(clip)
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
        )

    for clip in sorted(forced_include, key=lambda c: c.clip_id):
        ok, penalties = can_add_clip(clip, state, constraint_config)
        if ok and state.total_duration_sec + clip.duration_sec <= max_duration:
            commit_selection(clip, "force_include")
        else:
            explanations[clip.clip_id] = SelectionExplanation(
                selection_score=0.0,
                penalties=penalties,
                reserve_reason="force_include_blocked_by_constraints",
            )

    remaining = [
        clip for clip in eligible if clip.clip_id not in selected and clip.clip_id not in forced_exclude
    ]

    while state.total_duration_sec < min_duration and remaining:
        best_clip: ClipFeatures | None = None
        best_score = float("-inf")
        best_positive: dict = {}
        best_penalties: dict = {}

        for clip in remaining:
            if state.total_duration_sec + clip.duration_sec > max_duration:
                continue
            ok, penalties = can_add_clip(clip, state, constraint_config)
            if not ok:
                continue
            clip_score, positive, score_penalties = score_clip(clip)
            penalties.update(score_penalties)
            if clip_score > best_score or (
                clip_score == best_score and (best_clip is None or clip.clip_id < best_clip.clip_id)
            ):
                best_score = clip_score
                best_clip = clip
                best_positive = positive
                best_penalties = penalties

        if best_clip is None or best_score <= 0.0:
            break

        selected.append(best_clip.clip_id)
        add_clip_to_state(best_clip, state, constraint_config)
        for family, tokens in best_clip.features_by_family.items():
            for token in tokens:
                current_counts_by_family.setdefault(family, Counter())[token] += 1
        explanations[best_clip.clip_id] = SelectionExplanation(
            selection_score=best_score,
            positive_contributions=best_positive,
            penalties=best_penalties,
            selected_reason="greedy_marginal_utility",
        )
        remaining = [clip for clip in remaining if clip.clip_id != best_clip.clip_id]

    reserve_candidates = [clip for clip in eligible if clip.clip_id not in selected]
    reserve_candidates.sort(
        key=lambda clip: (
            -score_clip(clip)[0],
            clip.clip_id,
        )
    )
    reserve_count = int(round(len(eligible) * selection.reserve_ratio))
    reserve_ids = [clip.clip_id for clip in reserve_candidates[:reserve_count]]
    for clip in reserve_candidates:
        if clip.clip_id in reserve_ids and clip.clip_id not in explanations:
            clip_score, positive, penalties = score_clip(clip)
            explanations[clip.clip_id] = SelectionExplanation(
                selection_score=clip_score,
                positive_contributions=positive,
                penalties=penalties,
                reserve_reason="top_reserve_rank",
            )

    if selection.local_search.enabled and selected and reserve_ids:
        selected, reserve_ids, _ = local_search_improve(
            selected,
            reserve_ids,
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
        )

    for rank, clip_id in enumerate(selected, start=1):
        if clip_id in explanations:
            explanations[clip_id].rank = rank

    tail_ids = [clip.clip_id for clip in reserve_candidates if clip.clip_id not in reserve_ids]
    rng = random.Random(seed)
    rng.shuffle(tail_ids)
    reserve_ids.extend(tail_ids)

    return SelectionResult(
        selected_ids=selected,
        reserve_ids=reserve_ids,
        explanations=explanations,
    )
