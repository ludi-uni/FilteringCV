"""Acoustic diversity / redundancy penalties for coverage-aware select."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

import numpy as np

from cv_preprocess.config.dataset_builder import AcousticDiversityConfig
from cv_preprocess.selection.protocol import ClipFeatures

ACOUSTIC_FEATURE_GETTERS: dict[str, str] = {
    "duration": "duration_sec",
    "rms": "rms",
    "peak": "peak",
    "f0_median": "f0_median",
    "f0_range": "f0_range",
    "speech_rate": "speech_rate",
    "silence_ratio": "silence_ratio",
    "snr": "estimated_snr_db",
    "quality_score": "quality_score",
    "alignment_confidence": "alignment_confidence",
}


class AcousticEmbeddingBackend(Protocol):
    def extract(self, clip: ClipFeatures) -> np.ndarray:
        ...


@dataclass
class LightweightAcousticBackend:
    feature_names: list[str]
    missing_value_policy: str = "ignore"

    def raw_values(self, clip: ClipFeatures) -> dict[str, float | None]:
        metrics = dict(clip.acoustic_metrics or {})
        # Convenience aliases from ClipFeatures fields
        metrics.setdefault("duration", clip.duration_sec)
        metrics.setdefault("duration_sec", clip.duration_sec)
        if clip.quality_score is not None:
            metrics.setdefault("quality_score", float(clip.quality_score))
        out: dict[str, float | None] = {}
        for name in self.feature_names:
            attr = ACOUSTIC_FEATURE_GETTERS.get(name, name)
            value = metrics.get(name)
            if value is None:
                value = metrics.get(attr)
            if value is None:
                out[name] = None
            else:
                out[name] = float(value)
        return out

    def extract(self, clip: ClipFeatures) -> np.ndarray:
        # Per-clip extract without corpus normalization (caller should normalize).
        values = self.raw_values(clip)
        vec: list[float] = []
        for name in self.feature_names:
            raw = values.get(name)
            if raw is None:
                if self.missing_value_policy == "zero":
                    vec.append(0.0)
                else:
                    vec.append(float("nan"))
            else:
                vec.append(float(raw))
        return np.asarray(vec, dtype=np.float64)


@dataclass
class AcousticDiversityState:
    enabled: bool
    backend_name: str
    feature_names: list[str]
    weight: float
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    min_distance: dict[str, float] = field(default_factory=dict)
    selected_ids: list[str] = field(default_factory=list)
    missing_feature_clip_count: int = 0

    def redundancy_penalty(self, clip_id: str) -> float:
        if not self.enabled or self.weight <= 0:
            return 0.0
        if not self.selected_ids:
            return 0.0
        dist = self.min_distance.get(clip_id)
        if dist is None:
            return 0.0
        # Convert distance to similarity-like penalty in [0, 1]
        similarity = 1.0 / (1.0 + max(dist, 0.0))
        return self.weight * similarity

    def diversity_bonus(self, clip_id: str) -> float:
        if not self.enabled or self.weight <= 0:
            return 0.0
        if not self.selected_ids:
            return self.weight
        dist = self.min_distance.get(clip_id, 0.0)
        return self.weight * float(dist)

    def note_selected(self, clip_id: str) -> None:
        if not self.enabled:
            return
        self.selected_ids.append(clip_id)
        selected_vec = self.vectors.get(clip_id)
        if selected_vec is None:
            return
        for other_id, other_vec in self.vectors.items():
            if other_id == clip_id:
                continue
            dist = _nan_aware_distance(selected_vec, other_vec)
            prev = self.min_distance.get(other_id)
            if prev is None or dist < prev:
                self.min_distance[other_id] = dist


def _nan_aware_distance(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if not np.any(mask):
        return 0.0
    diff = a[mask] - b[mask]
    return float(np.linalg.norm(diff))


def build_acoustic_diversity_state(
    candidates: Sequence[ClipFeatures],
    config: AcousticDiversityConfig,
    *,
    feature_weight_override: float | None = None,
) -> AcousticDiversityState:
    backend_name = config.backend
    enabled = bool(config.enabled) and backend_name not in {"disabled"}
    weight = (
        float(feature_weight_override)
        if feature_weight_override is not None
        else float(config.weight)
    )
    if not enabled or weight <= 0:
        return AcousticDiversityState(
            enabled=False,
            backend_name=backend_name,
            feature_names=list(config.features),
            weight=0.0,
        )

    backend = LightweightAcousticBackend(
        feature_names=list(config.features),
        missing_value_policy=config.missing_value_policy,
    )
    raw_matrix: list[np.ndarray] = []
    clip_ids: list[str] = []
    missing = 0
    for clip in candidates:
        vec = backend.extract(clip)
        if np.isnan(vec).all():
            missing += 1
        elif np.isnan(vec).any() and config.missing_value_policy == "ignore":
            # partial missing is ok
            pass
        clip_ids.append(clip.clip_id)
        raw_matrix.append(vec)

    if not raw_matrix:
        return AcousticDiversityState(
            enabled=False,
            backend_name=backend_name,
            feature_names=list(config.features),
            weight=weight,
            missing_feature_clip_count=missing,
        )

    mat = np.vstack(raw_matrix)
    # Z-score per feature, ignoring NaNs (all-NaN columns → mean 0, std 1)
    means = np.zeros(mat.shape[1], dtype=np.float64)
    stds = np.ones(mat.shape[1], dtype=np.float64)
    for col in range(mat.shape[1]):
        values = mat[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        means[col] = float(np.mean(finite))
        col_std = float(np.std(finite))
        if col_std >= 1e-12:
            stds[col] = col_std
    normalized = (mat - means) / stds

    vectors = {cid: normalized[i] for i, cid in enumerate(clip_ids)}
    # Initialize min distances as +inf (no selected yet)
    min_distance = {cid: float("inf") for cid in clip_ids}
    return AcousticDiversityState(
        enabled=True,
        backend_name=backend_name,
        feature_names=list(config.features),
        weight=weight,
        vectors=vectors,
        min_distance=min_distance,
        missing_feature_clip_count=missing,
    )


def summarize_acoustic_diversity(
    state: AcousticDiversityState,
    selected_ids: Sequence[str],
) -> dict[str, object]:
    if not state.enabled or len(selected_ids) < 2:
        distances: list[float] = []
    else:
        distances = []
        ids = list(selected_ids)
        for i, left in enumerate(ids):
            left_vec = state.vectors.get(left)
            if left_vec is None:
                continue
            best = float("inf")
            for j, right in enumerate(ids):
                if i == j:
                    continue
                right_vec = state.vectors.get(right)
                if right_vec is None:
                    continue
                best = min(best, _nan_aware_distance(left_vec, right_vec))
            if best < float("inf"):
                distances.append(best)

    mean_nn = float(np.mean(distances)) if distances else None
    median_nn = float(np.median(distances)) if distances else None
    return {
        "enabled": state.enabled,
        "backend": state.backend_name,
        "feature_names": list(state.feature_names),
        "selected_clip_count": len(selected_ids),
        "mean_nearest_neighbor_distance": mean_nn,
        "median_nearest_neighbor_distance": median_nn,
        "missing_feature_clip_count": state.missing_feature_clip_count,
        "weight": state.weight,
    }


def resolve_acoustic_weight(
    selection_feature_weights: Mapping[str, float],
    acoustic_config: AcousticDiversityConfig,
) -> float:
    """Prefer acoustic_diversity config weight; fall back to feature_weights key."""
    if acoustic_config.enabled and acoustic_config.backend != "disabled":
        if acoustic_config.weight > 0:
            return float(acoustic_config.weight)
    return float(selection_feature_weights.get("acoustic_diversity", 0.0))
