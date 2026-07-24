"""Coverage deficit computation."""

from __future__ import annotations

from collections.abc import Mapping

from cv_preprocess.config.coverage import CoverageAutomationConfig


def compute_deficits(
    targets: Mapping[str, int],
    accepted_counts: Mapping[str, int],
) -> dict[str, int]:
    """deficit(f) = max(0, target(f) - accepted_count(f))."""
    return {
        feature: max(0, int(target) - int(accepted_counts.get(feature, 0)))
        for feature, target in targets.items()
        if int(target) > 0
    }


def remaining_required_deficits(
    config: CoverageAutomationConfig,
    deficits: Mapping[str, int],
) -> dict[str, int]:
    return {
        feature: deficit
        for feature, deficit in deficits.items()
        if deficit > 0 and config.is_required(feature)
    }


def total_deficit(deficits: Mapping[str, int]) -> int:
    return int(sum(int(v) for v in deficits.values()))
