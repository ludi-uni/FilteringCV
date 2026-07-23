from __future__ import annotations

import math

from cv_preprocess.selection.scoring import feature_utility, marginal_feature_utility


def test_feature_utility_diminishing_returns() -> None:
    first = feature_utility(1.0, 1.0, tau=1.0)
    second = feature_utility(1.0, 2.0, tau=1.0)
    assert second > first
    delta_first = feature_utility(1.0, 1.0, tau=1.0) - feature_utility(1.0, 0.0, tau=1.0)
    delta_second = feature_utility(1.0, 2.0, tau=1.0) - feature_utility(1.0, 1.0, tau=1.0)
    assert delta_second < delta_first


def test_marginal_feature_utility_matches_difference() -> None:
    marginal = marginal_feature_utility(2.0, 3.0, tau=0.5)
    expected = feature_utility(2.0, 4.0, 0.5) - feature_utility(2.0, 3.0, 0.5)
    assert math.isclose(marginal, expected)


def test_zero_weight_yields_zero_utility() -> None:
    assert feature_utility(0.0, 10.0, tau=1.0) == 0.0
    assert marginal_feature_utility(0.0, 10.0, tau=1.0) == 0.0
