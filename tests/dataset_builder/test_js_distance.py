from __future__ import annotations

import math

import pytest

from cv_preprocess.reports.coverage import js_distance, js_divergence


def test_js_divergence_identical_distributions_is_zero() -> None:
    dist = {"a": 2.0, "b": 2.0}
    assert js_divergence(dist, dist) == pytest.approx(0.0)


def test_js_distance_is_symmetric() -> None:
    p = {"a": 1.0, "b": 3.0}
    q = {"a": 2.0, "b": 2.0, "c": 1.0}
    assert js_distance(p, q) == pytest.approx(js_distance(q, p))


def test_js_distance_from_counts_matches_normalized_probs() -> None:
    counts_a = {"x": 1, "y": 1}
    counts_b = {"x": 2, "y": 0}
    probs_a = {"x": 0.5, "y": 0.5}
    probs_b = {"x": 1.0, "y": 0.0}
    assert js_distance(counts_a, counts_b) == pytest.approx(js_distance(probs_a, probs_b))


def test_js_distance_is_bounded() -> None:
    p = {"a": 1.0}
    q = {"b": 1.0}
    assert js_distance(p, q) == pytest.approx(math.sqrt(math.log(2.0)))
