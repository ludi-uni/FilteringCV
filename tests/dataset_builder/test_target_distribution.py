from __future__ import annotations

from cv_preprocess.selection.scoring import target_distribution


def test_target_distribution_uniform_when_alpha_zero() -> None:
    pool = {"a": 10, "b": 1, "c": 100}
    result = target_distribution(pool, alpha=0.0)
    assert abs(sum(result.values()) - 1.0) < 1e-9
    assert set(result) == {"a", "b", "c"}
    for value in result.values():
        assert abs(value - 1 / 3) < 1e-9


def test_target_distribution_favors_common_with_alpha_one() -> None:
    pool = {"rare": 1, "common": 9}
    result = target_distribution(pool, alpha=1.0)
    assert result["common"] > result["rare"]
    assert abs(result["common"] - 0.9) < 1e-9
    assert abs(result["rare"] - 0.1) < 1e-9


def test_target_distribution_temperature_below_one_flattens() -> None:
    pool = {"rare": 1, "common": 99}
    hot = target_distribution(pool, alpha=1.0)
    cool = target_distribution(pool, alpha=0.5)
    assert cool["rare"] > hot["rare"]
    assert cool["common"] < hot["common"]
