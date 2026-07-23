from __future__ import annotations

import polars as pl

from cv_preprocess.compute.loader import resolve_compute_backend
from cv_preprocess.compute.polars_backend import PolarsComputeBackend
from cv_preprocess.compute.python_backend import PythonComputeBackend


def _tiny_clips() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "clip_id": "a",
                "speaker_id": "spk1",
                "phonemes": "k o N",
                "biphones": ["k-o", "o-N"],
                "triphones": ["#-k-o", "k-o-N", "o-N-#"],
                "moras": ["ko", "N"],
                "fullcontext_labels": ["a/b", "c/d"],
            },
            {
                "clip_id": "b",
                "speaker_id": "spk2",
                "phonemes": "k o N",
                "biphones": ["k-o", "o-N"],
                "triphones": ["#-k-o", "k-o-N", "o-N-#"],
                "moras": ["ko", "N"],
                "fullcontext_labels": ["a/b", "e/f"],
            },
            {
                "clip_id": "c",
                "speaker_id": "spk1",
                "phonemes": "n i",
                "biphones": ["n-i"],
                "triphones": ["#-n-i", "n-i-#"],
                "moras": ["ni"],
                "fullcontext_labels": ["g/h"],
            },
        ]
    )


def _sorted_feature_counts(df: pl.DataFrame) -> pl.DataFrame:
    return df.sort(["feature_type", "feature"])


def test_python_and_polars_feature_counts_agree() -> None:
    clips = _tiny_clips()
    python_counts = _sorted_feature_counts(PythonComputeBackend().count_features(clips))
    polars_counts = _sorted_feature_counts(PolarsComputeBackend().count_features(clips))
    assert python_counts.equals(polars_counts)


def test_auto_resolves_to_polars_when_available() -> None:
    backend = resolve_compute_backend("auto")
    assert backend.name == "polars"


def test_explicit_python_backend() -> None:
    backend = resolve_compute_backend("python")
    assert isinstance(backend, PythonComputeBackend)
