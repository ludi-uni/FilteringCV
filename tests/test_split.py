from cv_preprocess.pipeline.split import assign_speaker_splits, build_counts


def test_assign_speaker_splits_balances_counts() -> None:
    counts = {"a": 50, "b": 30, "c": 20}
    sp = list(counts.keys())
    m = assign_speaker_splits(sp, counts, train=0.9, val=0.05, test=0.05, seed=1)
    assert set(m.values()) <= {"train", "val", "test"}
    assert len(m) == 3


def test_build_counts() -> None:
    rows = [("a", "1"), ("a", "2"), ("b", "3")]
    assert build_counts(rows) == {"a": 2, "b": 1}
