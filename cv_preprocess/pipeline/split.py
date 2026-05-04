from __future__ import annotations

import random
from collections import defaultdict


def assign_speaker_splits(
    speaker_ids: list[str],
    counts_by_speaker: dict[str, int],
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
) -> dict[str, str]:
    """Assign each speaker to exactly one split; greedy balance by utterance counts."""
    names = ("train", "val", "test")
    ratios = {"train": train, "val": val, "test": test}
    total_utts = sum(counts_by_speaker.get(s, 0) for s in speaker_ids)
    if total_utts == 0:
        return {}
    targets = {k: ratios[k] * total_utts for k in names}
    current = {k: 0.0 for k in names}
    rng = random.Random(seed)
    spks = list(speaker_ids)
    rng.shuffle(spks)
    out: dict[str, str] = {}
    for s in spks:
        n = float(counts_by_speaker.get(s, 0))
        # pick split with largest (target - current) deficit
        best = max(names, key=lambda k: targets[k] - current[k])
        out[s] = best
        current[best] += n
    return out


def build_counts(rows: list[tuple[str, str]]) -> dict[str, int]:
    """rows: list of (client_id, utt_key) per accepted utterance."""
    c: dict[str, int] = defaultdict(int)
    for cid, _ in rows:
        c[cid] += 1
    return dict(c)
