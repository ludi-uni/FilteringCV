from __future__ import annotations

from enum import Enum


class SplitProtocol(str, Enum):
    UNSEEN_SPEAKER = "unseen_speaker"
    SEEN_SPEAKER = "seen_speaker"
    SINGLE_SPEAKER = "single_speaker"


class SplitName(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SPLIT_ORDER: tuple[str, ...] = (SplitName.TRAIN.value, SplitName.VAL.value, SplitName.TEST.value)
