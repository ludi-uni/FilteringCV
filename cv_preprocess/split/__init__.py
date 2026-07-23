from __future__ import annotations

from cv_preprocess.split.leakage import ClipSplitRecord, LeakageViolation, detect_leakage
from cv_preprocess.split.protocol import SplitProtocol
from cv_preprocess.split.seen_speaker import assign_clip_splits
from cv_preprocess.split.single_speaker import assign_single_speaker_splits
from cv_preprocess.split.unseen_speaker import clips_by_split_from_speakers, plan_unseen_speaker_splits

__all__ = [
    "ClipSplitRecord",
    "LeakageViolation",
    "SplitProtocol",
    "assign_clip_splits",
    "assign_single_speaker_splits",
    "clips_by_split_from_speakers",
    "detect_leakage",
    "plan_unseen_speaker_splits",
]
