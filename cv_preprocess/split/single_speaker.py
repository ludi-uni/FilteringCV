from __future__ import annotations

from cv_preprocess.config.dataset_builder import DatasetBuilderSplitConfig
from cv_preprocess.split.leakage import ClipSplitRecord
from cv_preprocess.split.protocol import SplitProtocol
from cv_preprocess.split.seen_speaker import assign_clip_splits


def assign_single_speaker_splits(
    clips: list[ClipSplitRecord],
    config: DatasetBuilderSplitConfig,
) -> tuple[dict[str, str], list[str]]:
    """Assign clips without speaker separation; leakage policy still applies."""
    return assign_clip_splits(
        clips,
        config,
        protocol=SplitProtocol.SINGLE_SPEAKER,
    )
