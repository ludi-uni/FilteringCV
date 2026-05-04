from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

class OutputConfig(BaseModel):
    root: Path = Path("out/cv_tts")
    format: str = "wav"
    manifest: str = "metadata.jsonl"
    validated_tsv: str = "validated.tsv"
    rejects_name: str = "rejects.csv"
    report_name: str = "report.json"
    wav_subdir: str = "wavs"
    keep_rejected_intermediate: bool = False


class SplitConfig(BaseModel):
    mode: Literal["speaker_aware", "random"] = "speaker_aware"
    train: float = 0.9
    val: float = 0.05
    test: float = 0.05
    seed: int = 42
    emit_split_as_dev: bool = False

    @field_validator("train", "val", "test")
    @classmethod
    def positive_ratios(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("split ratios must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def sum_to_one(self) -> SplitConfig:
        s = self.train + self.val + self.test
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"split.train + val + test must equal 1.0, got {s}")
        return self
