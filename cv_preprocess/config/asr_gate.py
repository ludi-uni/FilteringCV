from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class AsrGateConfig(BaseModel):
    """ASR 逆認識による参照文との整合ゲート（``preprocess`` 内、MFA/NFA 通過後の波形を対象にしうる）。"""

    enabled: bool = False
    #: ``mock`` — テスト・CI 用。``nemo_transcribe`` — NeMo ASR（``asr_python`` 推奨: NFA と同じ venv）。
    backend: Literal["mock", "nemo_transcribe"] = "mock"
    #: NeMo チェックポイント名（``backend=nemo_transcribe`` 時）。``model_path`` と同時指定不可。
    pretrained_name: str | None = "nvidia/parakeet-tdt_ctc-0.6b-ja"
    model_path: Path | None = None
    sample_rate_hz: int = 16000

    min_asr_confidence: float | None = None
    max_char_error_rate: float = 0.12
    max_phoneme_error_rate: float = 0.10

    compare_text: bool = True
    compare_phonemes: bool = True
    #: 通過後に ``phonemes`` を仮説文の G2P で上書き（音声整合優先）。
    use_hypothesis_phonemes: bool = True
    #: ``use_hypothesis_phonemes`` 時、``text_norm`` も正規化済み仮説で上書きする。
    sync_text_norm_to_hypothesis: bool = True

    normalize_reference_text: bool = True
    normalize_hypothesis_text: bool = True
    #: 現状は ``normalize_for_tts`` のみ（将来拡張用キー）。
    text_normalizer: Literal["normalize_for_tts"] = "normalize_for_tts"

    persistent_worker: bool = True
    batch_size: int = 8
    timeout_sec: float = 3600.0
    #: 子プロセスの Python（未指定時は ``ASR_PYTHON`` → ``NFA_PYTHON`` → ``python3``）。
    asr_python: str | None = None
    #: 環境変数 ``CV_PREPROCESS_ASR_SUBPROCESS=1`` のとき常駐ワーカーを使わずバッチ毎起動（NFA と同様の逃げ道）。
    use_local_attention: bool = True

    missing_transcript: Literal["reject", "skip"] = "reject"
    decode_failure: Literal["reject", "skip"] = "reject"

    #: ``backend=mock`` のみ。``echo`` — 仮説=参照で常に合格。``mismatch_char`` — 末尾に記号を付け CER 増加。``empty`` — 空仮説。
    mock_mode: Literal["echo", "mismatch_char", "empty"] = "echo"

    #: 仮説長が参照長のこの倍を超えたら ``asr_duration_outlier``（``null`` で無効）。
    max_hypothesis_len_ratio: float | None = 3.0
    #: 参照が非空なのに仮説がこの文字数未満なら ``asr_duration_outlier``（``null`` で無効）。
    min_hypothesis_chars: int | None = 1

    @model_validator(mode="before")
    @classmethod
    def normalize_yaml_aliases(cls, data: Any) -> Any:
        """仕様ドラフトや旧キーとの互換（``backend: parakeet`` / ``text_normalizer: openjtalk_like`` / ``model_name``）。"""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        b = d.get("backend")
        if isinstance(b, str) and b.strip().lower() in ("parakeet", "parakeet_nemo"):
            d["backend"] = "nemo_transcribe"
        tn = d.get("text_normalizer")
        if isinstance(tn, str) and tn.strip().lower() in ("openjtalk_like", "openjtalk-like"):
            d["text_normalizer"] = "normalize_for_tts"
        mn = d.get("model_name")
        if mn not in (None, ""):
            pr = d.get("pretrained_name")
            if pr in (None, "") or (isinstance(pr, str) and not str(pr).strip()):
                d["pretrained_name"] = str(mn).strip()
        d.pop("model_name", None)
        d.pop("num_workers", None)
        return d

    @field_validator("pretrained_name", mode="before")
    @classmethod
    def pretrained_empty_as_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("batch_size")
    @classmethod
    def batch_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("asr_gate.batch_size must be >= 1")
        return v

    @field_validator("sample_rate_hz")
    @classmethod
    def sr_range(cls, v: int) -> int:
        if v < 4000 or v > 192000:
            raise ValueError("asr_gate.sample_rate_hz must be in [4000, 192000]")
        return v

    @field_validator("max_char_error_rate", "max_phoneme_error_rate")
    @classmethod
    def rates(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("asr_gate max_*_error_rate must be in [0, 1]")
        return v

    @field_validator("min_asr_confidence")
    @classmethod
    def conf_range(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0 or v > 1:
            raise ValueError("asr_gate.min_asr_confidence must be in [0, 1] or null")
        return v

    @model_validator(mode="after")
    def nemo_model_xor(self) -> AsrGateConfig:
        if not self.enabled or self.backend != "nemo_transcribe":
            return self
        has_pre = self.pretrained_name is not None and bool(str(self.pretrained_name).strip())
        has_path = self.model_path is not None
        if has_pre == has_path:
            raise ValueError(
                "asr_gate.backend=nemo_transcribe のとき、pretrained_name と model_path のどちらか一方だけを指定してください "
                "(ローカルのみなら pretrained_name: null と model_path)"
            )
        return self

    @model_validator(mode="after")
    def compare_flags(self) -> AsrGateConfig:
        if not self.enabled:
            return self
        if not self.compare_text and not self.compare_phonemes:
            raise ValueError("asr_gate.compare_text と compare_phonemes のどちらか一方は true にしてください")
        return self

    @model_validator(mode="after")
    def min_confidence_requires_mock(self) -> AsrGateConfig:
        if (
            self.enabled
            and self.backend == "nemo_transcribe"
            and self.min_asr_confidence is not None
        ):
            raise ValueError(
                "asr_gate.min_asr_confidence は backend=nemo_transcribe では未対応です "
                "(NeMo ワーカーは信頼度を返しません)。null にするか backend=mock を使ってください。"
            )
        return self


def resolve_asr_python(cfg: AsrGateConfig) -> str:
    if cfg.asr_python:
        return cfg.asr_python
    return os.environ.get("ASR_PYTHON") or os.environ.get("NFA_PYTHON", "python3")
