from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from cv_preprocess.config.audio_steps import AudioPipelineConfig
from cv_preprocess.config.gates_quality import SnrEstimatorConfig

class PhonemeManifestPipelineConfig(BaseModel):
    """``cv-preprocess phoneme-manifest`` 用。OpenJTalk G2P 互換の JSONL を生成する。"""

    output_path: Path = Path("data/aligned_phonemes.jsonl")
    #: ``g2p_text`` = 正規化文から ``g2p_phonemes``（preprocess と同じ）。``mfa_textgrid`` = MFA の TextGrid の phones を読み、マップで OJ トークン列に変換。
    source: Literal["g2p_text", "mfa_textgrid"] = "g2p_text"
    #: ``source: mfa_textgrid`` 時必須。各クリップは ``{stem(path)}.TextGrid``（``path`` は TSV のファイル名）をこのディレクトリ直下に置く。
    mfa_textgrid_root: Path | None = None
    #: MFA phones トークン → OpenJTalk G2P トークン列（空白区切りで複数可）。空文字は当該 MFA トークンを出力から省略。
    mfa_token_map_path: Path | None = None


class SecondaryPipelineConfig(BaseModel):
    """一次 ``preprocess`` の metadata.jsonl と WAV を入力に、補正チェーンと再品質ゲートを適用する。"""

    output_root: Path = Path("out/cv_tts_secondary")
    #: 一次出力ディレクトリ（WAV の親。省略時は ``output.root``）
    input_root: Path | None = None
    #: metadata.jsonl。省略時は ``input_root / output.manifest``
    input_manifest: Path | None = None
    audio_pipeline: AudioPipelineConfig
    #: ``quality_gate`` への部分上書き（一次設定をベースにマージ）
    quality_gate_overrides: dict[str, Any] = Field(default_factory=dict)
    #: ``quality_gate_profiles`` のキー。指定時は上書きのさらに上にプロファイルをマージ。
    quality_gate_profile: str | None = None
    snr: SnrEstimatorConfig | None = None
    rejects_name: str = "rejects_secondary.csv"
    report_name: str = "report_secondary.json"
    wav_subdir: str = "wavs"

