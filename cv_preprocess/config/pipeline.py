from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from cv_preprocess.config.align_gates import MfaGateConfig, NfaGateConfig
from cv_preprocess.config.asr_gate import AsrGateConfig
from cv_preprocess.config.audio_steps import AudioPipelineConfig
from cv_preprocess.config.aux_pipelines import PhonemeManifestPipelineConfig, SecondaryPipelineConfig
from cv_preprocess.config.gates_quality import (
    EarlyAudioGateConfig,
    QualityGateConfig,
    SnrEstimatorConfig,
    TwoPassDenoiseConfig,
)
from cv_preprocess.config.input import InputConfig, SpeakersConfig
from cv_preprocess.config.output_split import OutputConfig, SplitConfig
from cv_preprocess.config.text import TextConfig

class PipelineConfig(BaseModel):
    input: InputConfig
    speakers: SpeakersConfig = Field(default_factory=SpeakersConfig)
    audio_pipeline: AudioPipelineConfig = Field(default_factory=AudioPipelineConfig)
    #: **二段 denoise 完全分離**（``two_pass_denoise.enabled: true`` かつ両方指定時のみ有効）:
    #: MFA/NFA 足切り**前**までのチェーン。``denoise`` を含めない想定。
    audio_pipeline_align: AudioPipelineConfig | None = None
    #: 足切り通過後にだけ適用するチェーン（``denoise``・正規化・最終 trim 等）。``save_wav`` 可（ビット深度メタ用）。
    audio_pipeline_enhance: AudioPipelineConfig | None = None
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    #: 設定時は ``quality_gate_profiles[name]`` を先に適用し、その上に ``quality_gate`` をマージする。
    quality_gate_profile: str | None = None
    quality_gate_profiles: dict[str, dict[str, Any]] = Field(default_factory=dict)
    snr: SnrEstimatorConfig = Field(default_factory=SnrEstimatorConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    mfa_gate: MfaGateConfig = Field(default_factory=MfaGateConfig)
    nfa_gate: NfaGateConfig = Field(default_factory=NfaGateConfig)
    asr_gate: AsrGateConfig = Field(default_factory=AsrGateConfig)
    early_audio_gate: EarlyAudioGateConfig = Field(default_factory=EarlyAudioGateConfig)
    two_pass_denoise: TwoPassDenoiseConfig = Field(default_factory=TwoPassDenoiseConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    secondary: SecondaryPipelineConfig | None = None
    phoneme_manifest: PhonemeManifestPipelineConfig | None = None

    @model_validator(mode="before")
    @classmethod
    def merge_quality_gate_profile(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        prof = data.get("quality_gate_profile")
        if not prof:
            return data
        profiles = data.get("quality_gate_profiles") or {}
        if prof not in profiles:
            raise ValueError(
                f"quality_gate_profile {prof!r} not found in quality_gate_profiles "
                f"(keys: {list(profiles)})"
            )
        base_prof = profiles[prof]
        if not isinstance(base_prof, dict):
            raise ValueError(f"quality_gate_profiles[{prof!r}] must be a mapping")
        qg_user = dict(data.get("quality_gate") or {})
        data["quality_gate"] = {**base_prof, **qg_user}
        return data

    @model_validator(mode="after")
    def mfa_nfa_exclusive(self) -> PipelineConfig:
        if self.mfa_gate.enabled and self.nfa_gate.enabled:
            raise ValueError("mfa_gate.enabled と nfa_gate.enabled は同時に true にできません")
        return self

    @model_validator(mode="after")
    def split_align_enhance_pipelines(self) -> PipelineConfig:
        a, e = self.audio_pipeline_align, self.audio_pipeline_enhance
        if (a is None) ^ (e is None):
            raise ValueError(
                "audio_pipeline_align と audio_pipeline_enhance はどちらも null か、両方指定してください "
                "（片方だけは不可）。完全分離は two_pass_denoise.enabled: true と組み合わせます。"
            )
        if a is not None and e is not None:
            if not self.two_pass_denoise.enabled:
                raise ValueError(
                    "audio_pipeline_align / audio_pipeline_enhance を使う場合は "
                    "two_pass_denoise.enabled: true が必要です。"
                )
            if int(a.target_sample_rate) != int(e.target_sample_rate):
                raise ValueError(
                    "audio_pipeline_align.target_sample_rate と "
                    "audio_pipeline_enhance.target_sample_rate を一致させてください。"
                )
        return self

    @model_validator(mode="after")
    def nfa_model_xor(self) -> PipelineConfig:
        if not self.nfa_gate.enabled:
            return self
        ng = self.nfa_gate
        has_pre = ng.pretrained_name is not None and bool(str(ng.pretrained_name).strip())
        has_path = ng.model_path is not None
        if has_pre == has_path:
            raise ValueError(
                "nfa_gate.enabled のとき、pretrained_name と model_path のどちらか一方だけを指定してください "
                "(例: ローカル .nemo のみ使う場合は pretrained_name: null と model_path をセット)"
            )
        return self

    @model_validator(mode="after")
    def mfa_compare_needs_phonemize(self) -> PipelineConfig:
        if self.mfa_gate.enabled and self.mfa_gate.compare_phones_to_g2p and not self.text.phonemize:
            raise ValueError(
                "mfa_gate.compare_phones_to_g2p requires text.phonemize=true "
                "(G2P 音素との比較に必要)"
            )
        return self

    @model_validator(mode="after")
    def nfa_compare_needs_phonemize(self) -> PipelineConfig:
        if self.nfa_gate.enabled and self.nfa_gate.compare_tokens_to_g2p and not self.text.phonemize:
            raise ValueError(
                "nfa_gate.compare_tokens_to_g2p requires text.phonemize=true "
                "(G2P 音素との比較に必要)"
            )
        return self

    @model_validator(mode="after")
    def nfa_compare_needs_token_map_path(self) -> PipelineConfig:
        ng = self.nfa_gate
        if ng.enabled and ng.compare_tokens_to_g2p and ng.nfa_to_g2p_token_map_path is None:
            raise ValueError(
                "nfa_gate.compare_tokens_to_g2p=true のときは nfa_to_g2p_token_map_path に "
                "YAML（NFA CTM トークン → OpenJTalk G2P）を指定してください。"
                "草案は次で生成できます: cv-preprocess suggest-nfa-g2p-map -c <config.yaml> -o <map.yaml>"
            )
        return self

    @model_validator(mode="after")
    def nfa_pred_text_compare_requires_align_mode(self) -> PipelineConfig:
        ng = self.nfa_gate
        if ng.enabled and ng.compare_pred_text_to_norm and not ng.align_using_pred_text:
            raise ValueError(
                "nfa_gate.compare_pred_text_to_norm=true には nfa_gate.align_using_pred_text=true が必要です "
                "(NeMo が pred_text を出力するモード)"
            )
        return self

    @model_validator(mode="after")
    def nfa_pred_phonemes_need_phonemize(self) -> PipelineConfig:
        if self.nfa_gate.enabled and self.nfa_gate.compare_pred_text_to_norm and not self.text.phonemize:
            raise ValueError(
                "nfa_gate.compare_pred_text_to_norm は参照音素列に OpenJTalk G2P を使うため text.phonemize=true が必要です"
            )
        return self

    @model_validator(mode="after")
    def nfa_compare_phoneme_vs_transcript_exclusive(self) -> PipelineConfig:
        ng = self.nfa_gate
        if ng.enabled and ng.compare_tokens_to_g2p and ng.compare_pred_text_to_norm:
            raise ValueError(
                "nfa_gate.compare_tokens_to_g2p と compare_pred_text_to_norm は同時に true にできません "
                "(音素トークン照合と認識文照合はどちらか一方)"
            )
        return self

    @model_validator(mode="after")
    def mfa_prefilter_needs_mfa(self) -> PipelineConfig:
        if self.mfa_gate.prefilter.enabled and not self.mfa_gate.enabled:
            raise ValueError(
                "mfa_gate.prefilter.enabled は mfa_gate.enabled=true と併用してください "
                "(MFA 投入前足切りは MFA を使う場合のみ有効)"
            )
        return self

    @model_validator(mode="after")
    def nfa_prefilter_needs_nfa(self) -> PipelineConfig:
        if self.nfa_gate.prefilter.enabled and not self.nfa_gate.enabled:
            raise ValueError(
                "nfa_gate.prefilter.enabled は nfa_gate.enabled=true と併用してください "
                "(NFA 投入前足切りは NFA を使う場合のみ有効)"
            )
        return self

    @model_validator(mode="after")
    def asr_compare_needs_phonemize(self) -> PipelineConfig:
        if self.asr_gate.enabled and self.asr_gate.compare_phonemes and not self.text.phonemize:
            raise ValueError(
                "asr_gate.compare_phonemes=true には text.phonemize=true が必要です "
                "(参照・仮説の音素化に OpenJTalk G2P を使います)"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(raw)
