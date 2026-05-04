from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from cv_preprocess.config.gates_quality import PhonemeAlignmentCheckConfig

class TextConfig(BaseModel):
    max_text_len: int = 500
    min_text_len: int = 1
    require_japanese: bool = True
    phonemize: bool = True
    g2p_kana: bool = False
    phoneme_alignment_check: PhonemeAlignmentCheckConfig = Field(default_factory=PhonemeAlignmentCheckConfig)

    @model_validator(mode="after")
    def phoneme_alignment_needs_g2p(self) -> TextConfig:
        if self.phoneme_alignment_check.enabled and not self.phonemize:
            raise ValueError(
                "text.phoneme_alignment_check requires text.phonemize=true "
                "(コーパス側の比較元は G2P 音素列)"
            )
        return self
