from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class QualityGateSidonAfterEnhanceSplitConfig(BaseModel):
    """二段 denoise の **split enhance**（``audio_pipeline_enhance``）適用直後に、品質ゲートと **同一の reject 基準**
    （``run_quality_gate`` が ``ok=False``）のときだけ Sidon 復元を試み、その後ゲートを再実行する。

    重みは初回実行時に Hugging Face Hub から取得（``huggingface_hub``）。``uv sync --extra sidon`` が必要。
    """

    enabled: bool = False
    hf_repo_id: str = "sarulab-speech/sidon-v0.1"
    feature_extractor_filename_cuda: str = "feature_extractor_cuda.pt"
    decoder_filename_cuda: str = "decoder_cuda.pt"
    feature_extractor_filename_cpu: str = "feature_extractor_cpu.pt"
    decoder_filename_cpu: str = "decoder_cpu.pt"
    #: ``auto`` = CUDA 利用可なら ``cuda``、否则 ``cpu``（JIT ファイルも対応ペアを選択）
    device: Literal["auto", "cpu", "cuda"] = "auto"
    ssl_model_id: str = "facebook/w2v-bert-2.0"


class QualityGateConfig(BaseModel):
    min_duration_sec: float = 0.5
    max_duration_sec: float = 30.0
    max_silence_ratio: float = 0.5
    #: 無音判定の基準レベル。各フレーム RMS が ``ref * silence_ratio_rms_floor`` 未満なら無音。
    silence_ratio_rms_floor: float = 0.05
    #: 基準 ``ref`` に使う RMS のパーセンタイル（100 = 従来どおり max）。短い発話で一瞬のピークだけ
    #: 大きいとき max 基準だと無音率が異常に上がるため、90〜98 程度が実用的。
    silence_ratio_ref_percentile: float = 100.0
    min_estimated_snr_db: float | None = None
    max_clipping_ratio: float = 0.01
    max_abs_dc_offset: float = 0.05
    min_chars_per_sec: float | None = None
    max_chars_per_sec: float | None = None
    #: 日本語向け: 1 モーラあたりこれ未満の実長（秒）は「異常に短い」とみなす下限。
    #: ``trim_silence`` の ``max_keep_sec`` や末尾トリムで文章が欠けた候補を落とす。
    min_sec_per_mora: float | None = None
    #: ``required = mora_count * min_sec_per_mora * mora_gate_relax``。1.0 未満でやや緩める。
    mora_gate_relax: float = 1.0
    #: ``off`` = ティアなし（従来）。``annotate`` = A/B/C と ``quality_score`` をメタデータにだけ付与。
    #: ``reject_c`` / ``reject_b`` = ハードゲート通過後にティアで追加拒否（低品質の取りこぼし抑制）。
    quality_tier_mode: Literal["off", "annotate", "reject_c", "reject_b"] = "off"
    #: 推定 SNR が取れるときの A ティア下限（dB）。B ティアは ``quality_tier_b_min_snr_db`` 以上。
    quality_tier_a_min_snr_db: float = 16.0
    quality_tier_b_min_snr_db: float = 9.0
    #: 無音率の上限（小さいほど良い）。A は B より厳しく、どちらも ``max_silence_ratio`` 以下であること。
    quality_tier_a_max_silence_ratio: float = 0.40
    quality_tier_b_max_silence_ratio: float = 0.50
    #: A ティア向けのクリッピング率上限（``max_clipping_ratio`` 以下）。
    quality_tier_a_max_clipping_ratio: float = 0.008
    #: 推定 SNR が得られないとき、無音率だけで A/B を切る閾値（``max_silence_ratio`` 以下）。
    quality_tier_unknown_snr_silence_a: float = 0.38
    quality_tier_unknown_snr_silence_b: float = 0.50
    #: 末尾から連続する無音（秒）がこれを超えるクリップは **A にしない**（B 条件を満たせば B、否则 C）。
    #: ``null`` で無効（従来どおり SNR・無音率・クリップのみでティア）。
    quality_tier_a_max_trailing_silence_sec: float | None = None
    #: 末尾連続無音（秒）がこれを超えたら **ハード拒否**（``gate_trailing_silence``）。ティアより前に評価。
    #: ``measure_trailing_silence_sec`` と同一の窓・閾値（``snr`` の frame/hop、``silence_ratio_rms_floor`` / ``silence_ratio_ref_percentile``）。
    #: ``null`` で無効。
    max_trailing_silence_sec: float | None = None
    #: split enhance 直後の Sidon 救済（``two_pass`` + ``audio_pipeline_enhance`` のときのみ有効）
    sidon_after_enhance_split: QualityGateSidonAfterEnhanceSplitConfig = Field(
        default_factory=QualityGateSidonAfterEnhanceSplitConfig
    )

    @field_validator("max_trailing_silence_sec")
    @classmethod
    def max_trailing_hard_gate(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0.0 or v > 20.0:
            raise ValueError("max_trailing_silence_sec must be in [0, 20] or null")
        return v

    @field_validator("quality_tier_a_max_trailing_silence_sec")
    @classmethod
    def tier_max_trailing(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0.0 or v > 20.0:
            raise ValueError("quality_tier_a_max_trailing_silence_sec must be in [0, 20] or null")
        return v

    @field_validator("mora_gate_relax")
    @classmethod
    def mora_relax_range(cls, v: float) -> float:
        if v < 0.5 or v > 1.2:
            raise ValueError("mora_gate_relax must be in [0.5, 1.2]")
        return v

    @field_validator("min_sec_per_mora")
    @classmethod
    def min_sec_per_mora_range(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0.03 or v > 0.22:
            raise ValueError("min_sec_per_mora must be in [0.03, 0.22] or null")
        return v

    @field_validator("silence_ratio_ref_percentile")
    @classmethod
    def ref_percentile_range(cls, v: float) -> float:
        if v < 50.0 or v > 100.0:
            raise ValueError("silence_ratio_ref_percentile must be in [50, 100]")
        return v

    @field_validator("quality_tier_a_min_snr_db", "quality_tier_b_min_snr_db")
    @classmethod
    def tier_snr_range(cls, v: float) -> float:
        if v < -5.0 or v > 80.0:
            raise ValueError("quality_tier_*_min_snr_db must be in [-5, 80]")
        return v

    @field_validator(
        "quality_tier_a_max_silence_ratio",
        "quality_tier_b_max_silence_ratio",
        "quality_tier_unknown_snr_silence_a",
        "quality_tier_unknown_snr_silence_b",
    )
    @classmethod
    def tier_silence_range(cls, v: float) -> float:
        if v < 0.0 or v > 0.95:
            raise ValueError("quality_tier silence bounds must be in [0, 0.95]")
        return v

    @field_validator("quality_tier_a_max_clipping_ratio")
    @classmethod
    def tier_clip_range(cls, v: float) -> float:
        if v < 0.0 or v > 0.05:
            raise ValueError("quality_tier_a_max_clipping_ratio must be in [0, 0.05]")
        return v

    @model_validator(mode="after")
    def quality_tier_consistency(self) -> QualityGateConfig:
        if self.quality_tier_b_min_snr_db > self.quality_tier_a_min_snr_db - 1e-6:
            raise ValueError(
                "quality_tier_b_min_snr_db must be <= quality_tier_a_min_snr_db "
                "(B は A より緩い SNR 下限)"
            )
        if self.quality_tier_a_max_silence_ratio > self.quality_tier_b_max_silence_ratio + 1e-6:
            raise ValueError(
                "quality_tier_a_max_silence_ratio must be <= quality_tier_b_max_silence_ratio"
            )
        if self.quality_tier_unknown_snr_silence_a > self.quality_tier_unknown_snr_silence_b + 1e-6:
            raise ValueError(
                "quality_tier_unknown_snr_silence_a must be <= quality_tier_unknown_snr_silence_b"
            )
        if self.quality_tier_b_max_silence_ratio > self.max_silence_ratio + 1e-6:
            raise ValueError(
                "quality_tier_b_max_silence_ratio must be <= max_silence_ratio "
                "(ハードゲートをすり抜けた後の値なので)"
            )
        if self.quality_tier_unknown_snr_silence_b > self.max_silence_ratio + 1e-6:
            raise ValueError("quality_tier_unknown_snr_silence_b must be <= max_silence_ratio")
        if self.quality_tier_a_max_clipping_ratio > self.max_clipping_ratio + 1e-6:
            raise ValueError("quality_tier_a_max_clipping_ratio must be <= max_clipping_ratio")
        return self


class SnrEstimatorConfig(BaseModel):
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    noise_percentile: float = 15.0
    signal_percentile: float = 60.0
    min_frames: int = 4


class EarlyAudioGateConfig(BaseModel):
    """デコード直後、``audio_pipeline.target_sample_rate`` に合わせた波形で、重いチェーンの前に軽い足切り。"""

    enabled: bool = False
    check_duration: bool = True
    check_silence_ratio: bool = False
    check_snr: bool = False
    check_clipping: bool = True
    check_dc_offset: bool = True
    check_chars_per_sec: bool = False
    check_mora_duration: bool = False


class TwoPassDenoiseConfig(BaseModel):
    """第 1 パスで ``denoise`` をスキップし、足切り（MFA / NFA 等）通過後にのみ非 none の denoise を適用する。"""

    enabled: bool = False
    #: SGMSE のみ二段 denoise のとき、``enhance_batch`` に載せるクリップ数の上限（大きいほど速いが **VRAM・中間テンソル**が増える）。
    #: Docker/WSL で共有メモリ（``/dev/shm``）や GPU メモリが厳しいときは ``2``〜``4`` などに下げる。
    sgmse_micro_batch_max: int = 8

    @field_validator("sgmse_micro_batch_max")
    @classmethod
    def two_pass_micro_bounds(cls, v: int) -> int:
        if int(v) < 1 or int(v) > 32:
            raise ValueError("two_pass_denoise.sgmse_micro_batch_max must be in [1, 32]")
        return int(v)

    #: ``interleaved``（既定）— MFA/NFA バッチ通過のたびに第2パス（SGMSE 等）へ進む。
    #: ``after_align_complete`` — **全行の足切り・align フラッシュが終わった後**にだけ第2パスをまとめて実行する。
    #: NFA 常駐 subprocess を閉じてから SGMSE を走らせやすく **GPU VRAM の重なりを避けやすい**一方、通過クリップの波形を **メモリに溜める**ためコーパスが大きいと **ホスト RAM** が増える。
    enhance_phase: Literal["interleaved", "after_align_complete"] = "interleaved"

    @model_validator(mode="after")
    def enhance_phase_requires_two_pass(self) -> TwoPassDenoiseConfig:
        if self.enhance_phase == "after_align_complete" and not self.enabled:
            raise ValueError(
                "two_pass_denoise.enhance_phase=after_align_complete requires two_pass_denoise.enabled=true"
            )
        return self


class PhonemeAlignmentCheckConfig(BaseModel):
    """外部マニフェストのアライメント音素と G2P 音素を比較し、ずれが大きいクリップを拒否する。"""

    enabled: bool = False
    manifest_path: Path | None = None
    #: マニフェストに ``source_path`` が無いとき。skip=比較せず通過、reject=拒否
    missing_manifest_entry: Literal["skip", "reject"] = "skip"
    #: トークン編集距離 ÷ max(トークン数) の許容上限。0=トークン列完全一致のみ
    max_token_error_rate: float = 0.0

    @field_validator("max_token_error_rate")
    @classmethod
    def rate_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("phoneme_alignment_check.max_token_error_rate must be in [0, 1]")
        return v

    @model_validator(mode="after")
    def manifest_when_enabled(self) -> PhonemeAlignmentCheckConfig:
        if self.enabled and self.manifest_path is None:
            raise ValueError("phoneme_alignment_check.manifest_path is required when enabled")
        return self
