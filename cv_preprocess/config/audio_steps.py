from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from cv_preprocess.config.denoise_step import DenoiseStep
from cv_preprocess.config.gates_quality import QualityGateSidonAfterEnhanceSplitConfig

class DecodeStep(BaseModel):
    type: Literal["decode"] = "decode"


class ResampleStep(BaseModel):
    type: Literal["resample"] = "resample"
    sr: int = 22050


class EdgeUiClickStep(BaseModel):
    type: Literal["edge_ui_click"] = "edge_ui_click"
    lead_scan_ms: float = 120.0
    trail_scan_ms: float = 120.0
    max_transient_ms: float = 40.0
    peak_above_noise_db: float = 24.0
    removal: Literal["mute", "mute_then_trim", "fade"] = "mute_then_trim"
    #: 末尾クリック除去を **バースト直前の連続無音** がこの ms 以上のときに限る（録音停止 UI など）。
    #: ``null`` で無効（従来どおり末尾バーストのみで判定）。
    trail_click_requires_silence_ms: float | None = None
    #: 先頭クリック除去を **RMS ピークより左の連続低エネルギー** がこの ms 以上のときに限る。
    #: 語頭の「さ」等（先頭から有声）が UI クリックに誤判定されて欠けるのを防ぐ。``null`` で無効。
    lead_click_requires_pre_silence_ms: float | None = None
    #: 先頭クリックをミュートした **直後の区間だけ** に対し、``librosa.effects.trim`` 相当の
    #: ``top_db`` で無音を削る（全体の max 基準の eps トリムは使わない）。弱い子音の欠けを防ぐ。
    lead_post_click_top_db: float = 34.0
    #: ``null`` のとき ``peak_above_noise_db`` から内挿。``|diff|`` の ``median + k·MAD`` の k。
    diff_mad_k: float | None = None
    diff_median_smooth_samples: int = 5
    short_energy_ms: float = 3.0
    neighbor_energy_ms: float = 10.0
    #: ``null`` のとき ``peak_above_noise_db`` から内挿。短窓エネルギーが近傍比で何倍以上か。
    energy_spike_ratio: float | None = None
    speech_vad_frame_ms: float = 20.0
    speech_vad_hop_ms: float = 5.0
    #: ``null`` のとき ``peak_above_noise_db`` から内挿。持続有声の RMS 閾値の MAD 係数。
    speech_rms_mad_k: float | None = None
    min_speech_run_ms: float = 50.0
    mute_edge_fade_ms: float = 7.0
    #: True のとき、``run_steps_on_array`` では **少なくとも 1 回スペクトル系ステップ**
    #: （denoise / lip_noise_repair / lip_noise_suppress / diff_click_repair / bandwidth_extension / sidon_restore）
    #: のあとに限り、当該ステップをキューに入れ、**次の非スペクトル系ステップを実行する直前**
    #: にまとめて適用する。いままでスペクトルが一度も無いパス（align のみ等）ではキューせず
    #: その場で即時に適用する。
    post_spectral: bool = False

    @field_validator("diff_median_smooth_samples")
    @classmethod
    def edge_diff_smooth(cls, v: int) -> int:
        if v < 1 or v > 31:
            raise ValueError("edge_ui_click.diff_median_smooth_samples must be in [1, 31]")
        return v

    @field_validator("short_energy_ms", "neighbor_energy_ms")
    @classmethod
    def edge_energy_ms(cls, v: float) -> float:
        if v < 0.5 or v > 80.0:
            raise ValueError("edge_ui_click short_energy_ms / neighbor_energy_ms must be in [0.5, 80]")
        return v

    @field_validator("speech_vad_frame_ms")
    @classmethod
    def edge_vad_frame(cls, v: float) -> float:
        if v < 8.0 or v > 80.0:
            raise ValueError("edge_ui_click.speech_vad_frame_ms must be in [8, 80]")
        return v

    @field_validator("speech_vad_hop_ms")
    @classmethod
    def edge_vad_hop(cls, v: float) -> float:
        if v < 2.0 or v > 40.0:
            raise ValueError("edge_ui_click.speech_vad_hop_ms must be in [2, 40]")
        return v

    @field_validator("min_speech_run_ms")
    @classmethod
    def edge_min_speech_run(cls, v: float) -> float:
        if v < 15.0 or v > 400.0:
            raise ValueError("edge_ui_click.min_speech_run_ms must be in [15, 400]")
        return v

    @field_validator("mute_edge_fade_ms")
    @classmethod
    def edge_mute_fade_ms(cls, v: float) -> float:
        if v < 0.0 or v > 120.0:
            raise ValueError("edge_ui_click.mute_edge_fade_ms must be in [0, 120]")
        return v

    @field_validator("diff_mad_k", "energy_spike_ratio", "speech_rms_mad_k")
    @classmethod
    def edge_optional_positive(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 0.5 or v > 80.0:
            raise ValueError("edge_ui_click optional *_k / *_ratio must be null or in [0.5, 80]")
        return v

    @model_validator(mode="after")
    def edge_vad_hop_frame_order(self) -> EdgeUiClickStep:
        if self.speech_vad_hop_ms > self.speech_vad_frame_ms:
            raise ValueError("edge_ui_click.speech_vad_hop_ms must be <= speech_vad_frame_ms")
        return self

    @field_validator("lead_post_click_top_db")
    @classmethod
    def lead_post_click_db(cls, v: float) -> float:
        if v < 18.0 or v > 55.0:
            raise ValueError("lead_post_click_top_db must be in [18, 55]")
        return v

    @field_validator("trail_click_requires_silence_ms")
    @classmethod
    def trail_presilence_ms(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 5.0 or v > 3000.0:
            raise ValueError("trail_click_requires_silence_ms must be in [5, 3000] or null")
        return v

    @field_validator("lead_click_requires_pre_silence_ms")
    @classmethod
    def lead_presilence_ms(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 3.0 or v > 500.0:
            raise ValueError("lead_click_requires_pre_silence_ms must be in [3, 500] or null")
        return v



class NormalizeAudioStep(BaseModel):
    type: Literal["normalize"] = "normalize"
    method: Literal["loudness", "peak"] = "loudness"
    integrated_lufs: float = -23.0
    peak_dbfs: float = -1.0


class TrimSilenceStep(BaseModel):
    type: Literal["trim_silence"] = "trim_silence"
    max_keep_sec: float = 60.0
    head_tail_db: float = 40.0
    #: ``both`` = 従来どおり librosa trim。``trailing`` = 末尾のみ（BWE 後の尾部ノイズ・無音の残り対策）。
    #: ``leading`` = 先頭のみ。
    trim_sides: Literal["both", "leading", "trailing"] = "both"
    #: ``trim_sides: trailing`` のみ。末尾の非無音フレーム列で、無音に挟まれたこの長さ以下の島をスパイクとして無視（0 で無効）。
    max_trailing_spike_frames: int = 0
    #: ``librosa.effects.trim`` の STFT 窓。短い無音＋低ノイズの先頭を落としやすくする。
    trim_frame_length: int = 2048
    trim_hop_length: int = 512
    #: trim 後に先頭へ付与する無音（ミリ秒）。TTS 学習で語頭語尾の余白を確保したいときに使う。
    pad_head_ms: float = 0.0
    #: trim 後に末尾へ付与する無音（ミリ秒）。
    pad_tail_ms: float = 0.0

    @field_validator("max_trailing_spike_frames")
    @classmethod
    def trailing_spike_frames_bounds(cls, v: int) -> int:
        if v < 0 or v > 64:
            raise ValueError("max_trailing_spike_frames must be in [0, 64]")
        return v

    @field_validator("trim_frame_length")
    @classmethod
    def trim_frame_bounds(cls, v: int) -> int:
        if v < 256 or v > 8192:
            raise ValueError("trim_frame_length must be in [256, 8192]")
        return v

    @field_validator("trim_hop_length")
    @classmethod
    def trim_hop_bounds(cls, v: int) -> int:
        if v < 32 or v > 4096:
            raise ValueError("trim_hop_length must be in [32, 4096]")
        return v

    @model_validator(mode="after")
    def hop_not_larger_than_frame(self) -> TrimSilenceStep:
        if self.trim_hop_length > self.trim_frame_length:
            raise ValueError("trim_hop_length must be <= trim_frame_length")
        return self

    @field_validator("pad_head_ms", "pad_tail_ms")
    @classmethod
    def pad_ms_bounds(cls, v: float) -> float:
        if v < 0.0 or v > 500.0:
            raise ValueError("pad_*_ms must be in [0, 500]")
        return float(v)


class LowpassStep(BaseModel):
    type: Literal["lowpass"] = "lowpass"
    cutoff_hz: float = 11000.0
    order: int = 6


class HighpassStep(BaseModel):
    type: Literal["highpass"] = "highpass"
    cutoff_hz: float = 40.0
    order: int = 4


class LipNoiseSuppressStep(BaseModel):
    type: Literal["lip_noise_suppress"] = "lip_noise_suppress"
    n_fft: int = 2048
    hop_length: int = 512
    #: この Hz 以上のビンを抑制対象に含める（口内音の中高域寄り）。
    band_low_hz: float = 1400.0
    #: ``null`` でナイキストまで。
    band_high_hz: float | None = None
    #: フレーム全帯域エネルギー / 局所中央値。この倍率以上で「スパイク」候補。
    spike_ratio: float = 7.0
    #: 候補がこのフレーム数を超えて連続する場合は発話成分とみなし抑制しない。
    max_burst_frames: int = 6
    #: 対象帯のマグニチュード倍率（小さいほど強く抑える）。
    mag_gain: float = 0.52
    median_kernel_frames: int = 11
    temporal_smooth_frames: int = 3

    @field_validator("n_fft")
    @classmethod
    def lip_n_fft(cls, v: int) -> int:
        if v < 256 or v > 8192:
            raise ValueError("lip_noise_suppress.n_fft must be in [256, 8192]")
        return v

    @field_validator("hop_length")
    @classmethod
    def lip_hop(cls, v: int) -> int:
        if v < 32 or v > 4096:
            raise ValueError("lip_noise_suppress.hop_length must be in [32, 4096]")
        return v

    @field_validator("spike_ratio")
    @classmethod
    def lip_spike(cls, v: float) -> float:
        if v < 2.0 or v > 40.0:
            raise ValueError("spike_ratio must be in [2, 40]")
        return v

    @field_validator("max_burst_frames")
    @classmethod
    def lip_burst(cls, v: int) -> int:
        if v < 1 or v > 48:
            raise ValueError("max_burst_frames must be in [1, 48]")
        return v

    @field_validator("mag_gain")
    @classmethod
    def lip_mag_gain(cls, v: float) -> float:
        if v < 0.06 or v > 0.99:
            raise ValueError("mag_gain must be in [0.06, 0.99]")
        return v

    @field_validator("band_low_hz")
    @classmethod
    def lip_band_lo(cls, v: float) -> float:
        if v < 200.0 or v > 12000.0:
            raise ValueError("band_low_hz must be in [200, 12000]")
        return v

    @model_validator(mode="after")
    def lip_band_order(self) -> LipNoiseSuppressStep:
        if self.band_high_hz is not None and self.band_high_hz <= self.band_low_hz:
            raise ValueError("band_high_hz must be null or > band_low_hz")
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must be <= n_fft")
        return self


class LipNoiseRepairStep(BaseModel):
    """短い口内クリックを時間領域で検出し、前後からの補間で局所修復する（デクリック寄り）。"""

    type: Literal["lip_noise_repair"] = "lip_noise_repair"
    frame_ms: float = 2.5
    hop_ms: float = 0.65
    median_kernel_ms: float = 9.0
    rms_ratio_threshold: float = 5.5
    zcr_ratio_threshold: float = 2.6
    crest_factor_threshold: float = 6.0
    #: ``null``（省略可）で内蔵フラックス閾値と ZCR の OR 判定。``0`` 以下でフラックス無効（ZCR のみ）。正の値で明示閾値。
    flux_ratio_threshold: float | None = None
    max_event_ms: float = 24.0
    merge_gap_ms: float = 2.5
    repair_pad_ms: float = 1.8
    max_repair_ms: float = 28.0
    interpolation: Literal["linear", "cubic"] = "linear"
    max_repairs_per_clip: int = 96
    fft_bins: int = 256

    @field_validator("frame_ms")
    @classmethod
    def lnr_frame_ms(cls, v: float) -> float:
        if v < 0.8 or v > 12.0:
            raise ValueError("lip_noise_repair.frame_ms must be in [0.8, 12]")
        return v

    @field_validator("hop_ms")
    @classmethod
    def lnr_hop_ms(cls, v: float) -> float:
        if v < 0.2 or v > 4.0:
            raise ValueError("lip_noise_repair.hop_ms must be in [0.2, 4]")
        return v

    @field_validator("median_kernel_ms")
    @classmethod
    def lnr_med_k(cls, v: float) -> float:
        if v < 2.0 or v > 80.0:
            raise ValueError("lip_noise_repair.median_kernel_ms must be in [2, 80]")
        return v

    @field_validator(
        "rms_ratio_threshold",
        "zcr_ratio_threshold",
        "crest_factor_threshold",
    )
    @classmethod
    def lnr_positive_th(cls, v: float) -> float:
        if v < 1.05 or v > 50.0:
            raise ValueError("lip_noise_repair threshold must be in [1.05, 50]")
        return v

    @field_validator("flux_ratio_threshold")
    @classmethod
    def lnr_flux(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if v <= 0.0:
            return v
        if v < 1.05 or v > 50.0:
            raise ValueError(
                "lip_noise_repair.flux_ratio_threshold must be null, <=0 (disable flux), or in [1.05, 50]"
            )
        return v

    @field_validator("max_event_ms", "max_repair_ms")
    @classmethod
    def lnr_event_ms(cls, v: float) -> float:
        if v < 3.0 or v > 120.0:
            raise ValueError("lip_noise_repair max_*_ms must be in [3, 120]")
        return v

    @field_validator("merge_gap_ms", "repair_pad_ms")
    @classmethod
    def lnr_small_ms(cls, v: float) -> float:
        if v < 0.0 or v > 40.0:
            raise ValueError("lip_noise_repair merge/repair pad ms must be in [0, 40]")
        return v

    @field_validator("max_repairs_per_clip")
    @classmethod
    def lnr_max_rep(cls, v: int) -> int:
        if v < 0 or v > 500:
            raise ValueError("lip_noise_repair.max_repairs_per_clip must be in [0, 500]")
        return v

    @field_validator("fft_bins")
    @classmethod
    def lnr_fft(cls, v: int) -> int:
        if v < 64 or v > 2048:
            raise ValueError("lip_noise_repair.fft_bins must be in [64, 2048]")
        return v

    @model_validator(mode="after")
    def lnr_hop_frame_order(self) -> LipNoiseRepairStep:
        if self.hop_ms > self.frame_ms:
            raise ValueError("lip_noise_repair.hop_ms must be <= frame_ms")
        if self.max_repair_ms + 1e-6 < self.max_event_ms:
            raise ValueError("lip_noise_repair.max_repair_ms should be >= max_event_ms")
        return self


class DiffClickRepairStep(BaseModel):
    """1次／2次差分のスパイクでクリック候補を取り、短区間だけ線形または3次スプラインで埋める。"""

    type: Literal["diff_click_repair"] = "diff_click_repair"
    mad_k: float = 9.0
    #: ``null`` で無効。設定時は ``max(|y|)`` に対する 1 次差分の絶対下限を掛ける。
    min_abs_jump: float | None = None
    use_second_diff: bool = True
    second_diff_mad_k: float = 8.0
    #: True のとき 1 次と 2 次の両方でマークされたサンプルのみ（誤検出抑制）。``use_second_diff`` が必須。
    require_both: bool = False
    merge_gap_ms: float = 0.35
    repair_pad_ms: float = 1.5
    max_repair_ms: float = 20.0
    interpolation: Literal["linear", "cubic"] = "cubic"
    max_repairs_per_clip: int = 256

    @field_validator("mad_k", "second_diff_mad_k")
    @classmethod
    def dcr_mad_k(cls, v: float) -> float:
        if v < 2.5 or v > 35.0:
            raise ValueError("diff_click_repair mad_k / second_diff_mad_k must be in [2.5, 35]")
        return v

    @field_validator("min_abs_jump")
    @classmethod
    def dcr_min_abs(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if v < 1e-6 or v > 0.5:
            raise ValueError("diff_click_repair.min_abs_jump must be null or in (0, 0.5]")
        return v

    @field_validator("merge_gap_ms", "repair_pad_ms")
    @classmethod
    def dcr_gap_pad(cls, v: float) -> float:
        if v < 0.0 or v > 40.0:
            raise ValueError("diff_click_repair merge/repair_pad ms must be in [0, 40]")
        return v

    @field_validator("max_repair_ms")
    @classmethod
    def dcr_max_rep(cls, v: float) -> float:
        if v < 2.0 or v > 50.0:
            raise ValueError("diff_click_repair.max_repair_ms must be in [2, 50]")
        return v

    @field_validator("max_repairs_per_clip")
    @classmethod
    def dcr_max_repairs(cls, v: int) -> int:
        if v < 0 or v > 800:
            raise ValueError("diff_click_repair.max_repairs_per_clip must be in [0, 800]")
        return v

    @model_validator(mode="after")
    def dcr_require_both_second(self) -> DiffClickRepairStep:
        if self.require_both and not self.use_second_diff:
            raise ValueError("diff_click_repair.require_both requires use_second_diff: true")
        return self


class SaveWavStep(BaseModel):
    type: Literal["save_wav"] = "save_wav"
    bit_depth: int = 16


class BandwidthExtensionStep(BaseModel):
    """HiFi-GAN 系ボコーダでメル→波形を生成し帯域を補う。公式 jik876/hifi-gan の config.json + generator 重みを想定。"""

    type: Literal["bandwidth_extension"] = "bandwidth_extension"
    #: ``generator_checkpoint`` と同じディレクトリに ``config.json`` がある場合は省略可。
    config_json: Path | None = None
    generator_checkpoint: Path
    device: Literal["auto", "cpu", "cuda"] = "auto"


class SidonRestoreStep(QualityGateSidonAfterEnhanceSplitConfig):
    """Sidon 音声復元をパイプラインに直接挿入する（``uv sync --extra sidon``）。

    パラメータは ``quality_gate.sidon_after_enhance_split`` と同一（HF 初回取得）。
    ``enabled: false`` のとき当該ステップは no-op。
    """

    type: Literal["sidon_restore"] = "sidon_restore"
    enabled: bool = True


AudioStep = Annotated[
    Union[
        DecodeStep,
        ResampleStep,
        EdgeUiClickStep,
        DenoiseStep,
        NormalizeAudioStep,
        TrimSilenceStep,
        LowpassStep,
        HighpassStep,
        LipNoiseRepairStep,
        DiffClickRepairStep,
        LipNoiseSuppressStep,
        BandwidthExtensionStep,
        SidonRestoreStep,
        SaveWavStep,
    ],
    Field(discriminator="type"),
]


class AudioPipelineConfig(BaseModel):
    target_sample_rate: int = 22050
    channels: int = 1
    audio_pipeline_id: str = "default_v1"
    steps: list[AudioStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def inject_decode_if_missing(self) -> AudioPipelineConfig:
        if not self.steps:
            return self
        if self.steps[0].type != "decode":
            self.steps.insert(0, DecodeStep())
        return self

