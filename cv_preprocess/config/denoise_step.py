from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class DenoiseStep(BaseModel):
    type: Literal["denoise"] = "denoise"
    method: str = "none"
    #: ``dasheng`` 用。Hugging Face のモデル ID（既定は公式重み）。
    dasheng_model_id: str = "mispeech/dasheng-denoiser"
    dasheng_device: Literal["auto", "cpu", "cuda"] = "auto"
    #: CUDA 時に ``torch.autocast(fp16)`` で推論（速い／省 VRAM。音質は若干変わることがある）。
    dasheng_cuda_autocast_fp16: bool = True
    #: ``sgmse`` 用。Hugging Face / SpeechBrain のソース（既定は VoiceBank 学習の SGMSE）。
    sgmse_model_source: str = "speechbrain/sgmse-voicebank"
    #: ``sgmse`` 用。重みキャッシュ先。未指定時は ``~/.cache/cv_preprocess/sgmse-voicebank``。
    sgmse_savedir: str | None = None
    sgmse_device: Literal["auto", "cpu", "cuda"] = "auto"
    #: CUDA 時に ``torch.autocast(fp16)``（拡散系は品質劣化しやすいため既定はオフ）。
    sgmse_cuda_autocast_fp16: bool = False
    #: ``None`` のとき従来どおり入力 ``sr`` に戻す（48 kHz→16 kHz→48 kHz の往復）。
    #: 例えば ``16000`` にすると SGMSE 出力を 16 kHz のまま返し、後段 ``resample`` で 48 kHz に上げる
    #: （中間 16 kHz・出力のみ高 SR）構成で往復リサンプルを 1 回減らせる。
    sgmse_return_sample_rate: int | None = None
    #: ``wpe_deepfilternet`` 用。NARA-WPE を挟まず DFN のみにする場合は false。
    wpe_deepfilternet_run_wpe: bool = True
    #: WPE の STFT（入力 ``sr`` 基準）。
    wpe_n_fft: int = 512
    wpe_hop_length: int = 128
    wpe_taps: int = 10
    wpe_delay: int = 3
    wpe_iterations: int = 3
    wpe_psd_context: int = 0
    wpe_statistics_mode: Literal["full", "valid"] = "full"
    #: 公式 zip 名（``DeepFilterNet`` / ``DeepFilterNet2`` / ``DeepFilterNet3``）またはチェックポイント展開済みディレクトリ。
    #: 既定は `Rikorose/DeepFilterNet <https://github.com/Rikorose/DeepFilterNet>`_ の学習済み **DeepFilterNet2**（MIT / Apache-2.0 デュアル）。
    deepfilternet_model: str = "DeepFilterNet2"
    deepfilternet_post_filter: bool = False
    deepfilternet_device: Literal["auto", "cpu", "cuda"] = "auto"
    #: DeepFilterNet の loguru レベル。バッチ前処理では ``none`` 推奨。
    deepfilternet_log_level: str = "none"
    #: DeepFilterNet ``enhance`` の attenuation limit (dB)。``null`` で無効。
    deepfilternet_atten_lim_db: float | None = None
    #: ``spectral_subtract`` 用。先頭・末尾の何 ms をノイズ推定に使うか（処理済みサンプルレート基準）。
    noise_lead_ms: float = 140.0
    noise_trail_ms: float = 140.0
    #: 追加で、フレーム RMS が全体のこのパーセンタイル以下の区間をノイズ候補に混ぜる（無音・小休止の環境ノイズ）。
    noise_quiet_rms_percentile: float = 22.0
    #: ノイズ候補フレーム数の上限（平均が安定する程度）。
    max_noise_frames: int = 48
    #: 推定ノイズパワーに掛ける減算強度。大きいほど強く削るが音がこもりやすい。
    subtract_alpha: float = 1.15
    #: 出力マグニチュードの下限（元の ``spectral_floor`` 倍）。小さいとミュージカルノイズ、大きいと残響・ノイズが残りやすい。
    spectral_floor: float = 0.14
    #: 末尾何 ms を「反響・環境が残りやすい帯」として追加推定に使うか。0 で無効。
    reverb_tail_ms: float = 0.0
    #: 末尾帯のフレーム RMS が、その帯内のこのパーセンタイル以下のフレームだけ平均して推定（声より反響寄り）。
    reverb_tail_frame_percentile: float = 40.0
    #: 推定反響スペクトルをノイズ見積りへ混ぜる強さ（0〜1）。大きいほど反響抑えやすいが声が薄くなりやすい。
    reverb_tail_mix: float = 0.0
    n_fft: int = 2048
    hop_length: int = 512

    @field_validator("method")
    @classmethod
    def denoise_method(cls, v: object) -> str:
        s = str(v).strip().lower()
        allowed = {
            "none",
            "",
            "skip",
            "spectral_subtract",
            "dasheng",
            "dasheng_denoiser",
            "sgmse",
            "wpe_deepfilternet",
        }
        if s not in allowed:
            raise ValueError(f"denoise.method must be one of {sorted(allowed)}, got {v!r}")
        if s in ("", "skip"):
            return "none"
        if s == "dasheng_denoiser":
            return "dasheng"
        return s

    @field_validator("noise_quiet_rms_percentile")
    @classmethod
    def quiet_pct(cls, v: float) -> float:
        if v < 0.0 or v > 50.0:
            raise ValueError("noise_quiet_rms_percentile must be in [0, 50]")
        return v

    @field_validator("spectral_floor")
    @classmethod
    def spec_floor(cls, v: float) -> float:
        if v < 0.02 or v > 0.45:
            raise ValueError("spectral_floor must be in [0.02, 0.45]")
        return v

    @field_validator("subtract_alpha")
    @classmethod
    def sub_alpha(cls, v: float) -> float:
        if v < 0.3 or v > 3.0:
            raise ValueError("subtract_alpha must be in [0.3, 3.0]")
        return v

    @field_validator("reverb_tail_ms")
    @classmethod
    def reverb_tail_ms_range(cls, v: float) -> float:
        if v < 0.0 or v > 1200.0:
            raise ValueError("reverb_tail_ms must be in [0, 1200]")
        return v

    @field_validator("reverb_tail_frame_percentile")
    @classmethod
    def reverb_tail_pct(cls, v: float) -> float:
        if v < 5.0 or v > 95.0:
            raise ValueError("reverb_tail_frame_percentile must be in [5, 95]")
        return v

    @field_validator("reverb_tail_mix")
    @classmethod
    def reverb_tail_mix_range(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("reverb_tail_mix must be in [0, 1]")
        return v

    @field_validator("sgmse_return_sample_rate")
    @classmethod
    def sgmse_return_sr(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if int(v) < 4000 or int(v) > 96000:
            raise ValueError("sgmse_return_sample_rate must be in [4000, 96000] or null")
        return int(v)

    @field_validator("wpe_n_fft")
    @classmethod
    def wpe_n_fft_bounds(cls, v: int) -> int:
        if v < 256 or v > 4096:
            raise ValueError("wpe_n_fft must be in [256, 4096]")
        return v

    @field_validator("wpe_hop_length")
    @classmethod
    def wpe_hop_bounds(cls, v: int) -> int:
        if v < 32 or v > 2048:
            raise ValueError("wpe_hop_length must be in [32, 2048]")
        return v

    @field_validator("wpe_taps")
    @classmethod
    def wpe_taps_bounds(cls, v: int) -> int:
        if v < 2 or v > 30:
            raise ValueError("wpe_taps must be in [2, 30]")
        return v

    @field_validator("wpe_delay")
    @classmethod
    def wpe_delay_bounds(cls, v: int) -> int:
        if v < 0 or v > 20:
            raise ValueError("wpe_delay must be in [0, 20]")
        return v

    @field_validator("wpe_iterations")
    @classmethod
    def wpe_iter_bounds(cls, v: int) -> int:
        if v < 1 or v > 10:
            raise ValueError("wpe_iterations must be in [1, 10]")
        return v

    @field_validator("wpe_psd_context")
    @classmethod
    def wpe_psd_ctx_bounds(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError("wpe_psd_context must be in [0, 10]")
        return v

    @field_validator("deepfilternet_model")
    @classmethod
    def deepfilternet_model_non_empty(cls, v: object) -> str:
        s = str(v).strip()
        if not s:
            raise ValueError("deepfilternet_model must be non-empty")
        return s

    @field_validator("deepfilternet_log_level")
    @classmethod
    def dfn_log_level(cls, v: object) -> str:
        return str(v).strip()

