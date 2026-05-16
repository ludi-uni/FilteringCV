from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

class MfaPrefilterConfig(BaseModel):
    """``mfa align`` を呼ぶ前に、MFA 入力波形へ品質ゲートをかけて母集団を減らす。"""

    enabled: bool = False
    #: ``quality_gate`` の辞書に上書きマージして MFA 直前の ``y`` に適用（空なら ``quality_gate`` と同一）
    quality_gate_overrides: dict[str, Any] = Field(default_factory=dict)


class MfaGateConfig(BaseModel):
    """ノイズ除去後の波形に ``mfa align`` をかけ、失敗または（任意で）G2P 音素不一致で拒否する。"""

    enabled: bool = False
    mfa_executable: str = "mfa"
    #: ``mfa align`` の辞書（事前に ``mfa model download dictionary japanese_mfa`` 等）
    dictionary: str = "japanese_mfa"
    acoustic_model: str = "japanese_mfa"
    batch_size: int = 32
    num_jobs: int = 4
    #: ``true`` のとき ``num_jobs`` を ``max(1, os.cpu_count())`` で上書き（レポートに実効値を記録）
    auto_num_jobs: bool = False
    #: ``true`` のとき ``batch_size = max(設定値, min(batch_size_max, 実効 num_jobs * multiplier))``
    auto_scale_batch_size: bool = False
    auto_batch_jobs_multiplier: int = 8
    batch_size_max: int = 128
    single_speaker: bool = True  # 将来の CLI 拡張用（現状は常に付与）
    #: ``mfa align`` にそのまま渡す追加引数
    extra_align_args: list[str] = Field(default_factory=list)
    timeout_sec: float = 3600.0
    #: バッチ用一時ディレクトリの親。未指定時は呼び出し側で ``output.root`` を使う
    work_dir: Path | None = None
    #: 各バッチ終了後に中間 corpus / align 出力を削除
    clean_workdir: bool = True
    #: ``mfa align`` の ``--clean``
    clean: bool = True
    beam: int | None = None
    retry_beam: int | None = None
    g2p_model_path: str | None = None
    #: .lab 本文から空白を除去（日本語 MFA は連続表記を期待しがち）
    lab_strip_spaces: bool = True
    #: TextGrid の音素列と G2P を比較する。記号体系が異なる場合は
    #: ``mfa_to_g2p_token_map_path`` で MFA トークン→OpenJTalk G2P 列（YAML）を指定し、**変換後**に比較する。
    compare_phones_to_g2p: bool = False
    max_token_error_rate_vs_g2p: float = 0.0
    #: ``compare_phones_to_g2p: true`` 時に推奨。``phoneme-manifest`` の ``--mfa-token-map`` と同形式（MFA phones ラベル → 空白区切り G2P トークン列。空値は当該 MFA トークンを省略）。
    mfa_to_g2p_token_map_path: Path | None = None
    prefilter: MfaPrefilterConfig = Field(default_factory=MfaPrefilterConfig)

    @field_validator("batch_size")
    @classmethod
    def batch_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("mfa_gate.batch_size must be >= 1")
        return v

    @field_validator("auto_batch_jobs_multiplier")
    @classmethod
    def mfa_batch_mult(cls, v: int) -> int:
        if v < 1 or v > 64:
            raise ValueError("mfa_gate.auto_batch_jobs_multiplier must be in [1, 64]")
        return v

    @field_validator("batch_size_max")
    @classmethod
    def mfa_batch_max(cls, v: int) -> int:
        if v < 1 or v > 4096:
            raise ValueError("mfa_gate.batch_size_max must be in [1, 4096]")
        return v

    @field_validator("max_token_error_rate_vs_g2p")
    @classmethod
    def mfa_phone_rate(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("mfa_gate.max_token_error_rate_vs_g2p must be in [0, 1]")
        return v


class NfaPrefilterConfig(BaseModel):
    """NFA（``align.py``）を呼ぶ前に、入力波形へ品質ゲートをかけて母集団を減らす。"""

    enabled: bool = False
    #: ``quality_gate`` の辞書に上書きマージして NFA 直前の ``y`` に適用（空なら ``quality_gate`` と同一）
    quality_gate_overrides: dict[str, Any] = Field(default_factory=dict)


class NfaGateConfig(BaseModel):
    """ノイズ除去後の波形に NeMo Forced Aligner をかけ、失敗または（任意で）G2P トークン不一致で拒否する。"""

    enabled: bool = False
    #: NeMo チェックポイント名（例: ``nvidia/parakeet-tdt_ctc-0.6b-ja``）。``model_path`` と同時指定不可。
    pretrained_name: str | None = "nvidia/parakeet-tdt_ctc-0.6b-ja"
    #: ローカル ``.nemo``。指定時は ``pretrained_name`` を null にすること。
    model_path: Path | None = None
    #: NFA 用 Python（未指定時は環境変数 ``NFA_PYTHON``、無ければ ``python3``）
    nfa_python: str | None = None
    #: ``align.py`` があるディレクトリ（未指定時は ``NFA_ALIGN_DIR``）
    nfa_align_dir: Path | None = None
    #: アライメント入力 WAV のサンプルレート（Parakeet 系は 16000）
    model_sample_rate_hz: int = 16000
    batch_size: int = 1
    #: ``persistent_worker`` の init で NeMo に渡す ``chunk_batch_size``（``use_buffered_chunked_streaming`` 等でストリーミングする構成でピークメモリに効く。通常の一括アラインでは影響は小さいが **共有メモリ警告**が出る場合は ``nfa_gate.batch_size`` とあわせて下げる）。
    worker_chunk_batch_size: int = 32
    #: Conformer 系で NFA が試みるローカルアテンション
    use_local_attention: bool = True
    #: ``align.py`` にそのまま渡す追加 Hydra 引数（``key=value`` 形式の文字列）
    extra_align_args: list[str] = Field(default_factory=list)
    #: ``true`` のとき、NeMo を **1 subprocess 常駐**（`nfa_align_worker.py`）で動かしモデルを初回のみロードする。
    #: ``false`` または環境変数 ``CV_PREPROCESS_NFA_SUBPROCESS=1`` のときは従来どおりバッチごとに ``align.py`` を起動する。
    persistent_worker: bool = True
    #: ``two_pass_denoise`` 有効時、**第2パス（SGMSE 等）に入る直前**に常駐 NFA subprocess を終了し **子プロセス側の GPU を解放**する。
    #: 同一 GPU 上で NFA と SGMSE の **VRAM が重ならない**ようにする（次 NFA バッチでモデル再ロードのため **遅くなる**）。
    release_persistent_worker_before_two_pass_enhance: bool = False
    timeout_sec: float = 7200.0
    work_dir: Path | None = None
    clean_workdir: bool = True
    #: マニフェスト ``text`` から空白を除去（日本語の連続表記向け）
    manifest_strip_spaces: bool = True
    #: NeMo ``align_using_pred_text``。音声を認識した ``pred_text`` で強制アライメントする（参照 ``text`` ではない）。
    align_using_pred_text: bool = False
    #: ``true`` のとき、NeMo の ``pred_text`` を正規化のうえ **OpenJTalk G2P** し、参照の ``phonemes``（``text_norm`` 由来）と **音素トークン列の編集距離率**で照合する（音素マップ不要）。
    #: ``align_using_pred_text=true`` かつ ``text.phonemize=true`` が必須。``compare_tokens_to_g2p`` と同時に true にできない。
    compare_pred_text_to_norm: bool = False
    max_pred_phoneme_error_rate_vs_norm: float = 0.18
    #: CTM のモデルトークン列と G2P を比較する。記号体系が異なる場合は
    #: ``nfa_to_g2p_token_map_path`` で NFA トークン→OpenJTalk G2P 列（YAML）を指定し、**変換後**に比較する。
    compare_tokens_to_g2p: bool = False
    max_token_error_rate_vs_g2p: float = 0.0
    #: ``compare_tokens_to_g2p: true`` 時に推奨。形式は ``mfa_to_g2p_token_map_path`` と同じ（キーが NFA トークン）。
    nfa_to_g2p_token_map_path: Path | None = None
    prefilter: NfaPrefilterConfig = Field(default_factory=NfaPrefilterConfig)
    #: enhance 後の **最終** ``run_quality_gate``（accept 直前）に ``quality_gate`` へマージする上書き。
    #: ``nfa_gate.enabled: true`` のときのみ有効。空ならルート ``quality_gate`` のみ。
    quality_gate_overrides: dict[str, Any] = Field(default_factory=dict)
    #: 最終品質ゲート用プロファイル名（``quality_gate_profiles`` のキー）。``quality_gate`` より先にベースとしてマージし、続けて ``quality_gate_overrides`` を適用。
    quality_gate_profile: str | None = None

    @field_validator("pretrained_name", mode="before")
    @classmethod
    def nfa_pretrained_empty_as_none(cls, v: Any) -> Any:
        if v == "":
            return None
        return v

    @field_validator("batch_size")
    @classmethod
    def nfa_batch_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("nfa_gate.batch_size must be >= 1")
        return v

    @field_validator("worker_chunk_batch_size")
    @classmethod
    def nfa_worker_chunk_batch(cls, v: int) -> int:
        if int(v) < 1 or int(v) > 128:
            raise ValueError("nfa_gate.worker_chunk_batch_size must be in [1, 128]")
        return int(v)

    @field_validator("model_sample_rate_hz")
    @classmethod
    def nfa_sr_positive(cls, v: int) -> int:
        if v < 4000 or v > 192000:
            raise ValueError("nfa_gate.model_sample_rate_hz must be in [4000, 192000]")
        return v

    @field_validator("max_token_error_rate_vs_g2p")
    @classmethod
    def nfa_token_rate(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("nfa_gate.max_token_error_rate_vs_g2p must be in [0, 1]")
        return v

    @field_validator("max_pred_phoneme_error_rate_vs_norm")
    @classmethod
    def nfa_pred_phoneme_rate(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("nfa_gate.max_pred_phoneme_error_rate_vs_norm must be in [0, 1]")
        return v

