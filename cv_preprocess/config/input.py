from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

class InputConfig(BaseModel):
    corpus_root: Path
    clip_tsv: str = "validated.tsv"
    audio_subdir: str = "clips"
    locale_expected: str | None = "ja"
    #: 話者・メタフィルタ適用後、パス名ソートの先頭 N 件だけ処理する（試行・小規模抜き出し用）。``null`` で全件。
    max_clips: int | None = None

    @field_validator("max_clips")
    @classmethod
    def max_clips_positive(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if int(v) < 1:
            raise ValueError("input.max_clips must be >= 1 when set")
        return int(v)


class ClipMetadataFilters(BaseModel):
    """Common Voice クリップ TSV の属性による入力絞り込み。各リストが空ならその軸はフィルタしない。"""

    genders: list[str] = Field(default_factory=list)
    ages: list[str] = Field(default_factory=list)
    accents: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    locales: list[str] = Field(default_factory=list)
    segments: list[str] = Field(default_factory=list)
    #: 設定時、TSV の ``up_votes`` がこの値 **以上** の行だけ残す（``null`` で無効）。
    up_votes: int | None = None
    #: 設定時、TSV の ``down_votes`` がこの値 **以下** の行だけ残す（``null`` で無効）。
    down_votes: int | None = None

    @field_validator(
        "genders",
        "ages",
        "accents",
        "variants",
        "locales",
        "segments",
        mode="before",
    )
    @classmethod
    def coerce_str_lists(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, str):
            # スカラー ``""`` は「空セル許容」の指定に使うため ``[""]`` として残す
            return [str(v).strip()]
        return v

    @field_validator("up_votes", "down_votes", mode="before")
    @classmethod
    def coerce_optional_vote_threshold(cls, v: object) -> object:
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            raise ValueError("up_votes/down_votes must be an integer or null, not boolean")
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return int(s, 10)
        return v

    @field_validator("up_votes", "down_votes")
    @classmethod
    def vote_threshold_non_negative(cls, v: int | None) -> int | None:
        if v is None:
            return None
        if v < 0:
            raise ValueError("up_votes/down_votes must be >= 0 when set")
        return v

    @model_validator(mode="after")
    def strip_nonempty_tokens(self) -> ClipMetadataFilters:
        def clean_allowlist(xs: list[str]) -> list[str]:
            """要素ごとに strip。明示的な ``""`` は「TSV 空セルも許容」のマーカーとして残す。"""

            return [str(x).strip() for x in xs if x is not None]

        self.genders = clean_allowlist(self.genders)
        self.ages = clean_allowlist(self.ages)
        self.accents = clean_allowlist(self.accents)
        self.variants = clean_allowlist(self.variants)
        self.locales = clean_allowlist(self.locales)
        self.segments = clean_allowlist(self.segments)
        return self

    def is_active(self) -> bool:
        return bool(
            self.genders
            or self.ages
            or self.accents
            or self.variants
            or self.locales
            or self.segments
            or self.up_votes is not None
            or self.down_votes is not None
        )


_DEFAULT_MERGED_SPEAKER_CLIENT_ID = "__cv_merged_speaker__"


class SpeakersConfig(BaseModel):
    include_client_ids: list[str] = Field(default_factory=list)
    clip_metadata_filters: ClipMetadataFilters = Field(default_factory=ClipMetadataFilters)
    #: 各 ``client_id`` あたり **採用（manifest 追記）** するクリップの上限。最終品質ゲート等を通過したあとに数え、超えた分は ``rejects`` に ``max_clips_per_speaker`` で記録する。上限に達した話者の残り行は **テキスト検証より前に短絡拒否** して音声処理を省略する。同一バッチ内では ``source_path`` 昇順で採用枠を埋める。
    max_clips_per_speaker: int | None = None
    #: ``include_client_ids`` / ``clip_metadata_filters`` 適用後に残った全行の ``client_id`` を同一の合成 ID に置き換える（多話者を単一話者として出力・分割する）。
    merge_filtered_speakers_as_one: bool = False
    #: 上記が true のときの ``client_id``。未設定または空なら ``__cv_merged_speaker__``。
    merged_speaker_client_id: str | None = None

    @field_validator("include_client_ids", mode="before")
    @classmethod
    def coerce_include_client_ids(cls, v: object) -> object:
        """Allow YAML `include_client_ids: onehash` (scalar) as well as a list."""
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        return v

    @field_validator("merged_speaker_client_id", mode="before")
    @classmethod
    def coerce_merged_speaker_client_id(cls, v: object) -> object:
        if v is None or v == "":
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("max_clips_per_speaker", mode="before")
    @classmethod
    def coerce_max_clips_per_speaker(cls, v: object) -> object:
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            raise ValueError("speakers.max_clips_per_speaker must be an integer or null, not boolean")
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            return int(s, 10)
        return v

    @field_validator("max_clips_per_speaker")
    @classmethod
    def max_clips_per_speaker_positive(cls, v: int | None) -> int | None:
        if v is None:
            return v
        if int(v) < 1:
            raise ValueError("speakers.max_clips_per_speaker must be >= 1 when set")
        return int(v)

    def resolved_merged_speaker_client_id(self) -> str:
        if self.merged_speaker_client_id:
            return self.merged_speaker_client_id
        return _DEFAULT_MERGED_SPEAKER_CLIENT_ID
