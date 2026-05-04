from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import typer

from cv_preprocess.config import load_config
from cv_preprocess.pipeline.dataset_partition import run_dataset_partition, validate_group_by
from cv_preprocess.pipeline.ljspeech_tsv import metadata_jsonl_to_validated_tsv
from cv_preprocess.pipeline.preprocess import run_preprocess
from cv_preprocess.pipeline.mfa_g2p_map_suggest import Strategy, run_mfa_g2p_map_suggest
from cv_preprocess.pipeline.nfa_g2p_map_suggest import run_nfa_g2p_map_suggest
from cv_preprocess.pipeline.phoneme_manifest import run_phoneme_manifest
from cv_preprocess.pipeline.secondary import run_secondary
from cv_preprocess.pipeline.scan import scan_corpus
from cv_preprocess.text.normalize import normalize_for_tts
from cv_preprocess.text.phonemize import g2p_phonemes

app = typer.Typer(no_args_is_help=True, help="Common Voice → TTS preprocessing")


@app.command("scan")
def cmd_scan(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    cfg = load_config(config)
    info = scan_corpus(cfg)
    typer.echo(json.dumps(info, ensure_ascii=False, indent=2))


@app.command("metadata-jsonl-to-validated-tsv")
def cmd_metadata_jsonl_to_validated_tsv(
    metadata: Path = typer.Option(
        ...,
        "--metadata",
        "-m",
        exists=True,
        path_type=Path,
        help="metadata.jsonl のパス",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        path_type=Path,
        help="出力 validated.tsv（省略時は metadata と同じディレクトリに validated.tsv）",
    ),
) -> None:
    """metadata.jsonl から LJSpeech 互換の validated.tsv（3列・ヘッダなし）を生成する。"""
    dst = output if output is not None else metadata.parent / "validated.tsv"
    n = metadata_jsonl_to_validated_tsv(metadata, dst)
    typer.echo(json.dumps({"rows": n, "output": str(dst.resolve())}, ensure_ascii=False))


@app.command("dataset-partition")
def cmd_dataset_partition(
    metadata: Path = typer.Option(
        ...,
        "-m",
        "--metadata",
        exists=True,
        path_type=Path,
        help="preprocess が出力した metadata.jsonl",
    ),
    output: Path = typer.Option(
        ...,
        "-o",
        "--output",
        path_type=Path,
        help="バケット別サブディレクトリ（A / train / train__A 等）を作成する親ディレクトリ",
    ),
    audio_root: Path | None = typer.Option(
        None,
        "--audio-root",
        path_type=Path,
        help="WAV 実体のルート（metadata の audio_path からの相対先）。省略時は metadata と同じディレクトリ",
    ),
    group_by: str = typer.Option(
        "quality_tier",
        "--group-by",
        help="振り分けキー: quality_tier | split | split_quality_tier",
    ),
    min_quality_score: float | None = typer.Option(
        None,
        "--min-quality-score",
        help="この未満の quality_score を除外（annotate 済みメタ前提）",
    ),
    max_quality_score: float | None = typer.Option(
        None,
        "--max-quality-score",
        help="この超の quality_score を除外",
    ),
    only_tiers: str | None = typer.Option(
        None,
        "--only-tiers",
        help="カンマ区切り（例: A,B）。指定時は該当 quality_tier の行だけ出力",
    ),
    copy: bool = typer.Option(
        False,
        "--copy",
        help="WAV をコピーする（省略時はシンボリックリンク。リンク不可環境では自動でコピー）",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="件数集計のみ（ディスクへは書かない）",
    ),
) -> None:
    """品質ティア・split などに応じて WAV をバケット別フォルダへ集約する。

    各バケット ``{output}/{bucket}/`` に ``wavs/``・``metadata.jsonl``・``validated.tsv`` を出力する。
    学習用に A のみ、train かつ A のみ、スコア帯で切る、など `--group-by` とフィルタの組み合わせで指定する。
    """
    root = audio_root if audio_root is not None else metadata.parent
    try:
        gb = validate_group_by(group_by)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    tiers_list: list[str] | None = None
    if only_tiers is not None and only_tiers.strip():
        tiers_list = [x.strip() for x in only_tiers.split(",") if x.strip()]
    report = run_dataset_partition(
        metadata_path=metadata,
        audio_root=root,
        output_root=output,
        group_by=gb,
        min_quality_score=min_quality_score,
        max_quality_score=max_quality_score,
        only_tiers=tiers_list,
        use_symlink=not copy,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("preprocess")
def cmd_preprocess(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="進捗バー（stderr）を出さない。リダイレクトや CI 向け",
    ),
) -> None:
    cfg = load_config(config)
    report = run_preprocess(cfg, show_progress=not no_progress)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("phoneme-manifest")
def cmd_phoneme_manifest(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        path_type=Path,
        help="出力 JSONL（省略時は config.phoneme_manifest.output_path）",
    ),
    source: str | None = typer.Option(
        None,
        "--source",
        help="g2p_text（既定）| mfa_textgrid",
    ),
    mfa_textgrid_root: Path | None = typer.Option(
        None,
        "--mfa-textgrid-root",
        path_type=Path,
        help="source=mfa_textgrid 時: {stem(path)}.TextGrid を置くディレクトリ",
    ),
    mfa_token_map: Path | None = typer.Option(
        None,
        "--mfa-token-map",
        path_type=Path,
        help="MFA phones トークン → OpenJTalk G2P 列（YAML 辞書）",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="進捗バーを出さない",
    ),
) -> None:
    """OpenJTalk G2P 互換の音素照合 JSONL を生成（preprocess の text 条件と同一の絞り込み）。"""
    cfg = load_config(config)
    report = run_phoneme_manifest(
        cfg,
        output_path=output,
        source=source,
        mfa_textgrid_root=mfa_textgrid_root,
        mfa_token_map_path=mfa_token_map,
        show_progress=not no_progress,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("suggest-mfa-g2p-map")
def cmd_suggest_mfa_g2p_map(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    mfa_textgrid_root: Path = typer.Option(
        ...,
        "--mfa-textgrid-root",
        path_type=Path,
        exists=True,
        file_okay=False,
        help="{stem(TSV path)}.TextGrid を置くディレクトリ（phoneme-manifest の mfa_textgrid と同じ）",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        path_type=Path,
        help="書き出す YAML（*_report.json も隣に付く）",
    ),
    strategy: str = typer.Option(
        "adaptive",
        "--strategy",
        help="adaptive | zip_only | proportional_only",
    ),
    min_votes: int = typer.Option(2, "--min-votes", help="MFA キーごとの最有力 G2P への最低投票数"),
    min_ratio: float = typer.Option(
        0.55,
        "--min-ratio",
        help="最有力票 / 当該 MFA に関する総票の下限（0〜1）",
    ),
    existing_map: Path | None = typer.Option(
        None,
        "--existing-map",
        path_type=Path,
        exists=True,
        help="既存 YAML を読み、デフォルトでは未登録キーだけ埋める",
    ),
    overwrite_suggestions: bool = typer.Option(
        False,
        "--overwrite-suggestions",
        help="既存キーも含め提案で上書き（慎重に）",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="進捗バーを出さない",
    ),
) -> None:
    """MFA TextGrid と G2P を走査し、投票で ``mfa_to_g2p_token_map_path`` 用 YAML の草案を生成する。

    比例配置は近似のため誤対応が混ざる。必ず *_report.json の ambiguous / skipped を確認して人手で直すこと。
    """
    if strategy not in ("adaptive", "zip_only", "proportional_only"):
        raise typer.BadParameter("strategy must be adaptive | zip_only | proportional_only")
    cfg = load_config(config)
    report = run_mfa_g2p_map_suggest(
        cfg,
        mfa_textgrid_root=mfa_textgrid_root,
        output_yaml=output,
        strategy=cast(Strategy, strategy),
        min_votes=min_votes,
        min_ratio=min_ratio,
        existing_map_path=existing_map,
        fill_missing_keys_only=not overwrite_suggestions,
        show_progress=not no_progress,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("suggest-nfa-g2p-map")
def cmd_suggest_nfa_g2p_map(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        path_type=Path,
        help="書き出す YAML（*_report.json も隣に付く）",
    ),
    strategy: str = typer.Option(
        "adaptive",
        "--strategy",
        help="adaptive | zip_only | proportional_only",
    ),
    min_votes: int = typer.Option(2, "--min-votes", help="NFA キーごとの最有力 G2P への最低投票数"),
    min_ratio: float = typer.Option(
        0.55,
        "--min-ratio",
        help="最有力票 / 当該 NFA トークンに関する総票の下限（0〜1）",
    ),
    existing_map: Path | None = typer.Option(
        None,
        "--existing-map",
        path_type=Path,
        exists=True,
        help="既存 YAML を読み、デフォルトでは未登録キーだけ埋める",
    ),
    overwrite_suggestions: bool = typer.Option(
        False,
        "--overwrite-suggestions",
        help="既存キーも含め提案で上書き（慎重に）",
    ),
    max_clips: int | None = typer.Option(
        None,
        "--max-clips",
        help="先頭 N 件だけ処理（試行・デバッグ用）",
    ),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="進捗バーを出さない",
    ),
) -> None:
    """NFA（NeMo CTM トークン）と G2P を走査し、投票で ``nfa_to_g2p_token_map_path`` 用 YAML の草案を生成する。

    preprocess と同じ pass1 音声パイプラインのあとで NFA を呼ぶ。比例配置は近似のため誤対応が混ざる。
    必ず *_report.json の ambiguous_nfa / skipped_low_confidence を確認して人手で直すこと。
    """
    if strategy not in ("adaptive", "zip_only", "proportional_only"):
        raise typer.BadParameter("strategy must be adaptive | zip_only | proportional_only")
    cfg = load_config(config)
    report = run_nfa_g2p_map_suggest(
        cfg,
        output_yaml=output,
        strategy=cast(Strategy, strategy),
        min_votes=min_votes,
        min_ratio=min_ratio,
        existing_map_path=existing_map,
        fill_missing_keys_only=not overwrite_suggestions,
        show_progress=not no_progress,
        max_clips=max_clips,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("secondary")
def cmd_secondary(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    no_progress: bool = typer.Option(
        False,
        "--no-progress",
        help="進捗バー（stderr）を出さない",
    ),
) -> None:
    """一次 preprocess の metadata.jsonl / WAV に対し二次補正チェーンと再品質ゲートを適用する。"""
    cfg = load_config(config)
    report = run_secondary(cfg, show_progress=not no_progress)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("text-normalize")
def cmd_text_normalize(
    text: str = typer.Argument(..., help="Raw sentence"),
) -> None:
    typer.echo(normalize_for_tts(text))


@app.command("phonemize")
def cmd_phonemize(
    text: str = typer.Argument(..., help="Text (prefer normalized Japanese)"),
    kana: bool = typer.Option(False, "--kana", help="Output kana instead of phonemes"),
) -> None:
    typer.echo(g2p_phonemes(normalize_for_tts(text), kana=kana))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
