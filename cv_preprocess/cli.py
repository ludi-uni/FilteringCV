from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import cast

import typer

from cv_preprocess.application.analyze import analyze_project
from cv_preprocess.application.benchmark_selection import benchmark_selection
from cv_preprocess.application.audit import audit_dataset
from cv_preprocess.application.build import build_dataset
from cv_preprocess.application.materialize import materialize_dataset
from cv_preprocess.application.select import load_selection_plan, select_dataset
from cv_preprocess.application.split import load_split_plan, plan_dataset_split
from cv_preprocess.catalog.reader import load_catalog
from cv_preprocess.config import load_config
from cv_preprocess.reports.comparison import compare_runs
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


@app.command("benchmark-selection")
def cmd_benchmark_selection(
    catalog: Path = typer.Option(
        ...,
        "--catalog",
        exists=True,
        path_type=Path,
        help="Path to catalog/clips.parquet",
    ),
    repeat: int = typer.Option(3, "--repeat", min=1, help="Number of timed repetitions"),
    backend: str = typer.Option("auto", "--backend", help="auto | polars | python"),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        path_type=Path,
        help="Optional pipeline config for selection weights/constraints",
    ),
) -> None:
    """Benchmark selection scoring and greedy selection on an existing catalog."""
    cfg = load_config(config) if config is not None else None
    report = benchmark_selection(
        catalog,
        config=cfg,
        repeat=repeat,
        backend=backend,
    )
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("analyze")
def cmd_analyze(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    """Dataset builder analyze: emit catalog parquet and audio cache under work_dir."""
    cfg = load_config(config)
    result = analyze_project(cfg)
    typer.echo(
        json.dumps(
            {
                "catalog": result.catalog.model_dump(mode="json"),
                "eligible_count": result.eligible_count,
                "hard_rejected_count": result.hard_rejected_count,
                "warnings": result.warnings,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("plan-split")
def cmd_plan_split(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    """Dataset builder plan-split: assign speakers/clips to train/val/test splits."""
    cfg = load_config(config)
    work_dir = cfg.dataset_builder.work_dir
    catalog = load_catalog(work_dir)
    if catalog.clips_path is None:
        raise typer.BadParameter(f"catalog not found under {work_dir / 'catalog'}")
    result = plan_dataset_split(cfg, catalog)
    typer.echo(
        json.dumps(
            {
                "catalog": result.catalog.model_dump(mode="json"),
                "protocol": result.protocol,
                "ratios": result.ratios,
                "speaker_assignments": result.speaker_assignments,
                "clip_assignments": result.clip_assignments,
                "warnings": result.warnings,
                "plan_path": str(result.plan_path) if result.plan_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("materialize")
def cmd_materialize(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    """Dataset builder materialize: export selected clips to output directory."""
    cfg = load_config(config)
    work_dir = cfg.dataset_builder.work_dir
    catalog = load_catalog(work_dir)
    if catalog.clips_path is None:
        raise typer.BadParameter(f"catalog not found under {work_dir / 'catalog'}")
    plan_path = work_dir / "plans" / "selection_plan.parquet"
    if not plan_path.is_file():
        raise typer.BadParameter(f"selection plan not found: {plan_path}")
    selection_plan = load_selection_plan(catalog, plan_path)
    result = materialize_dataset(cfg, catalog, selection_plan)
    typer.echo(
        json.dumps(
            {
                "output_root": result.output_root,
                "selected_count": result.selected_count,
                "manifest_paths": result.manifest_paths,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("audit")
def cmd_audit(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    """Dataset builder audit: validate selection and split integrity."""
    cfg = load_config(config)
    work_dir = cfg.dataset_builder.work_dir
    catalog = load_catalog(work_dir)
    if catalog.clips_path is None:
        raise typer.BadParameter(f"catalog not found under {work_dir / 'catalog'}")
    plan_path = work_dir / "plans" / "selection_plan.parquet"
    if not plan_path.is_file():
        raise typer.BadParameter(f"selection plan not found: {plan_path}")
    selection_plan = load_selection_plan(catalog, plan_path)
    result = audit_dataset(cfg, catalog, selection_plan)
    typer.echo(
        json.dumps(
            {
                "passed": result.passed,
                "issues": result.issues,
                "catalog": result.catalog.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("build")
def cmd_build(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    force: bool = typer.Option(False, "--force", help="Re-run stages even if artifacts exist"),
) -> None:
    """Dataset builder build: orchestrate scan through audit with stage resume."""
    cfg = load_config(config)
    (
        scan_result,
        analyze_result,
        split_plan,
        selection_plan,
        materialize_result,
        audit_report,
    ) = build_dataset(cfg, force=force)
    typer.echo(
        json.dumps(
            {
                "scan": scan_result.model_dump(mode="json"),
                "analyze": {
                    "eligible_count": analyze_result.eligible_count,
                    "hard_rejected_count": analyze_result.hard_rejected_count,
                    "warnings": analyze_result.warnings,
                },
                "split_protocol": split_plan.protocol,
                "selected_count": len(selection_plan.selected_clip_ids),
                "materialize_output_root": materialize_result.output_root,
                "audit_passed": audit_report.passed,
                "audit_issues": audit_report.issues,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("compare-runs")
def cmd_compare_runs(
    left: Path = typer.Argument(..., exists=True, path_type=Path, help="Left work or output dir"),
    right: Path = typer.Argument(..., exists=True, path_type=Path, help="Right work or output dir"),
) -> None:
    """Compare two dataset builder runs or materialized outputs."""
    report = compare_runs(left, right)
    typer.echo(json.dumps(report, ensure_ascii=False, indent=2))


@app.command("select")
def cmd_select(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
) -> None:
    """Dataset builder select: greedy coverage selection from catalog."""
    cfg = load_config(config)
    work_dir = cfg.dataset_builder.work_dir
    catalog = load_catalog(work_dir)
    if catalog.clips_path is None:
        raise typer.BadParameter(f"catalog not found under {work_dir / 'catalog'}")
    split_plan_path = work_dir / "plans" / "split_plan.json"
    if split_plan_path.is_file():
        split_plan = load_split_plan(catalog, split_plan_path)
    else:
        split_plan = plan_dataset_split(cfg, catalog)
    result = select_dataset(cfg, catalog, split_plan)
    typer.echo(
        json.dumps(
            {
                "catalog": result.catalog.model_dump(mode="json"),
                "selected_count": len(result.selected_clip_ids),
                "reserve_count": len(result.reserve_clip_ids),
                "plan_path": str(result.plan_path) if result.plan_path else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
    if cfg.dataset_builder.enabled:
        warnings.warn(
            "dataset_builder.enabled is true; delegating preprocess to build_dataset. "
            "Prefer `cv-preprocess build` directly.",
            UserWarning,
            stacklevel=1,
        )
        (
            _scan,
            _analyze,
            _split,
            selection_plan,
            materialize_result,
            audit_report,
        ) = build_dataset(cfg)
        typer.echo(
            json.dumps(
                {
                    "delegated_to": "build_dataset",
                    "selected_count": len(selection_plan.selected_clip_ids),
                    "materialize_output_root": materialize_result.output_root,
                    "audit_passed": audit_report.passed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
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


@app.command("coverage-index")
def cmd_coverage_index(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    input_tsv: Path | None = typer.Option(None, "--input", exists=True, path_type=Path),
    clips_dir: Path | None = typer.Option(
        None,
        "--clips-dir",
        path_type=Path,
        help="Optional override; defaults to input.corpus_root / input.audio_subdir",
    ),
    output: Path = typer.Option(
        Path("output/coverage/clip-index.jsonl"),
        "--output",
        "-o",
        path_type=Path,
    ),
    force: bool = typer.Option(False, "--force"),
    incremental: bool = typer.Option(False, "--incremental"),
    workers: int = typer.Option(1, "--workers"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    """Build a lightweight clip index for coverage automation (no MFA/ASR)."""
    from cv_preprocess.coverage.indexer import build_clip_index

    cfg = load_config(config)
    if clips_dir is not None:
        # Keep corpus_root; audio_subdir override via temporary mutation is avoided —
        # document that clips-dir should match corpus layout. Warn if mismatched.
        expected = cfg.input.corpus_root / cfg.input.audio_subdir
        if clips_dir.resolve() != expected.resolve():
            typer.echo(
                f"Note: --clips-dir {clips_dir} differs from config audio path {expected}; "
                "index still resolves audio via corpus_root/audio_subdir.",
                err=True,
            )
    result = build_clip_index(
        cfg,
        output=output,
        input_tsv=input_tsv,
        force=force,
        incremental=incremental,
        workers=workers,
        limit=limit,
    )
    typer.echo(
        json.dumps(
            {
                "index_path": str(result.index_path),
                "meta_path": str(result.meta_path),
                "clip_count": result.clip_count,
                "config_hash": result.meta.config_hash,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("coverage-plan")
def cmd_coverage_plan(
    config: Path = typer.Option(..., "--config", "-c", exists=True, path_type=Path),
    index: Path = typer.Option(..., "--index", exists=True, path_type=Path),
    accepted_metadata: Path | None = typer.Option(None, "--accepted-metadata", path_type=Path),
    output: Path = typer.Option(Path("output/coverage/plan.json"), "--output", "-o", path_type=Path),
) -> None:
    """Plan the next coverage analysis batch from deficits and the lightweight index."""
    from cv_preprocess.coverage.counter import load_accepted_counts
    from cv_preprocess.coverage.indexer import load_index_jsonl
    from cv_preprocess.coverage.planner import plan_coverage
    from cv_preprocess.reports.serializer import write_json_atomic

    cfg = load_config(config)
    if not cfg.coverage.enabled:
        raise typer.BadParameter("coverage.enabled must be true")
    records = load_index_jsonl(index)
    catalog_clips = None
    meta = accepted_metadata
    if meta is not None and meta.suffix.lower() == ".parquet":
        catalog_clips = meta
        meta = None
    accepted = load_accepted_counts(accepted_metadata=meta, catalog_clips=catalog_clips)
    plan = plan_coverage(config=cfg.coverage, index_records=records, accepted_counts=accepted)
    write_json_atomic(output, plan.to_dict())
    typer.echo(json.dumps({"output": str(output), "batch_size": plan.batch_size}, ensure_ascii=False, indent=2))


@app.command("coverage-run")
def cmd_coverage_run(
    config: Path | None = typer.Option(None, "--config", "-c", path_type=Path),
    index: Path | None = typer.Option(None, "--index", path_type=Path),
    accepted_metadata: Path | None = typer.Option(None, "--accepted-metadata", path_type=Path),
    output: Path | None = typer.Option(None, "--output", "-o", path_type=Path),
    resume: Path | None = typer.Option(None, "--resume", path_type=Path),
    dry_run: bool = typer.Option(False, "--dry-run"),
    max_iterations: int | None = typer.Option(None, "--max-iterations"),
    max_clips: int | None = typer.Option(None, "--max-clips"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
) -> None:
    """Iteratively plan and analyze clips until coverage targets are met (or stop)."""
    from cv_preprocess.coverage.runner import run_coverage
    from cv_preprocess.coverage.state import load_run_state

    if resume is not None:
        run_dir = resume
        state_preview = load_run_state(run_dir)
        if config is None:
            raise typer.BadParameter("--config is required with --resume")
        if index is None:
            raise typer.BadParameter("--index is required with --resume")
        cfg = load_config(config)
        state = run_coverage(
            cfg,
            index_path=index,
            output_dir=run_dir,
            accepted_metadata=accepted_metadata,
            resume=True,
            dry_run=dry_run,
            max_iterations=max_iterations,
            max_clips=max_clips,
            batch_size=batch_size,
        )
        _ = state_preview
    else:
        if config is None or index is None or output is None:
            raise typer.BadParameter("--config, --index, and --output are required (or use --resume)")
        cfg = load_config(config)
        state = run_coverage(
            cfg,
            index_path=index,
            output_dir=output,
            accepted_metadata=accepted_metadata,
            resume=False,
            dry_run=dry_run,
            max_iterations=max_iterations,
            max_clips=max_clips,
            batch_size=batch_size,
        )
    typer.echo(
        json.dumps(
            {
                "run_id": state.run_id,
                "status": state.status.value,
                "iteration": state.iteration,
                "analyzed": len(state.analyzed_clip_ids),
                "accepted": len(state.accepted_clip_ids),
                "rejected": len(state.rejected_clip_ids),
                "remaining_deficits": state.remaining_deficits,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("coverage-report")
def cmd_coverage_report(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, path_type=Path),
    config: Path | None = typer.Option(None, "--config", "-c", path_type=Path),
) -> None:
    """Regenerate coverage reports from an existing run directory."""
    from cv_preprocess.coverage.report import generate_report_from_run_dir

    coverage_cfg = None
    if config is not None:
        coverage_cfg = load_config(config).coverage
    paths = generate_report_from_run_dir(run_dir, coverage_cfg)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}, ensure_ascii=False, indent=2))


def _default_gui_project_root(config: Path | None) -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "frontend").is_dir() or (cwd / "pyproject.toml").is_file():
        return cwd
    if config is not None:
        for candidate in (config.resolve().parent, *config.resolve().parents):
            if (candidate / "frontend").is_dir() or (candidate / "pyproject.toml").is_file():
                return candidate
    return cwd


@app.command("gui")
def cmd_gui(
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        exists=True,
        path_type=Path,
        help="Pipeline YAML (optional; uses last config or setup screen)",
    ),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (default localhost only)"),
    port: int = typer.Option(8765, "--port", help="Bind port"),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        path_type=Path,
        help="Project root for work/output/frontend (default: cwd or repo with frontend/)",
    ),
) -> None:
    """Start the dataset builder FastAPI GUI (serves frontend/dist when built)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter(
            "GUI dependencies missing; install with: uv sync --extra gui"
        ) from exc

    from cv_preprocess.web.app import create_app
    from cv_preprocess.web.last_config import to_project_relative, write_last_config
    from cv_preprocess.web.session_resolve import resolve_gui_startup_config

    root = project_root.resolve() if project_root is not None else _default_gui_project_root(config)
    # Soft-fail unloadable last_config (no -c); explicit -c still fails in create_app.
    resolved = resolve_gui_startup_config(project_root=root, cli_config=config)
    dist = root / "frontend" / "dist"
    if not dist.is_dir():
        typer.echo(
            f"Warning: frontend build not found at {dist}. "
            "Run: ./scripts/start-gui.sh   # or: cd frontend && pnpm install && pnpm build",
            err=True,
        )
    app = create_app(resolved, root)
    # Persist last_config only after create_app succeeds (validates explicit -c).
    if config is not None and resolved is not None:
        write_last_config(root, to_project_relative(root, resolved))
    uvicorn.run(app, host=host, port=port, log_level="info")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
