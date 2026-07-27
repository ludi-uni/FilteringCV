import type { JobType } from "../api/types";

export type JobPipelineKind = "stage" | "orchestrator" | "coverage-stage" | "coverage-orchestrator";

export interface JobPipelineItem {
  type: JobType;
  /** Display order within section; null for orchestrators */
  step: number | null;
  label: string;
  summary: string;
  kind: JobPipelineKind;
  /** Typical outputs / what to check after */
  produces: string;
  /** Optional pipeline section for UI grouping */
  section?: "builder" | "coverage";
}

/** Dataset builder stages (coverage is inserted between scan and analyze when enabled). */
export const JOB_PIPELINE: JobPipelineItem[] = [
  {
    type: "scan",
    step: 1,
    label: "Scan",
    summary: "コーパスと TSV を走査し、件数・パスを確認する（本処理の前の健全性チェック）。",
    kind: "stage",
    section: "builder",
    produces: "概要 JSON（ジョブ結果）",
  },
  {
    type: "analyze",
    step: 2,
    label: "Analyze",
    summary:
      "残りのクリップを品質解析する。coverage 実行後は既解析クリップを再利用し、未解析分だけ処理する。",
    kind: "stage",
    section: "builder",
    produces: "work/catalog/*.parquet、audio_cache",
  },
  {
    type: "plan-split",
    step: 3,
    label: "Plan split",
    summary:
      "train / val / test の分割計画を作る。unseen_speaker ではここで話者をバケット割当（select より先が意図どおり）。seen_speaker / single_speaker では薄い準備で、クリップの split は select 後に付く。",
    kind: "stage",
    section: "builder",
    produces: "work/plans/split_plan.json",
  },
  {
    type: "select",
    step: 4,
    label: "Select",
    summary:
      "coverage.features の必須目標を先に予約し、残り時間を貪欲選択で埋める（coverage-aware は既定オン）。監査は work/reports/selection/。unseen_speaker はバケット内選択、それ以外は全体選択のあと split 付与。",
    kind: "stage",
    section: "builder",
    produces: "work/plans/selection_plan.parquet、work/reports/selection/coverage-audit.*",
  },
  {
    type: "materialize",
    step: 5,
    label: "Materialize",
    summary:
      "選択クリップを最終出力へ書き出す。既定で exports/piper_plus と exports/style_bert_vits2 も生成（trainer_exports）。",
    kind: "stage",
    section: "builder",
    produces: "output wavs / metadata.* / exports/piper_plus / exports/style_bert_vits2",
  },
  {
    type: "audit",
    step: 6,
    label: "Audit",
    summary: "選択・分割・出力の整合性を検査する。",
    kind: "stage",
    section: "builder",
    produces: "監査レポート（ジョブ結果）",
  },
  {
    type: "build",
    step: null,
    label: "Build（推奨）",
    summary:
      "scan →（coverage 有効時は index/run）→ analyze → … → audit。coverage は analyze の前に走り、重い全件解析を避ける。",
    kind: "orchestrator",
    section: "builder",
    produces: "全段階 + work/run_manifest.json",
  },
];

/**
 * Rare-phoneme coverage automation — runs BEFORE full analyze.
 * Enable `coverage.enabled` (+ insert_before_analyze, default true).
 */
export const COVERAGE_PIPELINE: JobPipelineItem[] = [
  {
    type: "coverage-index",
    step: 1,
    label: "Coverage index",
    summary:
      "全候補の軽量インデックス（G2P・長さ・簡易品質のみ）。重い品質解析の前に必ずここから。",
    kind: "coverage-stage",
    section: "coverage",
    produces: "output/coverage/clip-index.jsonl",
  },
  {
    type: "coverage-plan",
    step: 2,
    label: "Coverage plan",
    summary: "不足特徴とスコアから次に解析すべき候補バッチを計画する（重い解析はしない）。",
    kind: "coverage-stage",
    section: "coverage",
    produces: "output/coverage/plan.json",
  },
  {
    type: "coverage-run",
    step: 3,
    label: "Coverage run",
    summary:
      "有望候補だけ品質解析を反復。analyze より先に実行し、カタログを部分構築する。",
    kind: "coverage-stage",
    section: "coverage",
    produces: "output/coverage/active-run/ + work/catalog（部分）",
  },
  {
    type: "coverage-report",
    step: 4,
    label: "Coverage report",
    summary: "active-run から JSON / CSV / Markdown / HTML レポートを再生成する。",
    kind: "coverage-stage",
    section: "coverage",
    produces: "report.md / report.html など",
  },
  {
    type: "coverage-build",
    step: null,
    label: "Coverage build",
    summary: "index → run → report。analyze / Build の直前に単独実行するとき用。",
    kind: "coverage-orchestrator",
    section: "coverage",
    produces: "index + active-run + reports",
  },
];

export const ALL_PIPELINES: JobPipelineItem[] = [...JOB_PIPELINE, ...COVERAGE_PIPELINE];

/** Unified display order when coverage is enabled: scan → coverage* → analyze → … */
export function orderedPipelineForDisplay(coverageEnabled: boolean): JobPipelineItem[] {
  const scan = JOB_PIPELINE.find((i) => i.type === "scan")!;
  const afterScan = JOB_PIPELINE.filter((i) => i.kind === "stage" && i.type !== "scan");
  const build = JOB_PIPELINE.find((i) => i.type === "build")!;
  if (!coverageEnabled) {
    return [...JOB_PIPELINE.filter((i) => i.kind === "stage"), build];
  }
  const coverageStages = COVERAGE_PIPELINE.filter((i) => i.kind === "coverage-stage");
  const coverageOrch = COVERAGE_PIPELINE.find((i) => i.kind === "coverage-orchestrator")!;
  return [scan, ...coverageStages, ...afterScan, coverageOrch, build];
}

export function jobPipelineItem(type: JobType): JobPipelineItem | undefined {
  return ALL_PIPELINES.find((item) => item.type === type);
}

export function formatJobTypeLabel(type: JobType): string {
  const item = jobPipelineItem(type);
  if (!item) return type;
  if (item.section === "coverage") {
    if (item.step != null) return `C${item.step}. ${item.label}`;
    return item.label;
  }
  if (item.step != null) return `${item.step}. ${item.label}`;
  return item.label;
}
