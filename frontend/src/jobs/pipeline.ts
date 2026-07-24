import type { JobType } from "../api/types";

export type JobPipelineKind = "stage" | "orchestrator";

export interface JobPipelineItem {
  type: JobType;
  /** 1-based stage order; null for the full-pipeline orchestrator */
  step: number | null;
  label: string;
  summary: string;
  kind: JobPipelineKind;
  /** Typical outputs / what to check after */
  produces: string;
}

/** Dataset builder job order shown in the GUI (scan → … → audit; build runs them). */
export const JOB_PIPELINE: JobPipelineItem[] = [
  {
    type: "scan",
    step: 1,
    label: "Scan",
    summary: "コーパスと TSV を走査し、件数・パスを確認する（本処理の前の健全性チェック）。",
    kind: "stage",
    produces: "概要 JSON（ジョブ結果）",
  },
  {
    type: "analyze",
    step: 2,
    label: "Analyze",
    summary: "クリップを解析し、品質ゲート・音声キャッシュ・カタログ Parquet を作る（重い段階）。",
    kind: "stage",
    produces: "work/catalog/*.parquet、audio_cache",
  },
  {
    type: "plan-split",
    step: 3,
    label: "Plan split",
    summary:
      "train / val / test の分割計画を作る。unseen_speaker ではここで話者をバケット割当（select より先が意図どおり）。seen_speaker / single_speaker では薄い準備で、クリップの split は select 後に付く。",
    kind: "stage",
    produces: "work/plans/split_plan.json",
  },
  {
    type: "select",
    step: 4,
    label: "Select",
    summary:
      "言語カバレッジを満たすようクリップを貪欲選択する。unseen_speaker では split ごとのバケット内で選択。それ以外は全体選択のあとクリップに split を付与（感覚的には select→split）。overrides 再適用もここ。",
    kind: "stage",
    produces: "work/plans/selection_plan.parquet",
  },
  {
    type: "materialize",
    step: 5,
    label: "Materialize",
    summary: "選択クリップを最終出力（WAV・メタデータ・分割マニフェスト）へ書き出す。",
    kind: "stage",
    produces: "output wavs / validated.tsv / metadata.jsonl など",
  },
  {
    type: "audit",
    step: 6,
    label: "Audit",
    summary: "選択・分割・出力の整合性を検査する。",
    kind: "stage",
    produces: "監査レポート（ジョブ結果）",
  },
  {
    type: "build",
    step: null,
    label: "Build（推奨）",
    summary:
      "上記 1〜6 を順に実行するワンショット。途中成果物があれば再開する。初めての実行はここから。",
    kind: "orchestrator",
    produces: "全段階 + work/run_manifest.json",
  },
];

export function jobPipelineItem(type: JobType): JobPipelineItem | undefined {
  return JOB_PIPELINE.find((item) => item.type === type);
}

export function formatJobTypeLabel(type: JobType): string {
  const item = jobPipelineItem(type);
  if (!item) return type;
  if (item.step != null) return `${item.step}. ${item.label}`;
  return item.label;
}
