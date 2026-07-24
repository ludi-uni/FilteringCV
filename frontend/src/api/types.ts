export type JobStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "cancelled"
  | "succeeded"
  | "failed"
  | "interrupted";

export type JobType =
  | "scan"
  | "analyze"
  | "plan-split"
  | "select"
  | "materialize"
  | "audit"
  | "build";

export interface JobSummary {
  id: string;
  job_type: JobType;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  error_message?: string | null;
}

export interface JobRecord extends JobSummary {
  config_path: string;
  force: boolean;
  pid?: number | null;
  result?: Record<string, unknown> | null;
}

export interface CreateJobRequest {
  job_type: JobType;
  force?: boolean;
}

export interface ProgressRecord {
  id?: number | null;
  job_id: string;
  stage: string;
  message: string;
  current?: number | null;
  total?: number | null;
  fraction?: number | null;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface DashboardSummary {
  config_path: string;
  work_dir: string;
  output_dir: string;
  catalog_ready: boolean;
  job_status_counts: Record<string, number>;
  recent_jobs: Array<{
    id: string;
    job_type: JobType;
    status: JobStatus;
    created_at: string;
    updated_at: string;
  }>;
  run_manifest: Record<string, unknown> | null;
}

export interface ClipItem {
  clip_id: string;
  speaker_id: string;
  disposition: string;
  text_norm: string;
  duration_sec: number;
  quality_score: number | null;
  audio_cache_rel_path: string;
  split: string;
  reject_reason: string | null;
  override_flags: string | null;
}

export interface ClipsPage {
  page: number;
  page_size: number;
  total: number;
  items: ClipItem[];
}

export interface FeatureCoverageEntry {
  feature_type: string;
  feature: string;
  pool_count: number;
  pool_speaker_count: number;
  pool_utterance_count: number;
}

export interface CoverageReport {
  total_clips: number;
  eligible_clips: number;
  feature_types: string[];
  unique_features: number;
  entries: FeatureCoverageEntry[];
  js_distance_to_uniform: Record<string, number>;
}

export type OverrideAction =
  | "force_include"
  | "force_exclude"
  | "hard_reject"
  | "return_to_reserve";

export interface ClipOverride {
  clip_id: string;
  action: OverrideAction;
  reason?: string | null;
  metadata?: Record<string, unknown>;
}

export interface OverrideListResponse {
  path: string;
  overrides: ClipOverride[];
}

export interface CompareRequest {
  left: string;
  right: string;
}

export interface CompareResult {
  left_dir: string;
  right_dir: string;
  config_diff: {
    left?: unknown;
    right?: unknown;
    same?: boolean;
  };
  duration_delta_sec: Record<string, number>;
  speaker_counts: {
    left_selected_clips: number;
    right_selected_clips: number;
    only_left: string[];
    only_right: string[];
    intersection: string[];
  };
  coverage_js_delta: Record<string, number>;
  clip_set_diff: {
    added: string[];
    removed: string[];
    unchanged_count: number;
  };
}

export interface ConfigSection {
  id: string;
  label: string;
  group: string;
}

export interface ConfigResponse {
  path: string;
  relative_path: string;
  yaml_text: string;
  data: Record<string, unknown>;
  sections: ConfigSection[];
  json_schema: Record<string, unknown>;
}

export interface ConfigValidateResponse {
  ok: boolean;
  data?: Record<string, unknown> | null;
  yaml_text?: string | null;
  errors: string[];
}

export interface ConfigSaveResponse {
  ok: boolean;
  path: string;
  data: Record<string, unknown>;
  yaml_text: string;
  message: string;
}

export interface SessionState {
  bound: boolean;
  config_path: string | null;
  project_root: string;
}

export interface SessionConfigItem {
  path: string;
}

export interface SessionConfigsResponse {
  configs: SessionConfigItem[];
}

/** Ordered job ids (see `frontend/src/jobs/pipeline.ts` for labels/descriptions). */
export const JOB_TYPES: JobType[] = [
  "scan",
  "analyze",
  "plan-split",
  "select",
  "materialize",
  "audit",
  "build",
];

export const DISPOSITIONS = [
  "hard_rejected",
  "eligible",
  "selected",
  "reserve",
] as const;

export const OVERRIDE_ACTIONS: OverrideAction[] = [
  "force_include",
  "force_exclude",
  "hard_reject",
  "return_to_reserve",
];
