import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { JobRecord, JobSummary, JobStatus, JobType } from "../api/types";
import { Pagination } from "../components/Pagination";
import { StatusBadge } from "../components/StatusBadge";
import { useJobWebSocket } from "../hooks/useJobWebSocket";
import {
  formatJobTypeLabel,
  jobPipelineItem,
  orderedPipelineForDisplay,
} from "../jobs/pipeline";

const TERMINAL = new Set<JobStatus>(["cancelled", "succeeded", "failed", "interrupted"]);

function isCoverageEnabled(data: Record<string, unknown> | undefined): boolean {
  const coverage = data?.coverage;
  if (!coverage || typeof coverage !== "object") return false;
  return Boolean((coverage as { enabled?: unknown }).enabled);
}

function waitingHint(status: JobStatus | null | undefined, connected: boolean): string {
  if (!status) return "Connecting…";
  if (status === "queued") return "Job queued — waiting for worker…";
  if (status === "cancelling") return "Cancelling…";
  if (status === "running") {
    return connected
      ? "Worker running — waiting for first progress event…"
      : "Worker running — reconnecting live progress…";
  }
  return "Waiting for progress…";
}

export function Jobs() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobRecord | null>(null);
  const [jobType, setJobType] = useState<JobType>("build");
  const [force, setForce] = useState(false);
  const [creating, setCreating] = useState(false);
  const [coverageEnabled, setCoverageEnabled] = useState(false);
  const [coverageFlagLoaded, setCoverageFlagLoaded] = useState(false);

  const pageSize = 25;
  const { messages, latest, terminalStatus, connected } = useJobWebSocket(selectedId);
  const selectedMeta = useMemo(() => jobPipelineItem(jobType), [jobType]);

  const loadJobs = useCallback(async (opts?: { silent?: boolean }) => {
    const silent = Boolean(opts?.silent);
    if (!silent) {
      setLoading(true);
      setError(null);
    }
    try {
      const offset = (page - 1) * pageSize;
      const list = await api.listJobs(pageSize, offset);
      setJobs(list);
    } catch (err) {
      if (!silent) {
        setError(err instanceof ApiError ? err.message : "Failed to load jobs");
      }
    } finally {
      if (!silent) setLoading(false);
    }
  }, [page]);

  const loadCoverageFlag = useCallback(async () => {
    try {
      const cfg = await api.getConfig();
      setCoverageEnabled(isCoverageEnabled(cfg.data));
    } catch {
      setCoverageEnabled(false);
    } finally {
      setCoverageFlagLoaded(true);
    }
  }, []);

  const applyJobSnapshot = useCallback((job: JobRecord) => {
    setSelectedJob((prev) => {
      if (
        prev &&
        prev.id === job.id &&
        prev.status === job.status &&
        prev.updated_at === job.updated_at &&
        prev.error_message === job.error_message
      ) {
        return prev;
      }
      return job;
    });
    setJobs((prev) => {
      const idx = prev.findIndex((row) => row.id === job.id);
      if (idx < 0) return prev;
      const row = prev[idx];
      if (
        row.status === job.status &&
        row.updated_at === job.updated_at &&
        row.error_message === job.error_message
      ) {
        return prev;
      }
      const next = [...prev];
      next[idx] = {
        id: job.id,
        job_type: job.job_type,
        status: job.status,
        created_at: job.created_at,
        updated_at: job.updated_at,
        started_at: job.started_at,
        finished_at: job.finished_at,
        error_message: job.error_message,
      };
      return next;
    });
  }, []);

  const refreshSelectedJob = useCallback(
    async (id: string) => {
      try {
        const job = await api.getJob(id);
        applyJobSnapshot(job);
        return job;
      } catch {
        return null;
      }
    },
    [applyJobSnapshot],
  );

  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);

  useEffect(() => {
    void loadCoverageFlag();
  }, [loadCoverageFlag]);

  useEffect(() => {
    if (!selectedId) {
      setSelectedJob(null);
      return;
    }
    void refreshSelectedJob(selectedId);
  }, [selectedId, refreshSelectedJob]);

  useEffect(() => {
    if (terminalStatus) {
      void loadJobs({ silent: true });
      if (selectedId) void refreshSelectedJob(selectedId);
    }
  }, [terminalStatus, loadJobs, selectedId, refreshSelectedJob]);

  // Light status poll only (no full table reload) while the selected job is active.
  useEffect(() => {
    if (!selectedId) return;
    if (selectedJob?.status && TERMINAL.has(selectedJob.status)) return;
    const timer = window.setInterval(() => {
      void refreshSelectedJob(selectedId);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [selectedId, selectedJob?.status, refreshSelectedJob]);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      const job = await api.createJob({ job_type: jobType, force });
      setSelectedId(job.id);
      applyJobSnapshot(job);
      setPage(1);
      await loadJobs({ silent: true });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to create job");
    } finally {
      setCreating(false);
    }
  }

  async function handleCancel(id: string) {
    setError(null);
    try {
      await api.cancelJob(id);
      await loadJobs({ silent: true });
      if (selectedId === id) await refreshSelectedJob(id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to cancel job");
    }
  }

  const activeStatus = selectedJob?.status ?? null;
  const isActive =
    selectedId != null && activeStatus != null && !TERMINAL.has(activeStatus);

  const progressPct =
    latest?.fraction != null
      ? Math.round(latest.fraction * 100)
      : latest?.current != null && latest?.total
        ? Math.round((latest.current / latest.total) * 100)
        : null;

  const phase =
    latest?.metadata && typeof latest.metadata.phase === "string"
      ? String(latest.metadata.phase)
      : null;

  const selectedCount =
    latest?.metadata && typeof latest.metadata.selected === "number"
      ? Number(latest.metadata.selected)
      : null;

  const reusedCount =
    latest?.metadata && typeof latest.metadata.reused === "number"
      ? Number(latest.metadata.reused)
      : null;

  const builtNewCount =
    latest?.metadata && typeof latest.metadata.built_new === "number"
      ? Number(latest.metadata.built_new)
      : null;

  const remainingDeficit =
    latest?.metadata && typeof latest.metadata.remaining_deficit_total === "number"
      ? Number(latest.metadata.remaining_deficit_total)
      : null;

  const durationSec =
    latest?.metadata && typeof latest.metadata.duration_sec === "number"
      ? Number(latest.metadata.duration_sec)
      : null;

  const targetDurationSec =
    latest?.metadata && typeof latest.metadata.target_duration_sec === "number"
      ? Number(latest.metadata.target_duration_sec)
      : null;

  const lastUpdatedLabel = latest?.created_at
    ? `updated ${new Date(latest.created_at).toLocaleTimeString()}`
    : null;

  const showIndeterminate = isActive && progressPct == null;

  const displayPipeline = useMemo(() => orderedPipelineForDisplay(true), []);
  const displayStages = displayPipeline.filter(
    (item) => item.kind === "stage" || item.kind === "coverage-stage",
  );
  const displayOrchestrators = displayPipeline.filter(
    (item) => item.kind === "orchestrator" || item.kind === "coverage-orchestrator",
  );

  return (
    <div>
      <header className="page-header">
        <h1>Jobs</h1>
        <p>
          初めてなら <strong>Build（推奨）</strong> を実行します。希少音素カバレッジは{" "}
          <strong>analyze の前</strong>に差し込まれます（<code>coverage.enabled</code> 時）。
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>パイプライン順</h2>
          <p className="job-pipeline-intro">
            推奨順は <code>scan</code> → <strong>coverage-*</strong> → <code>analyze</code> → … です。
            Build は <code>coverage.enabled</code> のとき coverage を analyze の前に実行し、既解析クリップは再利用します。
            Jobs 上は常に <code>plan-split</code> → <code>select</code> ですが、
            <strong>unseen_speaker</strong> では「話者割当→各バケット内 select」、
            <strong>seen_speaker / single_speaker</strong> では「全体 select→あとでクリップに split」です。
          </p>
          {coverageFlagLoaded && (
            <p
              className="job-pipeline-intro"
              style={{
                marginTop: "0.5rem",
                padding: "0.5rem 0.75rem",
                borderRadius: "6px",
                background: coverageEnabled
                  ? "rgba(34, 160, 90, 0.12)"
                  : "rgba(180, 120, 40, 0.12)",
              }}
            >
              Coverage automation:{" "}
              <strong>{coverageEnabled ? "enabled" : "disabled"}</strong>
              {coverageEnabled
                ? " — Build に coverage が含まれます。"
                : " — Config で coverage.enabled: true にすると Build に含まれます。ジョブ自体は下から単独実行できます（plan/run は enabled が必要）。"}
            </p>
          )}
          <ol className="job-pipeline">
            {displayStages.map((item, index) => (
              <li key={item.type}>
                <button
                  type="button"
                  className={
                    jobType === item.type
                      ? "job-pipeline-item is-selected"
                      : "job-pipeline-item"
                  }
                  onClick={() => setJobType(item.type)}
                >
                  <span className="job-pipeline-step">
                    {item.section === "coverage" ? `C${item.step}` : index + 1}
                  </span>
                  <span className="job-pipeline-body">
                    <span className="job-pipeline-title">
                      <span className="mono">{item.type}</span>
                      <span className="job-pipeline-label">{item.label}</span>
                      {item.section === "coverage" && (
                        <span className="pill" style={{ marginLeft: "0.35rem" }}>
                          coverage
                        </span>
                      )}
                    </span>
                    <span className="job-pipeline-summary">{item.summary}</span>
                    <span className="job-pipeline-produces">出力: {item.produces}</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>
          {displayOrchestrators.map((item) => (
            <button
              key={item.type}
              type="button"
              className={
                jobType === item.type
                  ? "job-pipeline-item job-pipeline-build is-selected"
                  : "job-pipeline-item job-pipeline-build"
              }
              onClick={() => setJobType(item.type)}
            >
              <span className="job-pipeline-step">★</span>
              <span className="job-pipeline-body">
                <span className="job-pipeline-title">
                  <span className="mono">{item.type}</span>
                  <span className="job-pipeline-label">{item.label}</span>
                </span>
                <span className="job-pipeline-summary">{item.summary}</span>
                <span className="job-pipeline-produces">出力: {item.produces}</span>
              </span>
            </button>
          ))}
        </section>

        <section className="card">
          <h2>ジョブを開始</h2>
          <form className="toolbar" onSubmit={handleCreate}>
            <label className="field" style={{ minWidth: 280 }}>
              <span className="label">選択中</span>
              <select
                className="select"
                value={jobType}
                onChange={(e) => setJobType(e.target.value as JobType)}
              >
                <optgroup label="推奨">
                  <option value="build">
                    build — Build（
                    {coverageEnabled ? "scan→coverage→analyze→…" : "1〜6（coverage 無効）"}）
                  </option>
                  <option value="coverage-build">
                    coverage-build — Coverage only（index→run→report）
                  </option>
                </optgroup>
                <optgroup label="個別ステージ（表示順）">
                  {displayStages.map((item, index) => (
                    <option key={item.type} value={item.type}>
                      {item.section === "coverage" ? `C${item.step}` : index + 1}. {item.type} —{" "}
                      {item.label}
                    </option>
                  ))}
                </optgroup>
              </select>
            </label>
            <label
              className="field"
              style={{ flexDirection: "row", alignItems: "center", gap: "0.4rem" }}
              title="既存の段階成果物があっても再実行する"
            >
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
              />
              <span className="label" style={{ margin: 0 }}>
                Force（再実行）
              </span>
            </label>
            <button type="submit" className="btn btn-primary" disabled={creating}>
              {creating ? "Starting…" : `Start ${jobType}`}
            </button>
          </form>
          {selectedMeta && (
            <div className="job-selected-help">
              <strong>{formatJobTypeLabel(jobType)}</strong>
              <p>{selectedMeta.summary}</p>
              <p className="job-pipeline-produces">出力: {selectedMeta.produces}</p>
            </div>
          )}
        </section>

        <section className="card">
          <h2>ジョブ一覧</h2>
          {loading ? (
            <p className="loading">Loading jobs…</p>
          ) : (
            <>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Type</th>
                      <th>Status</th>
                      <th>Created</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {jobs.length === 0 ? (
                      <tr>
                        <td colSpan={5} className="loading">
                          No jobs yet — start Build above
                        </td>
                      </tr>
                    ) : (
                      jobs.map((job) => (
                        <tr
                          key={job.id}
                          className={selectedId === job.id ? "row-selected" : undefined}
                          onClick={() => setSelectedId(job.id)}
                          style={{ cursor: "pointer" }}
                        >
                          <td className="mono truncate" title={job.id}>
                            {job.id.slice(0, 12)}…
                          </td>
                          <td title={jobPipelineItem(job.job_type)?.summary ?? job.job_type}>
                            {formatJobTypeLabel(job.job_type)}
                          </td>
                          <td>
                            <StatusBadge status={job.status} />
                          </td>
                          <td>{new Date(job.created_at).toLocaleString()}</td>
                          <td>
                            {!TERMINAL.has(job.status) && (
                              <button
                                type="button"
                                className="btn btn-sm btn-danger"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  void handleCancel(job.id);
                                }}
                              >
                                Cancel
                              </button>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={page}
                pageSize={pageSize}
                total={
                  jobs.length < pageSize
                    ? (page - 1) * pageSize + jobs.length
                    : page * pageSize + 1
                }
                onPageChange={setPage}
              />
            </>
          )}
        </section>

        {selectedId && (
          <section className="card">
            <h2>
              Live progress{" "}
              <span
                style={{
                  fontSize: "0.75rem",
                  color: connected ? "var(--success)" : "var(--text-muted)",
                }}
              >
                {connected ? "● connected" : "○ disconnected"}
              </span>
            </h2>
            <p className="mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              {selectedId}
            </p>
            {selectedJob && (
              <div className="progress-meta" style={{ marginBottom: "0.5rem" }}>
                <span>Status:</span> <StatusBadge status={selectedJob.status} />
                <span className="pill">{formatJobTypeLabel(selectedJob.job_type)}</span>
              </div>
            )}
            {(latest || showIndeterminate) && (
              <div style={{ marginBottom: "0.75rem" }}>
                <div className="progress-meta">
                  <strong>{latest?.stage ?? "job"}</strong>
                  {phase && <span className="pill">{phase}</span>}
                  {latest?.message ? (
                    <span> — {latest.message}</span>
                  ) : (
                    <span> — {waitingHint(activeStatus, connected)}</span>
                  )}
                </div>
                {(progressPct != null || showIndeterminate) && (
                  <>
                    <div className={`progress-bar${showIndeterminate ? " indeterminate" : ""}`}>
                      <div
                        className="progress-bar-fill"
                        style={
                          progressPct != null ? { width: `${progressPct}%` } : undefined
                        }
                      />
                    </div>
                    <div className="progress-stats">
                      {progressPct != null && <span>{progressPct}%</span>}
                      {latest?.current != null && latest?.total != null && (
                        <span>
                          {latest.current.toLocaleString()} / {latest.total.toLocaleString()}
                        </span>
                      )}
                      {selectedCount != null && <span>selected={selectedCount}</span>}
                      {reusedCount != null && <span>reused={reusedCount}</span>}
                      {builtNewCount != null && <span>built_new={builtNewCount}</span>}
                      {remainingDeficit != null && (
                        <span>deficit_left={remainingDeficit}</span>
                      )}
                      {durationSec != null && targetDurationSec != null && (
                        <span>
                          {(durationSec / 3600).toFixed(3)}h /{" "}
                          {(targetDurationSec / 3600).toFixed(2)}h
                        </span>
                      )}
                      {lastUpdatedLabel && (
                        <span style={{ color: "var(--text-muted)" }}>{lastUpdatedLabel}</span>
                      )}
                    </div>
                  </>
                )}
              </div>
            )}
            {(terminalStatus || (selectedJob && TERMINAL.has(selectedJob.status))) && (
              <p>
                Terminal status:{" "}
                <StatusBadge status={(terminalStatus as JobStatus) ?? selectedJob!.status} />
              </p>
            )}
            {selectedJob?.error_message && (
              <pre
                className="log-panel"
                style={{
                  whiteSpace: "pre-wrap",
                  color: "var(--danger, #c44)",
                  maxHeight: 180,
                  marginBottom: "0.75rem",
                }}
              >
                {selectedJob.error_message}
              </pre>
            )}
            <div className="log-panel">
              {messages.length === 0 ? (
                <div className="log-line loading">
                  {waitingHint(activeStatus, connected)}
                </div>
              ) : (
                messages.map((m, i) => (
                  <div key={m.id ?? i} className="log-line">
                    <span style={{ color: "var(--text-muted)" }}>
                      {new Date(m.created_at).toLocaleTimeString()}
                    </span>{" "}
                    [{m.stage}] {m.message}
                    {m.current != null && m.total != null
                      ? ` (${m.current}/${m.total}`
                      : ""}
                    {m.fraction != null
                      ? `${m.current != null && m.total != null ? ", " : " ("}${Math.round(m.fraction * 100)}%)`
                      : m.current != null && m.total != null
                        ? ")"
                        : ""}
                  </div>
                ))
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
