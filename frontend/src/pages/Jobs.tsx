import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { JobSummary, JobType } from "../api/types";
import { Pagination } from "../components/Pagination";
import { StatusBadge } from "../components/StatusBadge";
import { useJobWebSocket } from "../hooks/useJobWebSocket";
import {
  JOB_PIPELINE,
  formatJobTypeLabel,
  jobPipelineItem,
} from "../jobs/pipeline";

const TERMINAL = new Set(["cancelled", "succeeded", "failed", "interrupted"]);

export function Jobs() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [jobType, setJobType] = useState<JobType>("build");
  const [force, setForce] = useState(false);
  const [creating, setCreating] = useState(false);

  const pageSize = 25;
  const { messages, latest, terminalStatus, connected } = useJobWebSocket(selectedId);
  const selectedMeta = useMemo(() => jobPipelineItem(jobType), [jobType]);

  const loadJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const offset = (page - 1) * pageSize;
      const list = await api.listJobs(pageSize, offset);
      setJobs(list);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load jobs");
    } finally {
      setLoading(false);
    }
  }, [page]);

  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);

  useEffect(() => {
    if (terminalStatus) {
      void loadJobs();
    }
  }, [terminalStatus, loadJobs]);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    setCreating(true);
    setError(null);
    try {
      const job = await api.createJob({ job_type: jobType, force });
      setSelectedId(job.id);
      setPage(1);
      await loadJobs();
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
      await loadJobs();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to cancel job");
    }
  }

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

  const showIndeterminate =
    selectedId != null &&
    !terminalStatus &&
    connected &&
    progressPct == null &&
    messages.length > 0;

  return (
    <div>
      <header className="page-header">
        <h1>Jobs</h1>
        <p>
          データセットビルダーの段階実行。初めてなら{" "}
          <strong>Build（推奨）</strong> で 1〜6 をまとめて実行します。
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>パイプライン順</h2>
          <p className="job-pipeline-intro">
            通常は上から下へ進みます。<code>Build</code> は同じ順序をオーケストレートし、途中成果物があれば再開します。
            個別ステージは再実行や途中からのやり直し用です。
          </p>
          <ol className="job-pipeline">
            {JOB_PIPELINE.filter((item) => item.kind === "stage").map((item) => (
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
                  <span className="job-pipeline-step">{item.step}</span>
                  <span className="job-pipeline-body">
                    <span className="job-pipeline-title">
                      <span className="mono">{item.type}</span>
                      <span className="job-pipeline-label">{item.label}</span>
                    </span>
                    <span className="job-pipeline-summary">{item.summary}</span>
                    <span className="job-pipeline-produces">出力: {item.produces}</span>
                  </span>
                </button>
              </li>
            ))}
          </ol>
          {JOB_PIPELINE.filter((item) => item.kind === "orchestrator").map((item) => (
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
            <label className="field" style={{ minWidth: 220 }}>
              <span className="label">選択中</span>
              <select
                className="select"
                value={jobType}
                onChange={(e) => setJobType(e.target.value as JobType)}
              >
                <optgroup label="推奨">
                  <option value="build">build — Build（推奨・1〜6）</option>
                </optgroup>
                <optgroup label="個別ステージ（順番どおり）">
                  {JOB_PIPELINE.filter((item) => item.kind === "stage").map((item) => (
                    <option key={item.type} value={item.type}>
                      {item.step}. {item.type} — {item.label}
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
            {latest && (
              <div style={{ marginBottom: "0.75rem" }}>
                <div className="progress-meta">
                  <strong>{latest.stage}</strong>
                  {phase && <span className="pill">{phase}</span>}
                  {latest.message && <span> — {latest.message}</span>}
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
                      {latest.current != null && latest.total != null && (
                        <span>
                          {latest.current.toLocaleString()} / {latest.total.toLocaleString()}
                        </span>
                      )}
                      {selectedCount != null && <span>selected={selectedCount}</span>}
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
            {terminalStatus && (
              <p>
                Terminal status: <StatusBadge status={terminalStatus} />
              </p>
            )}
            <div className="log-panel">
              {messages.length === 0 ? (
                <div className="log-line loading">Waiting for progress…</div>
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
