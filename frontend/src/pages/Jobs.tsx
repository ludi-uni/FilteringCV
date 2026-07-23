import { FormEvent, useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { JobSummary, JobType } from "../api/types";
import { JOB_TYPES } from "../api/types";
import { Pagination } from "../components/Pagination";
import { StatusBadge } from "../components/StatusBadge";
import { useJobWebSocket } from "../hooks/useJobWebSocket";

const TERMINAL = new Set(["cancelled", "succeeded", "failed", "interrupted"]);

export function Jobs() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [jobType, setJobType] = useState<JobType>("scan");
  const [force, setForce] = useState(false);
  const [creating, setCreating] = useState(false);

  const pageSize = 25;
  const { messages, latest, terminalStatus, connected } = useJobWebSocket(selectedId);

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

  return (
    <div>
      <header className="page-header">
        <h1>Jobs</h1>
        <p>Create, monitor, and cancel pipeline jobs</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>New job</h2>
          <form className="toolbar" onSubmit={handleCreate}>
            <label className="field">
              <span className="label">Type</span>
              <select
                className="select"
                value={jobType}
                onChange={(e) => setJobType(e.target.value as JobType)}
              >
                {JOB_TYPES.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="field" style={{ flexDirection: "row", alignItems: "center", gap: "0.4rem" }}>
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
              />
              <span className="label" style={{ margin: 0 }}>Force</span>
            </label>
            <button type="submit" className="btn btn-primary" disabled={creating}>
              {creating ? "Creating…" : "Start job"}
            </button>
          </form>
        </section>

        <section className="card">
          <h2>Job list</h2>
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
                    {jobs.map((job) => (
                      <tr
                        key={job.id}
                        className={selectedId === job.id ? "row-selected" : undefined}
                        onClick={() => setSelectedId(job.id)}
                        style={{ cursor: "pointer" }}
                      >
                        <td className="mono truncate" title={job.id}>
                          {job.id.slice(0, 12)}…
                        </td>
                        <td>{job.job_type}</td>
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
                    ))}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={page}
                pageSize={pageSize}
                total={jobs.length < pageSize ? (page - 1) * pageSize + jobs.length : page * pageSize + 1}
                onPageChange={setPage}
              />
            </>
          )}
        </section>

        {selectedId && (
          <section className="card">
            <h2>
              Live progress{" "}
              <span style={{ fontSize: "0.75rem", color: connected ? "var(--success)" : "var(--text-muted)" }}>
                {connected ? "● connected" : "○ disconnected"}
              </span>
            </h2>
            <p className="mono" style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              {selectedId}
            </p>
            {latest && (
              <div style={{ marginBottom: "0.75rem" }}>
                <strong>{latest.stage}</strong>
                {latest.message && <span> — {latest.message}</span>}
                {progressPct != null && (
                  <>
                    <div className="progress-bar">
                      <div className="progress-bar-fill" style={{ width: `${progressPct}%` }} />
                    </div>
                    <small style={{ color: "var(--text-muted)" }}>{progressPct}%</small>
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
                    {m.fraction != null && ` (${Math.round(m.fraction * 100)}%)`}
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
