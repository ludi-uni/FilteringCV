import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { DashboardSummary } from "../api/types";
import { StatusBadge } from "../components/StatusBadge";

export function Dashboard() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const summary = await api.dashboard();
        if (!cancelled) setData(summary);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load dashboard");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) return <p className="loading">Loading dashboard…</p>;
  if (error) return <div className="error-banner">{error}</div>;
  if (!data) return null;

  const jobTotal = Object.values(data.job_status_counts).reduce((a, b) => a + b, 0);

  return (
    <div>
      <header className="page-header">
        <h1>Dashboard</h1>
        <p>Project overview and pipeline status</p>
      </header>

      <div className="stack">
        <section className="card">
          <h2>Paths</h2>
          <div className="grid-2" style={{ marginTop: "0.75rem" }}>
            <div>
              <div className="stat-label">Config</div>
              <code className="mono">{data.config_path}</code>
            </div>
            <div>
              <div className="stat-label">Work dir</div>
              <code className="mono">{data.work_dir}</code>
            </div>
            <div>
              <div className="stat-label">Output dir</div>
              <code className="mono">{data.output_dir}</code>
            </div>
            <div>
              <div className="stat-label">Catalog</div>
              <span style={{ color: data.catalog_ready ? "var(--success)" : "var(--warning)" }}>
                {data.catalog_ready ? "Ready" : "Not built"}
              </span>
            </div>
          </div>
        </section>

        <section className="card">
          <h2>Jobs ({jobTotal})</h2>
          <div className="grid-3" style={{ marginTop: "0.75rem" }}>
            {Object.entries(data.job_status_counts).map(([status, count]) => (
              <div key={status}>
                <div className="stat-value">{count}</div>
                <div className="stat-label">{status}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2>Recent jobs</h2>
            <Link to="/jobs">View all →</Link>
          </div>
          {data.recent_jobs.length === 0 ? (
            <p className="loading">No jobs yet</p>
          ) : (
            <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Status</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent_jobs.map((job) => (
                    <tr key={job.id}>
                      <td className="mono truncate">{job.id.slice(0, 8)}…</td>
                      <td>{job.job_type}</td>
                      <td>
                        <StatusBadge status={job.status} />
                      </td>
                      <td>{new Date(job.created_at).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {data.run_manifest && (
          <section className="card">
            <h2>Run manifest</h2>
            <pre className="json-block">{JSON.stringify(data.run_manifest, null, 2)}</pre>
          </section>
        )}
      </div>
    </div>
  );
}
