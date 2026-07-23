import { FormEvent, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CompareResult } from "../api/types";

export function Compare() {
  const [left, setLeft] = useState("work");
  const [right, setRight] = useState("work_prev");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await api.compare({ left, right });
      setResult(data);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Compare failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1>Compare runs</h1>
        <p>Diff two work or output directories under the project root</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>Run paths</h2>
          <form className="toolbar" onSubmit={handleSubmit}>
            <label className="field" style={{ flex: 1, minWidth: 200 }}>
              <span className="label">Left path</span>
              <input
                className="input"
                value={left}
                onChange={(e) => setLeft(e.target.value)}
                placeholder="work"
                required
              />
            </label>
            <label className="field" style={{ flex: 1, minWidth: 200 }}>
              <span className="label">Right path</span>
              <input
                className="input"
                value={right}
                onChange={(e) => setRight(e.target.value)}
                placeholder="work_prev"
                required
              />
            </label>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? "Comparing…" : "Compare"}
            </button>
          </form>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "0.5rem" }}>
            Paths are relative to the project root. Backend resolves and validates them.
          </p>
        </section>

        {result && (
          <>
            <section className="card">
              <h2>Summary</h2>
              <div className="grid-3">
                <div>
                  <div className="stat-label">Left selected</div>
                  <div className="stat-value">
                    {result.speaker_counts.left_selected_clips}
                  </div>
                </div>
                <div>
                  <div className="stat-label">Right selected</div>
                  <div className="stat-value">
                    {result.speaker_counts.right_selected_clips}
                  </div>
                </div>
                <div>
                  <div className="stat-label">Unchanged</div>
                  <div className="stat-value">
                    {result.clip_set_diff.unchanged_count}
                  </div>
                </div>
              </div>
              {result.config_diff && Object.keys(result.config_diff).length > 0 && (
                <div style={{ marginTop: "1rem" }}>
                  <h2>Config diff</h2>
                  <pre className="json-block">
                    {JSON.stringify(result.config_diff, null, 2)}
                  </pre>
                </div>
              )}
            </section>

            <section className="card">
              <h2>Clip set diff</h2>
              <div className="grid-2">
                <div>
                  <div className="stat-label">Added ({result.clip_set_diff.added.length})</div>
                  <pre className="json-block" style={{ maxHeight: 200 }}>
                    {result.clip_set_diff.added.slice(0, 100).join("\n")}
                    {result.clip_set_diff.added.length > 100 && "\n…"}
                  </pre>
                </div>
                <div>
                  <div className="stat-label">Removed ({result.clip_set_diff.removed.length})</div>
                  <pre className="json-block" style={{ maxHeight: 200 }}>
                    {result.clip_set_diff.removed.slice(0, 100).join("\n")}
                    {result.clip_set_diff.removed.length > 100 && "\n…"}
                  </pre>
                </div>
              </div>
            </section>

            {Object.keys(result.duration_delta_sec).length > 0 && (
              <section className="card">
                <h2>Duration delta (sec)</h2>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Stage</th>
                        <th>Δ seconds</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.duration_delta_sec).map(([stage, delta]) => (
                        <tr key={stage}>
                          <td>{stage}</td>
                          <td className="mono">{delta.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            {Object.keys(result.coverage_js_delta).length > 0 && (
              <section className="card">
                <h2>Coverage JS delta</h2>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature type</th>
                        <th>JS distance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.coverage_js_delta).map(([ft, dist]) => (
                        <tr key={ft}>
                          <td>{ft}</td>
                          <td className="mono">{dist.toFixed(6)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}

            <section className="card">
              <h2>Full response</h2>
              <pre className="json-block">{JSON.stringify(result, null, 2)}</pre>
            </section>
          </>
        )}
      </div>
    </div>
  );
}
