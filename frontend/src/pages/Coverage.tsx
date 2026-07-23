import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { CoverageReport, FeatureCoverageEntry } from "../api/types";
import { Pagination } from "../components/Pagination";

const PAGE_SIZE = 50;

export function Coverage() {
  const [report, setReport] = useState<CoverageReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [featureType, setFeatureType] = useState("");
  const [page, setPage] = useState(1);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await api.coverageReport();
        if (!cancelled) setReport(data);
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof ApiError ? err.message : "Failed to load coverage");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    if (!report) return [];
    let rows: FeatureCoverageEntry[] = report.entries;
    if (featureType) {
      rows = rows.filter((e) => e.feature_type === featureType);
    }
    const needle = search.trim().toLowerCase();
    if (needle) {
      rows = rows.filter(
        (e) =>
          e.feature.toLowerCase().includes(needle) ||
          e.feature_type.toLowerCase().includes(needle),
      );
    }
    return rows;
  }, [report, featureType, search]);

  const pageRows = useMemo(() => {
    const start = (page - 1) * PAGE_SIZE;
    return filtered.slice(start, start + PAGE_SIZE);
  }, [filtered, page]);

  useEffect(() => {
    setPage(1);
  }, [search, featureType]);

  if (loading) return <p className="loading">Loading coverage report…</p>;
  if (error) return <div className="error-banner">{error}</div>;
  if (!report) return null;

  return (
    <div>
      <header className="page-header">
        <h1>Coverage</h1>
        <p>Feature pool coverage from catalog</p>
      </header>

      <div className="stack">
        <section className="card">
          <div className="grid-3">
            <div>
              <div className="stat-value">{report.total_clips.toLocaleString()}</div>
              <div className="stat-label">Total clips</div>
            </div>
            <div>
              <div className="stat-value">{report.eligible_clips.toLocaleString()}</div>
              <div className="stat-label">Eligible</div>
            </div>
            <div>
              <div className="stat-value">{report.unique_features.toLocaleString()}</div>
              <div className="stat-label">Unique features</div>
            </div>
          </div>
          {Object.keys(report.js_distance_to_uniform).length > 0 && (
            <div style={{ marginTop: "1rem" }}>
              <h2>JS distance to uniform</h2>
              <div className="grid-3" style={{ marginTop: "0.5rem" }}>
                {Object.entries(report.js_distance_to_uniform).map(([ft, dist]) => (
                  <div key={ft}>
                    <div className="stat-value" style={{ fontSize: "1.1rem" }}>
                      {dist.toFixed(4)}
                    </div>
                    <div className="stat-label">{ft}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Feature entries</h2>
          <div className="toolbar">
            <label className="field">
              <span className="label">Search</span>
              <input
                className="input"
                placeholder="Feature or type…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="label">Feature type</span>
              <select
                className="select"
                value={featureType}
                onChange={(e) => setFeatureType(e.target.value)}
              >
                <option value="">All types</option>
                {report.feature_types.map((ft) => (
                  <option key={ft} value={ft}>
                    {ft}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Type</th>
                  <th>Feature</th>
                  <th>Pool count</th>
                  <th>Speakers</th>
                  <th>Utterances</th>
                </tr>
              </thead>
              <tbody>
                {pageRows.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="loading">
                      No matching entries
                    </td>
                  </tr>
                ) : (
                  pageRows.map((row) => (
                    <tr key={`${row.feature_type}:${row.feature}`}>
                      <td>{row.feature_type}</td>
                      <td className="mono">{row.feature}</td>
                      <td>{row.pool_count.toLocaleString()}</td>
                      <td>{row.pool_speaker_count.toLocaleString()}</td>
                      <td>{row.pool_utterance_count.toLocaleString()}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <Pagination
            page={page}
            pageSize={PAGE_SIZE}
            total={filtered.length}
            onPageChange={setPage}
          />
        </section>
      </div>
    </div>
  );
}
