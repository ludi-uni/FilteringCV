import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type {
  CoverageAutomationReport,
  CoverageReport,
  FeatureCoverageEntry,
} from "../api/types";
import { Pagination } from "../components/Pagination";

const PAGE_SIZE = 50;

export function Coverage() {
  const [report, setReport] = useState<CoverageReport | null>(null);
  const [automation, setAutomation] = useState<CoverageAutomationReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [featureType, setFeatureType] = useState("");
  const [page, setPage] = useState(1);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [pool, auto] = await Promise.all([
          api.coverageReport().catch((err: unknown) => {
            if (err instanceof ApiError && err.status === 404) return null;
            throw err;
          }),
          api.coverageAutomationReport(),
        ]);
        if (!cancelled) {
          setReport(pool);
          setAutomation(auto);
        }
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

  const summary = automation?.summary;

  return (
    <div>
      <header className="page-header">
        <h1>Coverage</h1>
        <p>カタログの特徴量プールと、希少音素カバレッジ自動化の実行状況</p>
      </header>

      <div className="stack">
        <section className="card">
          <h2>希少音素カバレッジ自動化</h2>
          {!automation?.enabled ? (
            <p>
              無効です。Config で <code>coverage.enabled: true</code> を設定すると、Build が{" "}
              <strong>analyze の前</strong>にカバレッジ確保を実行します。
            </p>
          ) : !automation.run_ready ? (
            <p>
              {automation.message || "まだ active-run がありません。"}{" "}
              <Link to="/jobs">Jobs</Link> で <code>build</code>（推奨）または{" "}
              <code>coverage-build</code> を開始してください。analyze より先に走ります。
              {automation.index_ready ? " （index は作成済み）" : ""}
            </p>
          ) : (
            <>
              <div className="grid-3">
                <div>
                  <div className="stat-value" style={{ fontSize: "1.1rem" }}>
                    {summary?.status ?? "—"}
                  </div>
                  <div className="stat-label">Status</div>
                </div>
                <div>
                  <div className="stat-value">
                    {(summary?.iteration ?? 0).toLocaleString()}
                  </div>
                  <div className="stat-label">Iterations</div>
                </div>
                <div>
                  <div className="stat-value">
                    {(summary?.remaining_deficit_total ?? 0).toLocaleString()}
                  </div>
                  <div className="stat-label">Remaining deficit</div>
                </div>
              </div>
              <div className="grid-3" style={{ marginTop: "0.75rem" }}>
                <div>
                  <div className="stat-value">
                    {(summary?.analyzed_clips ?? 0).toLocaleString()}
                  </div>
                  <div className="stat-label">Analyzed</div>
                </div>
                <div>
                  <div className="stat-value">
                    {(summary?.accepted_clips ?? 0).toLocaleString()}
                  </div>
                  <div className="stat-label">Accepted</div>
                </div>
                <div>
                  <div className="stat-value">
                    {summary?.overall_pass_rate != null
                      ? summary.overall_pass_rate.toFixed(3)
                      : "—"}
                  </div>
                  <div className="stat-label">Pass rate</div>
                </div>
              </div>
              <p className="mono" style={{ marginTop: "0.75rem", fontSize: "0.85rem" }}>
                {automation.run_dir}
              </p>
              {summary?.features && summary.features.length > 0 && (
                <div className="table-wrap" style={{ marginTop: "0.75rem" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Target</th>
                        <th>Accepted</th>
                        <th>Deficit</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {summary.features.slice(0, 40).map((row) => (
                        <tr key={row.feature}>
                          <td className="mono">{row.feature}</td>
                          <td>{row.target}</td>
                          <td>{row.accepted_after ?? 0}</td>
                          <td>{row.deficit ?? 0}</td>
                          <td>{row.status ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </>
          )}
        </section>

        {!report ? (
          <section className="card">
            <h2>カタログ特徴量プール</h2>
            <p>
              カタログがまだありません。<Link to="/jobs">Jobs</Link> で analyze / Build
              を実行してください。
            </p>
          </section>
        ) : (
          <>
            <section className="card">
              <h2>カタログ特徴量プール</h2>
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
                  <h3>JS distance to uniform</h3>
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
          </>
        )}
      </div>
    </div>
  );
}
