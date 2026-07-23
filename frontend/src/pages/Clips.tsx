import { FormEvent, useCallback, useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ClipItem, OverrideAction } from "../api/types";
import { DISPOSITIONS, OVERRIDE_ACTIONS } from "../api/types";
import { Pagination } from "../components/Pagination";

const PAGE_SIZE = 50;

export function Clips() {
  const [clips, setClips] = useState<ClipItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [speakerId, setSpeakerId] = useState("");
  const [disposition, setDisposition] = useState("");
  const [search, setSearch] = useState("");
  const [qualityMin, setQualityMin] = useState("");
  const [appliedFilters, setAppliedFilters] = useState({
    speakerId: "",
    disposition: "",
    search: "",
    qualityMin: "",
  });

  const [playingId, setPlayingId] = useState<string | null>(null);
  const [overrideClipId, setOverrideClipId] = useState("");
  const [overrideAction, setOverrideAction] = useState<OverrideAction>("force_include");
  const [overrideReason, setOverrideReason] = useState("");
  const [overrideBusy, setOverrideBusy] = useState(false);

  const loadClips = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.listClips({
        page,
        page_size: PAGE_SIZE,
        speaker_id: appliedFilters.speakerId || undefined,
        disposition: appliedFilters.disposition || undefined,
        search: appliedFilters.search || undefined,
      });
      let items = result.items as ClipItem[];
      const qMin = appliedFilters.qualityMin
        ? parseFloat(appliedFilters.qualityMin)
        : NaN;
      if (!Number.isNaN(qMin)) {
        items = items.filter(
          (c) => c.quality_score != null && c.quality_score >= qMin,
        );
      }
      setClips(items);
      setTotal(result.total);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load clips");
    } finally {
      setLoading(false);
    }
  }, [page, appliedFilters]);

  useEffect(() => {
    void loadClips();
  }, [loadClips]);

  function applyFilters(e: FormEvent) {
    e.preventDefault();
    setPage(1);
    setAppliedFilters({ speakerId, disposition, search, qualityMin });
  }

  async function submitOverride(e: FormEvent) {
    e.preventDefault();
    if (!overrideClipId.trim()) return;
    setOverrideBusy(true);
    setError(null);
    try {
      await api.upsertOverride({
        clip_id: overrideClipId.trim(),
        action: overrideAction,
        reason: overrideReason || null,
      });
      setOverrideClipId("");
      setOverrideReason("");
      await loadClips();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to save override");
    } finally {
      setOverrideBusy(false);
    }
  }

  function setOverrideForClip(clipId: string) {
    setOverrideClipId(clipId);
    window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  }

  return (
    <div>
      <header className="page-header">
        <h1>Clips</h1>
        <p>Browse catalog clips, play audio, apply overrides</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>Filters</h2>
          <form className="toolbar" onSubmit={applyFilters}>
            <label className="field">
              <span className="label">Speaker ID</span>
              <input
                className="input"
                value={speakerId}
                onChange={(e) => setSpeakerId(e.target.value)}
                placeholder="speaker_…"
              />
            </label>
            <label className="field">
              <span className="label">Disposition</span>
              <select
                className="select"
                value={disposition}
                onChange={(e) => setDisposition(e.target.value)}
              >
                <option value="">Any</option>
                {DISPOSITIONS.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span className="label">Search</span>
              <input
                className="input"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="clip_id, text, speaker…"
              />
            </label>
            <label className="field">
              <span className="label">Min quality (page)</span>
              <input
                className="input"
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={qualityMin}
                onChange={(e) => setQualityMin(e.target.value)}
                placeholder="0.0–1.0"
                title="Client-side filter on current page only — API has no quality param"
              />
            </label>
            <button type="submit" className="btn btn-primary">
              Apply
            </button>
          </form>
        </section>

        <section className="card">
          <h2>Clip list</h2>
          {loading ? (
            <p className="loading">Loading clips…</p>
          ) : (
            <>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Clip ID</th>
                      <th>Speaker</th>
                      <th>Disposition</th>
                      <th>Quality</th>
                      <th>Duration</th>
                      <th>Text</th>
                      <th>Audio</th>
                      <th>Override</th>
                    </tr>
                  </thead>
                  <tbody>
                    {clips.length === 0 ? (
                      <tr>
                        <td colSpan={8} className="loading">
                          No clips found
                        </td>
                      </tr>
                    ) : (
                      clips.map((clip) => (
                        <tr key={clip.clip_id}>
                          <td className="mono truncate" title={clip.clip_id}>
                            {clip.clip_id}
                          </td>
                          <td className="mono">{clip.speaker_id}</td>
                          <td>{clip.disposition}</td>
                          <td>
                            {clip.quality_score != null
                              ? clip.quality_score.toFixed(3)
                              : "—"}
                          </td>
                          <td>{clip.duration_sec?.toFixed(2)}s</td>
                          <td className="truncate" title={clip.text_norm}>
                            {clip.text_norm}
                          </td>
                          <td>
                            {clip.audio_cache_rel_path ? (
                              <>
                                <button
                                  type="button"
                                  className="btn btn-sm"
                                  onClick={() => setPlayingId(clip.clip_id)}
                                >
                                  Play
                                </button>
                                {playingId === clip.clip_id && (
                                  <audio
                                    controls
                                    autoPlay
                                    src={api.audioUrl(clip.audio_cache_rel_path)}
                                    style={{ display: "block", marginTop: "0.25rem", maxWidth: 200 }}
                                  />
                                )}
                              </>
                            ) : (
                              "—"
                            )}
                          </td>
                          <td>
                            <button
                              type="button"
                              className="btn btn-sm"
                              onClick={() => setOverrideForClip(clip.clip_id)}
                            >
                              Override
                            </button>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              <Pagination
                page={page}
                pageSize={PAGE_SIZE}
                total={total}
                onPageChange={setPage}
              />
            </>
          )}
        </section>

        <section className="card">
          <h2>Upsert override</h2>
          <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Uses <code>PUT /api/overrides</code> per backend API.
          </p>
          <form className="toolbar" onSubmit={submitOverride}>
            <label className="field">
              <span className="label">Clip ID</span>
              <input
                className="input"
                required
                value={overrideClipId}
                onChange={(e) => setOverrideClipId(e.target.value)}
              />
            </label>
            <label className="field">
              <span className="label">Action</span>
              <select
                className="select"
                value={overrideAction}
                onChange={(e) => setOverrideAction(e.target.value as OverrideAction)}
              >
                {OVERRIDE_ACTIONS.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
            </label>
            <label className="field" style={{ flex: 1, minWidth: 200 }}>
              <span className="label">Reason</span>
              <input
                className="input"
                value={overrideReason}
                onChange={(e) => setOverrideReason(e.target.value)}
                placeholder="optional"
              />
            </label>
            <button type="submit" className="btn btn-primary" disabled={overrideBusy}>
              {overrideBusy ? "Saving…" : "Save override"}
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}
