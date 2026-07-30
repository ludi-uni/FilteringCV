import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { ConfigResponse } from "../api/types";
import { FieldEditor } from "./ConfigFields";

type EditorMode = "form" | "yaml";
type FilterChip = "all" | "filters" | "builder" | "audio" | "gates" | "changed";

const FILTER_GROUPS: Record<Exclude<FilterChip, "all" | "changed">, string[]> = {
  filters: ["speakers", "input"],
  builder: ["dataset_builder", "compute", "coverage"],
  audio: [
    "quality_gate",
    "early_audio_gate",
    "audio_pipeline",
    "audio_pipeline_align",
    "audio_pipeline_enhance",
    "two_pass_denoise",
    "snr",
  ],
  gates: ["mfa_gate", "nfa_gate", "asr_gate"],
};

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function setAtPath(root: Record<string, unknown>, path: string[], value: unknown): void {
  let cursor: Record<string, unknown> = root;
  for (let i = 0; i < path.length - 1; i += 1) {
    const key = path[i];
    const next = cursor[key];
    if (next == null || typeof next !== "object" || Array.isArray(next)) {
      cursor[key] = {};
    }
    cursor = cursor[key] as Record<string, unknown>;
  }
  cursor[path[path.length - 1]] = value;
}

function pathsEqual(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function matchesSearch(path: string, value: unknown, query: string): boolean {
  if (!query.trim()) return true;
  const q = query.trim().toLowerCase();
  if (path.toLowerCase().includes(q)) return true;
  try {
    return JSON.stringify(value).toLowerCase().includes(q);
  } catch {
    return false;
  }
}

export function Config() {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [meta, setMeta] = useState<ConfigResponse | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const [baseline, setBaseline] = useState<Record<string, unknown>>({});
  const [yamlText, setYamlText] = useState("");
  const [mode, setMode] = useState<EditorMode>("form");
  const [section, setSection] = useState<string>("speakers");
  const [search, setSearch] = useState("");
  const [chip, setChip] = useState<FilterChip>("filters");
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const cfg = await api.getConfig();
      setMeta(cfg);
      setDraft(clone(cfg.data));
      setBaseline(clone(cfg.data));
      setYamlText(cfg.yaml_text);
      const preferred =
        cfg.sections.find((s) => s.id === "speakers")?.id ??
        cfg.sections[0]?.id ??
        "input";
      setSection(preferred);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load config");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const dirty = useMemo(() => !pathsEqual(draft, baseline), [draft, baseline]);

  // Sidebar filter only — never unmount the active section editor while typing.
  // Search matches against baseline so live edits cannot hide the current section.
  const visibleSections = useMemo(() => {
    if (!meta) return [];
    return meta.sections.filter((s) => {
      if (chip === "changed") {
        return !pathsEqual(draft[s.id], baseline[s.id]) || s.id === section;
      }
      if (chip !== "all") {
        const allowed = FILTER_GROUPS[chip];
        if (!allowed.includes(s.id) && s.id !== "schema_version") {
          if (s.group === "meta") return false;
          return false;
        }
      }
      if (!search.trim()) return true;
      if (s.id === section) return true;
      return matchesSearch(s.id, baseline[s.id], search);
    });
  }, [meta, chip, draft, baseline, search, section]);

  useEffect(() => {
    if (visibleSections.length === 0) return;
    if (!visibleSections.some((s) => s.id === section)) {
      setSection(visibleSections[0].id);
    }
  }, [visibleSections, section]);

  function updatePath(path: string[], value: unknown) {
    setDraft((prev) => {
      const next = clone(prev);
      setAtPath(next, path, value);
      return next;
    });
    setNotice(null);
  }

  async function handleValidate() {
    setError(null);
    setValidationErrors([]);
    try {
      const result =
        mode === "yaml"
          ? await api.validateConfig({ yaml_text: yamlText })
          : await api.validateConfig({ data: draft });
      if (!result.ok) {
        setValidationErrors(result.errors);
        setError("Validation failed");
        return false;
      }
      if (result.data) {
        setDraft(clone(result.data));
      }
      setNotice("Config is valid");
      return true;
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Validation failed");
      return false;
    }
  }

  async function handleSave() {
    setSaving(true);
    setError(null);
    setNotice(null);
    setValidationErrors([]);
    try {
      const saved =
        mode === "yaml"
          ? await api.saveConfig({ yaml_text: yamlText, mode: "yaml" })
          : await api.saveConfig({ data: draft, mode: "data" });
      setDraft(clone(saved.data));
      setBaseline(clone(saved.data));
      setYamlText(saved.yaml_text);
      setNotice(saved.message);
      if (meta) {
        setMeta({ ...meta, data: saved.data, yaml_text: saved.yaml_text });
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  function handleDiscard() {
    setDraft(clone(baseline));
    if (meta) setYamlText(meta.yaml_text);
    setNotice("Discarded unsaved changes");
    setValidationErrors([]);
  }

  const sectionValue = draft[section];
  const sectionOriginal = baseline[section];

  if (loading) {
    return <p className="loading">Loading config…</p>;
  }

  return (
    <div>
      <header className="page-header">
        <h1>Config</h1>
        <p>
          Edit pipeline settings and overwrite{" "}
          <span className="mono">{meta?.relative_path ?? "config.yaml"}</span>
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {notice && <div className="notice-banner">{notice}</div>}
      {validationErrors.length > 0 && (
        <div className="error-banner">
          <ul style={{ margin: 0, paddingLeft: "1.1rem" }}>
            {validationErrors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="config-toolbar card">
        <div className="toolbar" style={{ flexWrap: "wrap" }}>
          <label className="field" style={{ minWidth: "220px", flex: 1 }}>
            <span className="label">Search</span>
            <input
              className="input"
              placeholder="Filter keys, speakers, values…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
          <div className="chip-row">
            {(
              [
                ["all", "All"],
                ["filters", "Filters"],
                ["builder", "Builder"],
                ["audio", "Audio"],
                ["gates", "Gates"],
                ["changed", "Changed"],
              ] as Array<[FilterChip, string]>
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={`chip${chip === id ? " chip-active" : ""}`}
                onClick={() => setChip(id)}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="chip-row">
            <button
              type="button"
              className={`chip${mode === "form" ? " chip-active" : ""}`}
              onClick={() => setMode("form")}
            >
              Form
            </button>
            <button
              type="button"
              className={`chip${mode === "yaml" ? " chip-active" : ""}`}
              onClick={() => {
                void (async () => {
                  try {
                    const res = await api.validateConfig({ data: draft });
                    if (res.ok && res.yaml_text) {
                      setYamlText(res.yaml_text);
                      if (res.data) setDraft(clone(res.data));
                    }
                  } catch {
                    // keep existing yaml text
                  }
                  setMode("yaml");
                })();
              }}
            >
              YAML
            </button>
          </div>
        </div>
        <div className="toolbar" style={{ marginTop: "0.75rem" }}>
          <button type="button" className="btn" onClick={() => void handleValidate()}>
            Validate
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={saving || (!dirty && mode === "form")}
            onClick={() => void handleSave()}
          >
            {saving ? "Saving…" : "Save & overwrite YAML"}
          </button>
          <button type="button" className="btn" disabled={!dirty} onClick={handleDiscard}>
            Discard
          </button>
          <button type="button" className="btn" onClick={() => void load()}>
            Reload
          </button>
          {dirty && <span className="pill">unsaved changes</span>}
        </div>
      </div>

      {mode === "yaml" ? (
        <section className="card" style={{ marginTop: "1rem" }}>
          <h2>YAML editor</h2>
          <textarea
            className="textarea mono"
            rows={28}
            value={yamlText}
            onChange={(e) => {
              setYamlText(e.target.value);
              setNotice(null);
            }}
          />
        </section>
      ) : (
        <div className="config-layout">
          <aside className="config-nav card">
            <h2>Sections</h2>
            <ul className="config-section-list">
              {visibleSections.map((s) => {
                const changed = !pathsEqual(draft[s.id], baseline[s.id]);
                return (
                  <li key={s.id}>
                    <button
                      type="button"
                      className={`config-section-btn${section === s.id ? " active" : ""}`}
                      onClick={() => setSection(s.id)}
                    >
                      <span>{s.label}</span>
                      {changed && <span className="dirty-dot" />}
                    </button>
                  </li>
                );
              })}
            </ul>
          </aside>
          <section className="card config-editor">
            <h2>{meta?.sections.find((s) => s.id === section)?.label ?? section}</h2>
            {sectionValue === undefined ? (
              <p className="loading">Section not present in this config (using defaults).</p>
            ) : (
              <FieldEditor
                path={[section]}
                value={sectionValue}
                original={sectionOriginal}
                onChange={updatePath}
              />
            )}
            {section === "speakers" && (
              <div className="config-callout">
                <strong>Filtering tips</strong>
                <ul>
                  <li>
                    <code>include_client_ids</code>: leave empty for all speakers; paste one ID per
                    line to restrict.
                  </li>
                  <li>
                    <code>clip_metadata_filters</code>: empty lists disable that axis; add{" "}
                    <code>&quot;&quot;</code> to allow blank TSV cells.
                  </li>
                  <li>
                    <code>input.max_clips</code>: set a small number for smoke tests; use null for
                    full corpus.
                  </li>
                </ul>
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
