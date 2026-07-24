import { useEffect, useState } from "react";
import { api, ApiError } from "../api/client";
import type { SessionConfigItem } from "../api/types";

interface SetupProps {
  onBound: () => void | Promise<void>;
}

function formatApiError(err: unknown): string {
  if (err instanceof ApiError) {
    return err.message;
  }
  return "Request failed";
}

export function Setup({ onBound }: SetupProps) {
  const [configs, setConfigs] = useState<SessionConfigItem[]>([]);
  const [selectedPath, setSelectedPath] = useState("");
  const [createPath, setCreatePath] = useState("config/default.yaml");
  const [overwrite, setOverwrite] = useState(false);
  const [loadingConfigs, setLoadingConfigs] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoadingConfigs(true);
      setError(null);
      try {
        const response = await api.listSessionConfigs();
        if (cancelled) return;
        setConfigs(response.configs);
        if (response.configs.length > 0) {
          setSelectedPath(response.configs[0].path);
        }
      } catch (err) {
        if (!cancelled) {
          setError(formatApiError(err));
        }
      } finally {
        if (!cancelled) setLoadingConfigs(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleBind() {
    if (!selectedPath) {
      setError("Select a config file first");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.bindSession(selectedPath);
      await onBound();
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleCreate() {
    setBusy(true);
    setError(null);
    try {
      await api.createSession({
        path: createPath.trim() || "config/default.yaml",
        overwrite,
      });
      await onBound();
    } catch (err) {
      setError(formatApiError(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="setup-shell">
      <header className="page-header">
        <h1>Setup</h1>
        <p>Choose or create a config before using the GUI</p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="stack">
        <section className="card">
          <h2>Existing configs</h2>
          {loadingConfigs ? (
            <p className="loading">Loading configs…</p>
          ) : configs.length === 0 ? (
            <p className="loading">No YAML configs found under config/</p>
          ) : (
            <div className="toolbar" style={{ marginTop: "0.75rem", flexWrap: "wrap" }}>
              <label className="field" style={{ flex: 1, minWidth: "220px" }}>
                <span className="label">Config path</span>
                <select
                  className="select"
                  value={selectedPath}
                  onChange={(e) => setSelectedPath(e.target.value)}
                  disabled={busy}
                >
                  {configs.map((item) => (
                    <option key={item.path} value={item.path}>
                      {item.path}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className="btn btn-primary"
                disabled={busy || !selectedPath}
                onClick={() => void handleBind()}
              >
                Use this config
              </button>
            </div>
          )}
        </section>

        <section className="card">
          <h2>Create from example</h2>
          <p style={{ color: "var(--text-secondary)", marginTop: "0.35rem" }}>
            Copy <span className="mono">config/example.yaml</span> to a new path and bind it.
          </p>
          <div className="toolbar" style={{ marginTop: "0.75rem", flexWrap: "wrap" }}>
            <label className="field" style={{ flex: 1, minWidth: "220px" }}>
              <span className="label">Target path</span>
              <input
                className="input mono"
                value={createPath}
                onChange={(e) => setCreatePath(e.target.value)}
                disabled={busy}
                placeholder="config/default.yaml"
              />
            </label>
            <label
              className="field"
              style={{
                flexDirection: "row",
                alignItems: "center",
                gap: "0.5rem",
                marginTop: "1.1rem",
              }}
            >
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(e) => setOverwrite(e.target.checked)}
                disabled={busy}
              />
              <span>Overwrite if exists</span>
            </label>
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy}
              onClick={() => void handleCreate()}
              style={{ marginTop: "1.1rem" }}
            >
              Create from example
            </button>
          </div>
        </section>
      </div>
    </div>
  );
}
