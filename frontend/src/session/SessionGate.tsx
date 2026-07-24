import { useCallback, useEffect, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { SessionState } from "../api/types";
import { Layout } from "../components/Layout";
import { Clips } from "../pages/Clips";
import { Compare } from "../pages/Compare";
import { Config } from "../pages/Config";
import { Coverage } from "../pages/Coverage";
import { Dashboard } from "../pages/Dashboard";
import { Jobs } from "../pages/Jobs";
import { Setup } from "../pages/Setup";

const SWITCH_BLOCKED_MESSAGE = "Finish or cancel jobs before switching";

export function SessionGate() {
  const [session, setSession] = useState<SessionState | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [switchError, setSwitchError] = useState<string | null>(null);
  const [switching, setSwitching] = useState(false);

  const refresh = useCallback(async () => {
    const next = await api.session();
    setSession(next);
    setLoadError(null);
    return next;
  }, []);

  const loadInitialSession = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const next = await api.session();
      setSession(next);
    } catch (err) {
      setSession(null);
      setLoadError(
        err instanceof ApiError
          ? err.message
          : err instanceof Error
            ? err.message
            : "Failed to load session",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadInitialSession();
  }, [loadInitialSession]);

  async function handleBound() {
    setSwitchError(null);
    await refresh();
  }

  async function handleSwitchConfig() {
    setSwitching(true);
    setSwitchError(null);
    try {
      await api.unbindSession();
      await refresh();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setSwitchError(SWITCH_BLOCKED_MESSAGE);
      } else {
        setSwitchError(err instanceof ApiError ? err.message : "Failed to switch config");
      }
    } finally {
      setSwitching(false);
    }
  }

  if (loading) {
    return <p className="loading" style={{ padding: "2rem" }}>Loading…</p>;
  }

  // Network/API failure: show error + retry. Do not treat as unbound Setup.
  if (loadError || session == null) {
    return (
      <div style={{ padding: "1.5rem 2rem", maxWidth: "720px", margin: "0 auto" }}>
        <div className="error-banner">{loadError ?? "Failed to load session"}</div>
        <button type="button" className="btn btn-primary" onClick={() => void loadInitialSession()}>
          Retry
        </button>
      </div>
    );
  }

  // Only show Setup when the API explicitly reports bound: false.
  if (!session.bound) {
    return (
      <div style={{ padding: "1.5rem 2rem", maxWidth: "720px", margin: "0 auto" }}>
        <Setup onBound={handleBound} />
      </div>
    );
  }

  return (
    <BrowserRouter>
      {switchError && (
        <div className="error-banner" style={{ margin: "0.75rem 1rem 0" }}>
          {switchError}
        </div>
      )}
      <Routes>
        <Route
          element={
            <Layout
              configPath={session.config_path}
              onSwitchConfig={() => void handleSwitchConfig()}
              switchDisabled={switching}
            />
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="jobs" element={<Jobs />} />
          <Route path="config" element={<Config />} />
          <Route path="coverage" element={<Coverage />} />
          <Route path="clips" element={<Clips />} />
          <Route path="compare" element={<Compare />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
