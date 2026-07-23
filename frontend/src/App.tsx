import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { Clips } from "./pages/Clips";
import { Compare } from "./pages/Compare";
import { Config } from "./pages/Config";
import { Coverage } from "./pages/Coverage";
import { Dashboard } from "./pages/Dashboard";
import { Jobs } from "./pages/Jobs";

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
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
