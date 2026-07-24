import { NavLink, Outlet } from "react-router-dom";
import styles from "./Layout.module.css";

const NAV: Array<{ to: string; label: string; end?: boolean }> = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/jobs", label: "Jobs" },
  { to: "/config", label: "Config" },
  { to: "/coverage", label: "Coverage" },
  { to: "/clips", label: "Clips" },
  { to: "/compare", label: "Compare" },
];

interface LayoutProps {
  configPath?: string | null;
  onSwitchConfig?: () => void;
  switchDisabled?: boolean;
}

export function Layout({
  configPath = null,
  onSwitchConfig,
  switchDisabled = false,
}: LayoutProps) {
  return (
    <div className={styles.shell}>
      <aside className={styles.sidebar}>
        <div className={styles.brand}>
          <span className={styles.brandMark}>FCV</span>
          <span className={styles.brandText}>FilteringCV</span>
        </div>
        <nav className={styles.nav}>
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                isActive ? `${styles.link} ${styles.active}` : styles.link
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <footer className={styles.footer}>
          {configPath && (
            <div className={styles.configPath} title={configPath}>
              {configPath}
            </div>
          )}
          {onSwitchConfig && (
            <button
              type="button"
              className={`btn btn-sm ${styles.switchBtn}`}
              disabled={switchDisabled}
              onClick={onSwitchConfig}
            >
              Switch config
            </button>
          )}
          <div>Dataset builder ops</div>
        </footer>
      </aside>
      <main className={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}
