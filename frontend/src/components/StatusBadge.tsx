import type { JobStatus } from "../api/types";

interface StatusBadgeProps {
  status: JobStatus | string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const normalized = status.toLowerCase().replace(/_/g, "-");
  return <span className={`badge badge-${normalized}`}>{status}</span>;
}
