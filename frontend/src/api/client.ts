import type {
  ClipOverride,
  ClipsPage,
  CompareRequest,
  CompareResult,
  CoverageReport,
  CreateJobRequest,
  DashboardSummary,
  JobRecord,
  JobSummary,
  OverrideListResponse,
  ProgressRecord,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) {
        detail = payload.detail;
      }
    } catch {
      // ignore parse errors
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  dashboard: () => request<DashboardSummary>("/api/dashboard"),

  listJobs: (limit = 50, offset = 0) =>
    request<JobSummary[]>(`/api/jobs?limit=${limit}&offset=${offset}`),

  getJob: (id: string) => request<JobRecord>(`/api/jobs/${id}`),

  createJob: (body: CreateJobRequest) =>
    request<JobRecord>("/api/jobs", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  cancelJob: (id: string) =>
    request<JobRecord>(`/api/jobs/${id}/cancel`, { method: "POST" }),

  listProgress: (id: string, afterId = 0) =>
    request<ProgressRecord[]>(
      `/api/jobs/${id}/progress?after_id=${afterId}&limit=500`,
    ),

  listClips: (params: {
    page?: number;
    page_size?: number;
    disposition?: string;
    speaker_id?: string;
    split?: string;
    search?: string;
  }) => {
    const query = new URLSearchParams();
    if (params.page) query.set("page", String(params.page));
    if (params.page_size) query.set("page_size", String(params.page_size));
    if (params.disposition) query.set("disposition", params.disposition);
    if (params.speaker_id) query.set("speaker_id", params.speaker_id);
    if (params.split) query.set("split", params.split);
    if (params.search) query.set("search", params.search);
    return request<ClipsPage>(`/api/catalog/clips?${query.toString()}`);
  },

  coverageReport: () => request<CoverageReport>("/api/reports/coverage"),

  listOverrides: () => request<OverrideListResponse>("/api/overrides"),

  upsertOverride: (body: ClipOverride) =>
    request<OverrideListResponse>("/api/overrides", {
      method: "PUT",
      body: JSON.stringify(body),
    }),

  deleteOverride: (clipId: string) =>
    request<OverrideListResponse>(`/api/overrides/${encodeURIComponent(clipId)}`, {
      method: "DELETE",
    }),

  compare: (body: CompareRequest) =>
    request<CompareResult>("/api/compare", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  audioUrl: (relativePath: string) =>
    `/api/audio/${relativePath
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/")}`,
};

export function jobWebSocketUrl(jobId: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = window.location.host;
  return `${protocol}//${host}/ws/jobs/${jobId}`;
}
