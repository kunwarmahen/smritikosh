/**
 * Typed API client for the Smritikosh FastAPI backend.
 *
 * Usage (server component):
 *   import { createApiClient } from "@/lib/api-client"
 *   import { auth } from "../../../auth"
 *   const session = await auth()
 *   const api = createApiClient(session?.accessToken)
 *   const events = await api.getRecentEvents("alice")
 *
 * Usage (client component via hooks):
 *   Use the hooks in @/hooks/* which manage the token from useSession()
 */

const API_URL =
  typeof window === "undefined"
    ? (process.env.SMRITIKOSH_API_URL ?? "http://localhost:8080")
    : "/api/backend";

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit & { token?: string } = {},
): Promise<T> {
  const { token, ...fetchOptions } = options;
  const headers: Record<string, string> = {};

  // Skip Content-Type for FormData (browser will set it with boundary)
  if (!(fetchOptions.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  Object.assign(headers, fetchOptions.headers as Record<string, string>);

  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { ...fetchOptions, headers });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {}
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export function createApiClient(token?: string) {
  const opts = (extra?: RequestInit) => ({ ...extra, token });

  return {
    // ── Health ───────────────────────────────────────────────────────────
    health: () => request("/health", opts()),

    // ── Auth ─────────────────────────────────────────────────────────────
    getMe: () => request("/auth/me", opts()),
    register: (body: {
      username: string;
      password: string;
      role: string;
      app_ids?: string[];
      email?: string;
    }) => request("/auth/register", opts({ method: "POST", body: JSON.stringify(body) })),

    // ── Memory ───────────────────────────────────────────────────────────
    getRecentEvents: (userId: string, params?: { app_ids?: string[]; limit?: number }) => {
      const q = new URLSearchParams();
      if (params?.app_ids) params.app_ids.forEach((id) => q.append("app_ids", id));
      if (params?.limit) q.set("limit", String(params.limit));
      return request(`/memory/${userId}?${q}`, opts());
    },

    searchMemory: (body: {
      user_id: string;
      query: string;
      app_ids?: string[];
      limit?: number;
      from_date?: string;
      to_date?: string;
    }) => request("/memory/search", opts({ method: "POST", body: JSON.stringify(body) })),

    deleteEvent: (eventId: string) =>
      request(`/memory/event/${eventId}`, opts({ method: "DELETE" })),

    deleteUserMemory: (userId: string, appId = "default") =>
      request(`/memory/user/${userId}?app_id=${appId}`, opts({ method: "DELETE" })),

    // ── Feedback ─────────────────────────────────────────────────────────
    submitFeedback: (body: {
      event_id: string;
      user_id: string;
      feedback_type: "positive" | "negative" | "neutral";
      app_id?: string;
      comment?: string;
    }) => request("/feedback", opts({ method: "POST", body: JSON.stringify(body) })),

    // ── Context ───────────────────────────────────────────────────────────
    buildContext: (body: { user_id: string; query: string; app_ids?: string[] }) =>
      request("/context", opts({ method: "POST", body: JSON.stringify(body) })),

    // ── Identity ─────────────────────────────────────────────────────────
    getIdentity: (userId: string, appId = "default") =>
      request(`/identity/${userId}?app_id=${appId}`, opts()),

    // ── Audit ─────────────────────────────────────────────────────────────
    getAuditTimeline: (
      userId: string,
      params?: {
        app_id?: string;
        event_type?: string;
        limit?: number;
        offset?: number;
        from_ts?: string;
        to_ts?: string;
      },
    ) => {
      const q = new URLSearchParams({ app_id: params?.app_id ?? "default" });
      if (params?.event_type) q.set("event_type", params.event_type);
      if (params?.limit) q.set("limit", String(params.limit));
      if (params?.offset) q.set("offset", String(params.offset));
      if (params?.from_ts) q.set("from_ts", params.from_ts);
      if (params?.to_ts) q.set("to_ts", params.to_ts);
      return request(`/audit/${userId}?${q}`, opts());
    },

    getEventLineage: (eventId: string) =>
      request(`/audit/event/${eventId}/lineage`, opts()),

    getAuditStats: (userId: string, appId = "default") =>
      request(`/audit/stats/${userId}?app_id=${appId}`, opts()),

    // ── Procedures ────────────────────────────────────────────────────────
    getProcedures: (userId: string, appIds?: string[], activeOnly = false) => {
      const q = new URLSearchParams();
      if (appIds) appIds.forEach((id) => q.append("app_ids", id));
      if (activeOnly) q.set("active_only", "true");
      return request(`/procedures/${userId}?${q}`, opts());
    },

    createProcedure: (body: {
      user_id: string;
      trigger: string;
      instruction: string;
      app_id?: string;
      priority?: number;
      category?: string;
    }) => request("/procedures", opts({ method: "POST", body: JSON.stringify(body) })),

    updateProcedure: (procedureId: string, body: Partial<{ priority: number; is_active: boolean; instruction: string }>) =>
      request(`/procedures/${procedureId}`, opts({ method: "PATCH", body: JSON.stringify(body) })),

    deleteProcedure: (procedureId: string) =>
      request(`/procedures/${procedureId}`, opts({ method: "DELETE" })),

    // ── Manual fact entry ─────────────────────────────────────────────────
    storeFact: (body: {
      user_id: string;
      app_id?: string;
      category: string;
      key: string;
      value: string;
      note?: string;
      source_type?: string;
      confidence?: number;
    }) => request("/memory/fact", opts({ method: "POST", body: JSON.stringify(body) })),

    // ── Media ingestion ───────────────────────────────────────────────────
    uploadMedia: (formData: FormData) =>
      request("/ingest/media", opts({ method: "POST", body: formData })),

    getMediaStatus: (mediaId: string) =>
      request(`/ingest/media/${mediaId}/status`, opts()),

    confirmMediaFacts: (mediaId: string, body: {
      user_id: string;
      app_id?: string;
      confirmed_indices: number[];
    }) =>
      request(`/ingest/media/${mediaId}/confirm`, opts({ method: "POST", body: JSON.stringify(body) })),

    // ── Voice enrollment ──────────────────────────────────────────────────
    getVoiceEnrollmentStatus: (userId: string, appId = "default") =>
      request(`/user/${userId}/voice-enrollment?app_id=${appId}`, opts()),

    enrollVoice: (userId: string, formData: FormData) =>
      request(`/user/${userId}/voice-enrollment`, opts({ method: "POST", body: formData })),

    deleteVoiceEnrollment: (userId: string, appId = "default") =>
      request(`/user/${userId}/voice-enrollment?app_id=${appId}`, opts({ method: "DELETE" })),

    // ── Memory event detail & links ───────────────────────────────────────
    getEvent: (eventId: string) =>
      request(`/memory/event/${eventId}`, opts()),

    getEventLinks: (eventId: string) =>
      request(`/memory/event/${eventId}/links`, opts()),

    // ── Graph ─────────────────────────────────────────────────────────────
    getFactGraph: (userId: string, appId = "default", minConfidence = 0) => {
      const q = new URLSearchParams({ app_id: appId });
      if (minConfidence > 0) q.set("min_confidence", String(minConfidence));
      return request(`/graph/facts/${userId}?${q}`, opts());
    },

    // ── Admin — user management ───────────────────────────────────────────
    adminListUsers: (params?: { limit?: number; offset?: number; role?: string }) => {
      const q = new URLSearchParams();
      if (params?.limit)  q.set("limit",  String(params.limit));
      if (params?.offset) q.set("offset", String(params.offset));
      if (params?.role)   q.set("role",   params.role);
      return request(`/admin/users?${q}`, opts());
    },

    adminGetUser: (username: string) =>
      request(`/admin/users/${username}`, opts()),

    adminPatchUser: (username: string, body: { is_active?: boolean; role?: string; app_ids?: string[] }) =>
      request(`/admin/users/${username}`, opts({ method: "PATCH", body: JSON.stringify(body) })),

    // ── Admin — jobs ──────────────────────────────────────────────────────
    adminConsolidate: (userId: string, appId = "default") =>
      request(`/admin/consolidate`, opts({ method: "POST", body: JSON.stringify({ user_id: userId, app_id: appId }) })),

    adminPrune: (userId: string, appId = "default") =>
      request(`/admin/prune`, opts({ method: "POST", body: JSON.stringify({ user_id: userId, app_id: appId }) })),

    adminCluster: (userId: string, appId = "default") =>
      request(`/admin/cluster`, opts({ method: "POST", body: JSON.stringify({ user_id: userId, app_id: appId }) })),

    adminMineBeliefs: (userId: string, appId = "default") =>
      request(`/admin/mine-beliefs`, opts({ method: "POST", body: JSON.stringify({ user_id: userId, app_id: appId }) })),

    // ── Admin — embedding health ──────────────────────────────────────────
    adminEmbeddingHealth: () =>
      request(`/admin/embedding-health`, opts()),

    adminReEmbed: () =>
      request(`/admin/re-embed`, opts({ method: "POST" })),

    // ── API keys ──────────────────────────────────────────────────────────
    listApiKeys: () =>
      request(`/keys`, opts()),

    createApiKey: (body: { name: string; app_ids?: string[] }) =>
      request(`/keys`, opts({ method: "POST", body: JSON.stringify(body) })),

    revokeApiKey: (keyId: string) =>
      request(`/keys/${keyId}`, opts({ method: "DELETE" })),
  };
}

export { ApiError };
export type ApiClient = ReturnType<typeof createApiClient>;
