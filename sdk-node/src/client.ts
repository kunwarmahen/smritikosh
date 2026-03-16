/**
 * Smritikosh SDK — TypeScript client.
 *
 * Uses native fetch (Node ≥ 18 / modern browsers).  No runtime dependencies.
 */

import type {
  SmritikoshClientOptions,
  // encode
  EncodeOptions,
  EncodedEvent,
  // buildContext
  BuildContextOptions,
  MemoryContext,
  LLMMessage,
  // getRecent
  GetRecentOptions,
  RecentEvent,
  // submitFeedback
  SubmitFeedbackOptions,
  FeedbackRecord,
  // getIdentity
  GetIdentityOptions,
  IdentityProfile,
  // deleteEvent / deleteUserMemory
  DeleteEventOptions,
  DeleteEventResult,
  DeleteUserMemoryOptions,
  DeleteUserMemoryResult,
  // procedures
  StoreProcedureOptions,
  ProcedureCreated,
  ListProceduresOptions,
  ProcedureRecord,
  DeleteProcedureOptions,
  DeleteProcedureResult,
  DeleteUserProceduresOptions,
  DeleteUserProceduresResult,
  // reconsolidate
  ReconsolidateOptions,
  ReconsolidationResult,
  // search
  SearchOptions,
  SearchResult,
  SearchResultItem,
  // ingest
  IngestPushOptions,
  IngestFileOptions,
  IngestEmailOptions,
  IngestCalendarOptions,
  IngestResult,
  // admin
  AdminJobOptions,
  AdminJobResponse,
  // health
  HealthStatus,
} from "./types.js";

// ── helpers ───────────────────────────────────────────────────────────────────

function toIso(d: string | Date): string {
  return d instanceof Date ? d.toISOString() : d;
}

function parseIngestResult(raw: Record<string, unknown>): IngestResult {
  return {
    source: raw["source"] as string,
    eventsIngested: raw["events_ingested"] as number,
    eventsFailed: raw["events_failed"] as number,
    eventIds: raw["event_ids"] as string[],
  };
}

class SmritikoshError extends Error {
  constructor(
    public readonly status: number,
    public readonly body: string,
  ) {
    super(`Smritikosh API error ${status}: ${body}`);
    this.name = "SmritikoshError";
  }
}

// ── client ────────────────────────────────────────────────────────────────────

export class SmritikoshClient {
  private readonly baseUrl: string;
  private readonly appId: string | undefined;
  private readonly timeoutMs: number;
  private readonly headers: Record<string, string>;

  constructor(options: SmritikoshClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/$/, "");
    this.appId = options.appId;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.headers = {
      "Content-Type": "application/json",
      ...options.headers,
    };
  }

  // ── low-level HTTP ─────────────────────────────────────────────────────────

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    const init: RequestInit = {
      method,
      headers: this.headers,
      signal: controller.signal,
    };
    if (body !== undefined) init.body = JSON.stringify(body);

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, init);
    } finally {
      clearTimeout(timer);
    }

    const text = await response.text();
    if (!response.ok) {
      throw new SmritikoshError(response.status, text);
    }
    return JSON.parse(text) as T;
  }

  private get<T>(path: string): Promise<T> {
    return this.request<T>("GET", path);
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("POST", path, body);
  }

  private patch<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>("PATCH", path, body);
  }

  private delete<T>(path: string): Promise<T> {
    return this.request<T>("DELETE", path);
  }

  private async postForm<T>(path: string, form: FormData): Promise<T> {
    // Omit Content-Type so the browser/Node sets it with the multipart boundary.
    const headers: Record<string, string> = { ...this.headers };
    delete headers["Content-Type"];

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers,
        body: form,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    const text = await response.text();
    if (!response.ok) throw new SmritikoshError(response.status, text);
    return JSON.parse(text) as T;
  }

  // ── encode ─────────────────────────────────────────────────────────────────

  async encode(options: EncodeOptions): Promise<EncodedEvent> {
    const raw = await this.post<Record<string, unknown>>("/memory/encode", {
      user_id: options.userId,
      content: options.content,
      app_id: options.appId ?? this.appId,
      metadata: options.metadata,
    });
    return {
      eventId: raw["event_id"] as string,
      userId: raw["user_id"] as string,
      importanceScore: raw["importance_score"] as number,
      factsExtracted: raw["facts_extracted"] as number,
      extractionFailed: raw["extraction_failed"] as boolean,
    };
  }

  // ── buildContext ───────────────────────────────────────────────────────────

  async buildContext(options: BuildContextOptions): Promise<MemoryContext> {
    const body: Record<string, unknown> = {
      user_id: options.userId,
      query: options.query,
      app_id: options.appId ?? this.appId,
    };
    if (options.fromDate !== undefined) body["from_date"] = toIso(options.fromDate);
    if (options.toDate !== undefined) body["to_date"] = toIso(options.toDate);

    const raw = await this.post<Record<string, unknown>>("/context/build", body);
    const contextText = raw["context_text"] as string;
    const messages = raw["messages"] as LLMMessage[];
    const totalMemories = raw["total_memories"] as number;

    return {
      userId: raw["user_id"] as string,
      query: raw["query"] as string,
      contextText,
      messages,
      totalMemories,
      embeddingFailed: raw["embedding_failed"] as boolean,
      intent: raw["intent"] as string,
      reconsolidationScheduled: (raw["reconsolidation_scheduled"] as boolean | undefined) ?? false,
      isEmpty() {
        return totalMemories === 0;
      },
    };
  }

  // ── getRecent ──────────────────────────────────────────────────────────────

  async getRecent(options: GetRecentOptions): Promise<RecentEvent[]> {
    const params = new URLSearchParams({ user_id: options.userId });
    if (options.appId ?? this.appId) params.set("app_id", (options.appId ?? this.appId)!);
    if (options.limit !== undefined) params.set("limit", String(options.limit));

    const raw = await this.get<{ events: Array<Record<string, unknown>> }>(
      `/memory/recent?${params}`,
    );
    return raw.events.map((e) => ({
      eventId: e["event_id"] as string,
      rawText: e["raw_text"] as string,
      importanceScore: e["importance_score"] as number,
      consolidated: e["consolidated"] as boolean,
      createdAt: e["created_at"] as string,
    }));
  }

  // ── submitFeedback ─────────────────────────────────────────────────────────

  async submitFeedback(options: SubmitFeedbackOptions): Promise<FeedbackRecord> {
    const raw = await this.post<Record<string, unknown>>("/memory/feedback", {
      event_id: options.eventId,
      user_id: options.userId,
      feedback_type: options.feedbackType,
      app_id: options.appId ?? this.appId,
      comment: options.comment,
    });
    return {
      feedbackId: raw["feedback_id"] as string,
      eventId: raw["event_id"] as string,
      newImportanceScore: raw["new_importance_score"] as number,
    };
  }

  // ── getIdentity ────────────────────────────────────────────────────────────

  async getIdentity(options: GetIdentityOptions): Promise<IdentityProfile> {
    const params = new URLSearchParams({ user_id: options.userId });
    if (options.appId ?? this.appId) params.set("app_id", (options.appId ?? this.appId)!);

    const raw = await this.get<Record<string, unknown>>(`/identity?${params}`);
    return {
      userId: raw["user_id"] as string,
      appId: raw["app_id"] as string,
      summary: raw["summary"] as string,
      dimensions: raw["dimensions"] as IdentityProfile["dimensions"],
      beliefs: raw["beliefs"] as IdentityProfile["beliefs"],
      totalFacts: raw["total_facts"] as number,
      computedAt: raw["computed_at"] as string,
      isEmpty: raw["is_empty"] as boolean,
    };
  }

  // ── deleteEvent ────────────────────────────────────────────────────────────

  async deleteEvent(options: DeleteEventOptions): Promise<DeleteEventResult> {
    const raw = await this.delete<Record<string, unknown>>(
      `/memory/event/${encodeURIComponent(options.eventId)}`,
    );
    return {
      deleted: raw["deleted"] as boolean,
      eventId: raw["event_id"] as string,
    };
  }

  // ── deleteUserMemory ───────────────────────────────────────────────────────

  async deleteUserMemory(options: DeleteUserMemoryOptions): Promise<DeleteUserMemoryResult> {
    const params = new URLSearchParams({ user_id: options.userId });
    if (options.appId ?? this.appId) params.set("app_id", (options.appId ?? this.appId)!);

    const raw = await this.delete<Record<string, unknown>>(
      `/memory/user/${encodeURIComponent(options.userId)}?${params}`,
    );
    return {
      eventsDeleted: raw["events_deleted"] as number,
      userId: raw["user_id"] as string,
      appId: raw["app_id"] as string,
    };
  }

  // ── storeProcedure ─────────────────────────────────────────────────────────

  async storeProcedure(options: StoreProcedureOptions): Promise<ProcedureCreated> {
    const raw = await this.post<Record<string, unknown>>("/procedures", {
      user_id: options.userId,
      trigger: options.trigger,
      instruction: options.instruction,
      app_id: options.appId ?? this.appId,
      category: options.category,
      priority: options.priority,
      confidence: options.confidence,
      source: options.source,
    });
    return {
      procedureId: raw["procedure_id"] as string,
      userId: raw["user_id"] as string,
      trigger: raw["trigger"] as string,
      instruction: raw["instruction"] as string,
      category: raw["category"] as string,
      priority: raw["priority"] as number,
      isActive: raw["is_active"] as boolean,
      hitCount: raw["hit_count"] as number,
      confidence: raw["confidence"] as number,
      source: raw["source"] as string,
      createdAt: raw["created_at"] as string,
    };
  }

  // ── listProcedures ─────────────────────────────────────────────────────────

  async listProcedures(options: ListProceduresOptions): Promise<ProcedureRecord[]> {
    const params = new URLSearchParams();
    if (options.appId ?? this.appId) params.set("app_id", (options.appId ?? this.appId)!);
    if (options.activeOnly !== undefined) params.set("active_only", String(options.activeOnly));
    if (options.category !== undefined) params.set("category", options.category);

    const raw = await this.get<{ procedures: Array<Record<string, unknown>> }>(
      `/procedures/${encodeURIComponent(options.userId)}?${params}`,
    );
    return raw.procedures.map((p) => ({
      procedureId: p["procedure_id"] as string,
      trigger: p["trigger"] as string,
      instruction: p["instruction"] as string,
      category: p["category"] as string,
      priority: p["priority"] as number,
      isActive: p["is_active"] as boolean,
      hitCount: p["hit_count"] as number,
    }));
  }

  // ── deleteProcedure ────────────────────────────────────────────────────────

  async deleteProcedure(options: DeleteProcedureOptions): Promise<DeleteProcedureResult> {
    const raw = await this.delete<Record<string, unknown>>(
      `/procedures/${encodeURIComponent(options.procedureId)}`,
    );
    return {
      deleted: raw["deleted"] as boolean,
      procedureId: raw["procedure_id"] as string,
    };
  }

  // ── deleteUserProcedures ───────────────────────────────────────────────────

  async deleteUserProcedures(
    options: DeleteUserProceduresOptions,
  ): Promise<DeleteUserProceduresResult> {
    const params = new URLSearchParams();
    if (options.appId ?? this.appId) params.set("app_id", (options.appId ?? this.appId)!);

    const raw = await this.delete<Record<string, unknown>>(
      `/procedures/user/${encodeURIComponent(options.userId)}?${params}`,
    );
    return {
      proceduresDeleted: raw["procedures_deleted"] as number,
      userId: raw["user_id"] as string,
      appId: raw["app_id"] as string,
    };
  }

  // ── reconsolidate ──────────────────────────────────────────────────────────

  async reconsolidate(options: ReconsolidateOptions): Promise<ReconsolidationResult> {
    const raw = await this.post<Record<string, unknown>>("/admin/reconsolidate", {
      event_id: options.eventId,
      query: options.query,
      user_id: options.userId,
    });
    return {
      eventId: raw["event_id"] as string,
      userId: raw["user_id"] as string,
      updated: raw["updated"] as boolean,
      skipped: raw["skipped"] as boolean,
      skipReason: (raw["skip_reason"] as string | undefined) ?? "",
      oldSummary: (raw["old_summary"] as string | undefined) ?? "",
      newSummary: (raw["new_summary"] as string | undefined) ?? "",
    };
  }

  // ── admin jobs ─────────────────────────────────────────────────────────────

  private async adminJob(
    endpoint: string,
    options: AdminJobOptions = {},
  ): Promise<AdminJobResponse> {
    const raw = await this.post<Record<string, unknown>>(endpoint, {
      user_id: options.userId,
      app_id: options.appId ?? this.appId,
    });
    return {
      job: raw["job"] as string,
      usersProcessed: raw["users_processed"] as number,
      results: (raw["results"] as Array<Record<string, unknown>>).map((r) => ({
        userId: r["user_id"] as string,
        appId: r["app_id"] as string,
        skipped: r["skipped"] as boolean,
        detail: r["detail"] as string,
      })),
    };
  }

  async adminConsolidate(options: AdminJobOptions = {}): Promise<AdminJobResponse> {
    return this.adminJob("/admin/consolidate", options);
  }

  async adminPrune(options: AdminJobOptions = {}): Promise<AdminJobResponse> {
    return this.adminJob("/admin/prune", options);
  }

  async adminCluster(options: AdminJobOptions = {}): Promise<AdminJobResponse> {
    return this.adminJob("/admin/cluster", options);
  }

  async adminMineBeliefs(options: AdminJobOptions = {}): Promise<AdminJobResponse> {
    return this.adminJob("/admin/mine-beliefs", options);
  }

  // ── search ─────────────────────────────────────────────────────────────────

  async search(options: SearchOptions): Promise<SearchResult> {
    const body: Record<string, unknown> = {
      user_id: options.userId,
      query: options.query,
      app_id: options.appId ?? this.appId,
      limit: options.limit,
    };
    if (options.fromDate !== undefined) body["from_date"] = toIso(options.fromDate);
    if (options.toDate !== undefined) body["to_date"] = toIso(options.toDate);

    const raw = await this.post<Record<string, unknown>>("/memory/search", body);
    return {
      userId: raw["user_id"] as string,
      query: raw["query"] as string,
      results: (raw["results"] as Array<Record<string, unknown>>).map((r) => ({
        eventId: r["event_id"] as string,
        rawText: r["raw_text"] as string,
        importanceScore: r["importance_score"] as number,
        hybridScore: r["hybrid_score"] as number,
        similarityScore: r["similarity_score"] as number,
        recencyScore: r["recency_score"] as number,
        consolidated: r["consolidated"] as boolean,
        createdAt: r["created_at"] as string,
      })),
      total: raw["total"] as number,
      embeddingFailed: raw["embedding_failed"] as boolean,
    };
  }

  // ── ingest ─────────────────────────────────────────────────────────────────

  async ingestPush(options: IngestPushOptions): Promise<IngestResult> {
    const raw = await this.post<Record<string, unknown>>("/ingest/push", {
      user_id: options.userId,
      content: options.content,
      source: options.source ?? "api",
      source_id: options.sourceId ?? "",
      app_id: options.appId ?? this.appId,
      metadata: options.metadata ?? {},
    });
    return parseIngestResult(raw);
  }

  async ingestFile(options: IngestFileOptions): Promise<IngestResult> {
    const form = new FormData();
    form.append("user_id", options.userId);
    form.append("app_id", options.appId ?? this.appId ?? "default");
    form.append(
      "file",
      new Blob([options.fileContent]),
      options.filename,
    );
    const raw = await this.postForm<Record<string, unknown>>("/ingest/file", form);
    return parseIngestResult(raw);
  }

  async ingestEmail(options: IngestEmailOptions): Promise<IngestResult> {
    const raw = await this.post<Record<string, unknown>>("/ingest/email/sync", {
      user_id: options.userId,
      host: options.host,
      port: options.port ?? 993,
      username: options.username,
      password: options.password,
      mailbox: options.mailbox ?? "INBOX",
      limit: options.limit ?? 20,
      unseen_only: options.unseenOnly ?? true,
      app_id: options.appId ?? this.appId,
    });
    return parseIngestResult(raw);
  }

  async ingestCalendar(options: IngestCalendarOptions): Promise<IngestResult> {
    const filename = options.filename ?? "calendar.ics";
    const form = new FormData();
    form.append("user_id", options.userId);
    form.append("app_id", options.appId ?? this.appId ?? "default");
    form.append("file", new Blob([options.fileContent]), filename);
    const raw = await this.postForm<Record<string, unknown>>("/ingest/calendar", form);
    return parseIngestResult(raw);
  }

  // ── health ─────────────────────────────────────────────────────────────────

  async health(): Promise<HealthStatus> {
    const raw = await this.get<Record<string, unknown>>("/health");
    return {
      status: raw["status"] as string,
      version: raw["version"] as string,
      postgres: (raw["postgres"] as string | undefined) ?? "unknown",
      neo4j: (raw["neo4j"] as string | undefined) ?? "unknown",
    };
  }
}
