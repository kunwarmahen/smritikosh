/**
 * Smritikosh SDK — TypeScript types.
 *
 * All interfaces mirror the server-side Pydantic response schemas so callers
 * get full type safety without depending on the server package directly.
 */

// ── encode ────────────────────────────────────────────────────────────────────

export interface EncodeOptions {
  userId: string;
  content: string;
  appId?: string;
  metadata?: Record<string, unknown>;
}

export interface EncodedEvent {
  eventId: string;
  userId: string;
  importanceScore: number;
  factsExtracted: number;
  extractionFailed: boolean;
}

// ── buildContext ──────────────────────────────────────────────────────────────

export interface BuildContextOptions {
  userId: string;
  query: string;
  appId?: string;
  /** ISO 8601 datetime — only include events on or after this date. */
  fromDate?: string | Date;
  /** ISO 8601 datetime — only include events on or before this date. */
  toDate?: string | Date;
}

export interface LLMMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface MemoryContext {
  userId: string;
  query: string;
  /** Structured markdown ready to prepend to any LLM system prompt. */
  contextText: string;
  /** OpenAI-style message list. Append your user turn and call the LLM. */
  messages: LLMMessage[];
  totalMemories: number;
  embeddingFailed: boolean;
  intent: string;
  reconsolidationScheduled: boolean;
  isEmpty(): boolean;
}

// ── getRecent ─────────────────────────────────────────────────────────────────

export interface GetRecentOptions {
  userId: string;
  appId?: string;
  limit?: number;
}

export interface RecentEvent {
  eventId: string;
  rawText: string;
  importanceScore: number;
  consolidated: boolean;
  createdAt: string;
}

// ── submitFeedback ────────────────────────────────────────────────────────────

export type FeedbackType = "positive" | "negative" | "neutral";

export interface SubmitFeedbackOptions {
  eventId: string;
  userId: string;
  feedbackType: FeedbackType;
  appId?: string;
  comment?: string;
}

export interface FeedbackRecord {
  feedbackId: string;
  eventId: string;
  newImportanceScore: number;
}

// ── getIdentity ───────────────────────────────────────────────────────────────

export interface GetIdentityOptions {
  userId: string;
  appId?: string;
}

export interface BeliefItem {
  statement: string;
  category: string;
  confidence: number;
  evidenceCount: number;
}

export interface IdentityDimension {
  category: string;
  dominantValue: string;
  confidence: number;
  factCount: number;
}

export interface IdentityProfile {
  userId: string;
  appId: string;
  /** LLM-generated narrative description of the user. */
  summary: string;
  dimensions: IdentityDimension[];
  beliefs: BeliefItem[];
  totalFacts: number;
  computedAt: string;
  isEmpty: boolean;
}

// ── deleteEvent ───────────────────────────────────────────────────────────────

export interface DeleteEventOptions {
  eventId: string;
}

export interface DeleteEventResult {
  deleted: boolean;
  eventId: string;
}

// ── deleteUserMemory ──────────────────────────────────────────────────────────

export interface DeleteUserMemoryOptions {
  userId: string;
  appId?: string;
}

export interface DeleteUserMemoryResult {
  eventsDeleted: number;
  userId: string;
  appId: string;
}

// ── procedures ────────────────────────────────────────────────────────────────

export type ProcedureCategory =
  | "topic_response"
  | "communication"
  | "preference"
  | "domain_workflow";

export interface StoreProcedureOptions {
  userId: string;
  /** Topic/keyword phrase that activates this rule, e.g. "LLM deployment". */
  trigger: string;
  /** Behavioral instruction to apply when triggered. */
  instruction: string;
  appId?: string;
  category?: ProcedureCategory;
  /** Priority 1 (low) – 10 (high). Default 5. */
  priority?: number;
  confidence?: number;
  source?: string;
}

export interface ProcedureCreated {
  procedureId: string;
  userId: string;
  trigger: string;
  instruction: string;
  category: string;
  priority: number;
  isActive: boolean;
  hitCount: number;
  confidence: number;
  source: string;
  createdAt: string;
}

export interface ListProceduresOptions {
  userId: string;
  appId?: string;
  activeOnly?: boolean;
  category?: string;
}

export interface ProcedureRecord {
  procedureId: string;
  trigger: string;
  instruction: string;
  category: string;
  priority: number;
  isActive: boolean;
  hitCount: number;
}

export interface DeleteProcedureOptions {
  procedureId: string;
}

export interface DeleteProcedureResult {
  deleted: boolean;
  procedureId: string;
}

export interface DeleteUserProceduresOptions {
  userId: string;
  appId?: string;
}

export interface DeleteUserProceduresResult {
  proceduresDeleted: number;
  userId: string;
  appId: string;
}

// ── reconsolidate ─────────────────────────────────────────────────────────────

export interface ReconsolidateOptions {
  eventId: string;
  /** The query that surfaced this memory — used as the recall context. */
  query: string;
  userId: string;
}

export interface ReconsolidationResult {
  eventId: string;
  userId: string;
  updated: boolean;
  skipped: boolean;
  skipReason: string;
  oldSummary: string;
  newSummary: string;
}

// ── admin jobs ────────────────────────────────────────────────────────────────

export interface AdminJobOptions {
  /** Target a specific user. If omitted, runs for all eligible users. */
  userId?: string;
  appId?: string;
}

export interface AdminJobResult {
  userId: string;
  appId: string;
  skipped: boolean;
  detail: string;
}

export interface AdminJobResponse {
  job: string;
  usersProcessed: number;
  results: AdminJobResult[];
}

// ── health ────────────────────────────────────────────────────────────────────

export interface HealthStatus {
  status: string;
  version: string;
}

// ── client config ─────────────────────────────────────────────────────────────

export interface SmritikoshClientOptions {
  /** Base URL of the Smritikosh server, e.g. "http://localhost:8080". */
  baseUrl: string;
  /**
   * Default application namespace.  All methods use this unless overridden
   * with a per-call `appId`.  Isolates memory across multiple applications
   * sharing one server.
   */
  appId?: string;
  /** Per-request timeout in milliseconds. Default: 30 000. */
  timeoutMs?: number;
  /** Extra headers sent with every request, e.g. `{ Authorization: "Bearer ..." }`. */
  headers?: Record<string, string>;
}
