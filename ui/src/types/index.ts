// ── API response types (mirrors FastAPI schemas) ──────────────────────────────

export interface MemoryEvent {
  event_id: string;
  user_id: string;
  app_id?: string;
  raw_text: string;
  summary?: string | null;
  importance_score: number;
  recall_count?: number;
  reconsolidation_count?: number;
  consolidated: boolean;
  cluster_id?: number | null;
  cluster_label?: string | null;
  source_type?: string;
  created_at: string;
  updated_at?: string;
  last_reconsolidated_at?: string | null;
  // Present when this event came from a search result
  hybrid_score?: number;
  similarity_score?: number;
}

export interface FactRequest {
  category: string;
  key: string;
  value: string;
  note?: string;
  source_type?: string;
  confidence?: number;
}

export interface FactResponse {
  user_id: string;
  app_id: string;
  category: string;
  key: string;
  value: string;
  confidence: number;
  frequency_count: number;
  source_type: string;
  status: string;
  first_seen_at: string;
  last_seen_at: string;
}

export interface RecentEventsResponse {
  user_id: string;
  app_ids: string[];
  events: MemoryEvent[];
}

export interface SearchResultItem {
  event_id: string;
  raw_text: string;
  summary?: string | null;
  importance_score: number;
  hybrid_score: number;
  similarity_score: number;
  recency_score: number;
  consolidated: boolean;
  created_at: string;
}

export interface SearchResponse {
  user_id: string;
  query: string;
  results: SearchResultItem[];
  total: number;
  embedding_failed: boolean;
}

export interface IdentityDimension {
  category: string;
  dominant_value: string;
  confidence: number;
  fact_count: number;
}

export interface UserBelief {
  statement: string;
  category: string;
  confidence: number;
  evidence_count: number;
}

export interface IdentityProfile {
  user_id: string;
  app_id: string;
  summary: string;
  dimensions: IdentityDimension[];
  beliefs: UserBelief[];
  total_facts: number;
  computed_at: string;
  is_empty: boolean;
}

export interface AuditEvent {
  id: string;
  event_type: string;
  user_id: string;
  app_id: string;
  event_id?: string | null;
  session_id?: string | null;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface AuditStats {
  [event_type: string]: number;
}

export interface Procedure {
  procedure_id: string;
  user_id: string;
  app_id: string;
  trigger: string;
  instruction: string;
  category: string;
  priority: number;
  is_active: boolean;
  hit_count: number;
  created_at?: string;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "error";
  version: string;
  postgres: "ok" | "error";
  neo4j: "ok" | "error";
  mongodb: "ok" | "error" | "not_configured";
  llm_model: string;
  llm_status: "ok" | "error";
}

export interface AdminUser {
  username: string;
  email?: string | null;
  role: string;
  app_ids: string[];
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface AdminUsersResponse {
  users: AdminUser[];
  total: number;
  limit: number;
  offset: number;
}

export interface FeedbackResponse {
  feedback_id: string;
  event_id: string;
  new_importance_score: number;
}

export interface FactGraphNode {
  id: string;
  label: string;
  node_type: "user" | "fact";
  category?: string | null;
  key?: string | null;
  confidence?: number | null;
  frequency_count?: number | null;
  source_event_ids?: string[];
}

export interface FactGraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
}

export interface FactGraph {
  user_id: string;
  app_id: string;
  nodes: FactGraphNode[];
  edges: FactGraphEdge[];
}

export interface MemoryLink {
  link_id: string;
  from_event_id: string;
  from_event_preview: string;
  to_event_id: string;
  to_event_preview: string;
  relation_type: "caused" | "preceded" | "related" | "contradicts";
  created_at: string;
}

export interface MemoryLinksResponse {
  event_id: string;
  links: MemoryLink[];
}

// ── Media ingestion types ──────────────────────────────────────────────────────

export interface PendingFact {
  content: string;
  category: string;
  key: string;
  value: string;
  relevance_score: number;
  confidence?: number;
}

export interface MediaUploadResponse {
  media_id: string;
  user_id: string;
  app_id: string;
  content_type: string;
  status: 'processing' | 'complete' | 'nothing_found' | 'failed';
  facts_extracted: number;
  facts_pending_review: number;
  message: string;
}

export interface MediaStatusResponse extends MediaUploadResponse {
  pending_facts: PendingFact[];
}

// ── UI-only types ─────────────────────────────────────────────────────────────

export type ImportanceLevel = "high" | "medium" | "low";

export function importanceLevel(score: number): ImportanceLevel {
  if (score >= 0.7) return "high";
  if (score >= 0.4) return "medium";
  return "low";
}

export const EVENT_TYPE_LABELS: Record<string, string> = {
  "memory.encoded":           "Encoded",
  "memory.facts_extracted":   "Facts extracted",
  "memory.consolidated":      "Consolidated",
  "memory.reconsolidated":    "Reconsolidated",
  "memory.pruned":            "Pruned",
  "memory.prune_run":         "Pruning run",
  "memory.reconsolidate_run": "Reconsolidation run",
  "memory.clustered":         "Clustered",
  "belief.mined":             "Belief mined",
  "feedback.submitted":       "Feedback",
  "context.built":            "Context built",
  "search.performed":         "Search",
};

export const EVENT_TYPE_COLORS: Record<string, string> = {
  "memory.encoded":           "text-violet-400",
  "memory.facts_extracted":   "text-blue-400",
  "memory.consolidated":      "text-emerald-400",
  "memory.reconsolidated":    "text-cyan-400",
  "memory.pruned":            "text-rose-400",
  "memory.prune_run":         "text-rose-300",
  "memory.reconsolidate_run": "text-cyan-300",
  "memory.clustered":         "text-amber-400",
  "belief.mined":             "text-purple-400",
  "feedback.submitted":       "text-green-400",
  "context.built":            "text-sky-400",
  "search.performed":         "text-indigo-400",
};
