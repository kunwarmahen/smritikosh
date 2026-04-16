"""
Pydantic request/response schemas for the Smritikosh API.

Kept separate from DB models so the API contract can evolve independently
of the internal storage representation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── POST /auth/token  POST /auth/register  GET /auth/me ───────────────────────


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str
    app_ids: list[str]


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8)
    role: str = Field("user", description="'user' or 'admin'")
    app_ids: list[str] = Field(default_factory=lambda: ["default"])
    email: Optional[str] = None


class UserResponse(BaseModel):
    user_id: str
    username: str
    role: str
    app_ids: list[str]
    email: Optional[str]
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


# ── POST /memory/event ────────────────────────────────────────────────────────


class EventRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    content: str = Field(..., description="Raw interaction text to encode into memory")
    app_id: str = Field("default", description="Application namespace (for multi-app isolation)")
    metadata: dict = Field(default_factory=dict, description="Optional extra context (source, channel, etc.)")


class EventResponse(BaseModel):
    event_id: str
    user_id: str
    importance_score: float
    facts_extracted: int
    extraction_failed: bool

    model_config = {"from_attributes": True}


# ── POST /context ─────────────────────────────────────────────────────────────


class ContextRequest(BaseModel):
    user_id: str = Field(..., description="User to retrieve memory for")
    query: str = Field(..., description="Current user query or topic to retrieve context around")
    app_ids: list[str] | None = Field(None, description="App namespaces to search. Defaults to all apps in your token.")
    from_date: Optional[datetime] = Field(None, description="Only include events on or after this datetime (ISO 8601)")
    to_date: Optional[datetime] = Field(None, description="Only include events on or before this datetime (ISO 8601)")


class ContextResponse(BaseModel):
    user_id: str
    query: str
    context_text: str           # ready to inject into LLM system prompt
    messages: list[dict]        # OpenAI-style [{role: system, content: ...}]
    total_memories: int
    embedding_failed: bool
    intent: str                 # detected query intent (e.g. "career", "technical")
    reconsolidation_scheduled: bool = False   # True if background reconsolidation was triggered


# ── GET /memory/{user_id} ─────────────────────────────────────────────────────


class RecentEventItem(BaseModel):
    event_id: str
    raw_text: str
    importance_score: float
    consolidated: bool
    created_at: str
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None

    model_config = {"from_attributes": True}


class RecentEventsResponse(BaseModel):
    user_id: str
    app_ids: list[str]
    events: list[RecentEventItem]


# ── GET /memory/event/{event_id}  GET /memory/event/{event_id}/links ──────────


class MemoryEventDetail(BaseModel):
    event_id: str
    user_id: str
    app_id: str
    raw_text: str
    summary: Optional[str] = None
    importance_score: float
    recall_count: int = 0
    reconsolidation_count: int = 0
    consolidated: bool
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None
    created_at: str
    updated_at: str
    last_reconsolidated_at: Optional[str] = None

    model_config = {"from_attributes": True}


class MemoryLinkItem(BaseModel):
    link_id: str
    from_event_id: str
    from_event_preview: str
    to_event_id: str
    to_event_preview: str
    relation_type: str          # caused | preceded | related | contradicts
    created_at: str


class MemoryLinksResponse(BaseModel):
    event_id: str
    links: list[MemoryLinkItem]


# ── POST /feedback ────────────────────────────────────────────────────────────


class FeedbackRequest(BaseModel):
    event_id: str = Field(..., description="UUID of the recalled event being rated")
    user_id: str = Field(..., description="User submitting the feedback")
    feedback_type: str = Field(
        ..., description="Signal quality: 'positive', 'negative', or 'neutral'"
    )
    app_id: str = Field("default", description="Application namespace")
    comment: str | None = Field(None, description="Optional free-text note")


class FeedbackResponse(BaseModel):
    feedback_id: str
    event_id: str
    new_importance_score: float


# ── GET /identity/{user_id} ───────────────────────────────────────────────────


class IdentityDimensionItem(BaseModel):
    category: str
    dominant_value: str
    confidence: float
    fact_count: int


class BeliefItem(BaseModel):
    statement: str
    category: str
    confidence: float
    evidence_count: int


class IdentityResponse(BaseModel):
    user_id: str
    app_id: str
    summary: str
    dimensions: list[IdentityDimensionItem]
    beliefs: list[BeliefItem]
    total_facts: int
    computed_at: str
    is_empty: bool


# ── GET /graph/facts/{user_id} ────────────────────────────────────────────────


class FactGraphNode(BaseModel):
    id: str
    label: str
    node_type: str          # "user" | "fact"
    category: str | None = None
    confidence: float | None = None
    frequency_count: int | None = None


class FactGraphEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str           # e.g. HAS_PREFERENCE, RELATED_TO
    strength: float | None = None


class FactGraphResponse(BaseModel):
    user_id: str
    app_id: str
    nodes: list[FactGraphNode]
    edges: list[FactGraphEdge]


# ── POST /procedures ──────────────────────────────────────────────────────────


class ProcedureRequest(BaseModel):
    user_id: str = Field(..., description="User this rule applies to")
    trigger: str = Field(..., description="Topic/keyword phrase that activates this rule (e.g. 'LLM deployment')")
    instruction: str = Field(..., description="Behavioral instruction to follow (e.g. 'mention GPU optimization, batching')")
    app_id: str = Field("default", description="Application namespace")
    category: str = Field("topic_response", description="Rule category: topic_response, communication, preference, domain_workflow")
    priority: int = Field(5, ge=1, le=10, description="Priority 1 (low) – 10 (high)")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    source: str = Field("manual", description="Origin: 'manual' (API-created) or 'inferred' (auto-extracted)")


class ProcedureResponse(BaseModel):
    procedure_id: str
    user_id: str
    trigger: str
    instruction: str
    category: str
    priority: int
    is_active: bool
    hit_count: int
    confidence: float
    source: str
    created_at: str

    model_config = {"from_attributes": True}


class ProcedureItem(BaseModel):
    procedure_id: str
    trigger: str
    instruction: str
    category: str
    priority: int
    is_active: bool
    hit_count: int

    model_config = {"from_attributes": True}


class ProcedureListResponse(BaseModel):
    user_id: str
    app_id: str
    procedures: list[ProcedureItem]


class ProcedureUpdateRequest(BaseModel):
    trigger: Optional[str] = None
    instruction: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=10)
    is_active: Optional[bool] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class DeleteProcedureResponse(BaseModel):
    deleted: bool
    procedure_id: str


class DeleteUserProceduresResponse(BaseModel):
    procedures_deleted: int
    user_id: str
    app_id: str


# ── DELETE /memory/{event_id} ─────────────────────────────────────────────────


class DeleteEventResponse(BaseModel):
    deleted: bool
    event_id: str


# ── DELETE /memory/user/{user_id} ─────────────────────────────────────────────


class DeleteUserMemoryResponse(BaseModel):
    events_deleted: int
    user_id: str
    app_id: str


# ── POST /admin/reconsolidate ─────────────────────────────────────────────────


class ReconsolidateRequest(BaseModel):
    event_id: str = Field(..., description="UUID of the event to reconsolidate")
    query: str = Field(..., description="Recall context — the query that surfaced this event")
    user_id: str = Field(..., description="Owner of the event")
    force: bool = Field(False, description="Bypass gate checks (recall_count, importance, cooldown). For testing only.")


class ReconsolidateResponse(BaseModel):
    event_id: str
    user_id: str
    updated: bool
    skipped: bool
    skip_reason: str = ""
    old_summary: str = ""
    new_summary: str = ""


# ── POST /admin/* ──────────────────────────────────────────────────────────────


class AdminJobRequest(BaseModel):
    user_id: Optional[str] = Field(
        None,
        description="User to run the job for. If omitted, runs for all eligible users.",
    )
    app_id: str = Field("default", description="Application namespace")
    min_age_days: Optional[int] = Field(None, description="Override pruning min age in days. 0 means prune regardless of age. For testing only.")
    importance_threshold: Optional[float] = Field(None, description="Override pruning importance threshold. For testing only.")
    min_recall_count: Optional[int] = Field(None, description="Override pruning min recall count. For testing only.")


class AdminJobResult(BaseModel):
    """Summary of one user's job outcome."""
    user_id: str
    app_id: str
    skipped: bool
    detail: str = ""


class AdminJobResponse(BaseModel):
    job: str
    users_processed: int
    results: list[AdminJobResult]


# ── GET /health ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str                  # "ok" | "degraded" | "error"
    version: str = "0.1.0"
    postgres: str = "unknown"    # "ok" | "error"
    neo4j: str = "unknown"       # "ok" | "error"


# ── POST /memory/search ───────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    user_id: str = Field(..., description="User to search memories for")
    query: str = Field(..., description="Natural-language query to search against")
    app_ids: list[str] | None = Field(None, description="App namespaces to search. Defaults to all apps in your token.")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")
    from_date: Optional[datetime] = Field(None, description="Only include events on or after this datetime")
    to_date: Optional[datetime] = Field(None, description="Only include events on or before this datetime")


class SearchResultItem(BaseModel):
    event_id: str
    raw_text: str
    importance_score: float
    hybrid_score: float
    similarity_score: float
    recency_score: float
    consolidated: bool
    created_at: str


class SearchResponse(BaseModel):
    user_id: str
    query: str
    results: list[SearchResultItem]
    total: int
    embedding_failed: bool


# ── GET /memory/export ────────────────────────────────────────────────────────


class ExportEventItem(BaseModel):
    """One line of the NDJSON export.  All fields are present on every row."""
    event_id: str
    raw_text: str
    summary: Optional[str]
    importance_score: Optional[float]
    consolidated: bool
    recall_count: int
    cluster_label: Optional[str]
    created_at: str


# ── GET /admin/users  GET /admin/users/{username}  PATCH /admin/users/{username} ─


class AdminUserItem(BaseModel):
    username: str
    email: Optional[str] = None
    role: str
    app_ids: list[str]
    is_active: bool
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class AdminUsersResponse(BaseModel):
    users: list[AdminUserItem]
    total: int
    limit: int
    offset: int


class AdminUserPatch(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[str] = None
    app_ids: Optional[list[str]] = None
