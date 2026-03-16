"""
Pydantic request/response schemas for the Smritikosh API.

Kept separate from DB models so the API contract can evolve independently
of the internal storage representation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


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
    app_id: str = Field("default", description="Application namespace")
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

    model_config = {"from_attributes": True}


class RecentEventsResponse(BaseModel):
    user_id: str
    app_id: str
    events: list[RecentEventItem]


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
    app_id: str = Field("default", description="Application namespace")
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
