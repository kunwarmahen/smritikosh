"""
Pydantic request/response schemas for the Smritikosh API.

Kept separate from DB models so the API contract can evolve independently
of the internal storage representation.
"""

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


class ContextResponse(BaseModel):
    user_id: str
    query: str
    context_text: str           # ready to inject into LLM system prompt
    messages: list[dict]        # OpenAI-style [{role: system, content: ...}]
    total_memories: int
    embedding_failed: bool
    intent: str                 # detected query intent (e.g. "career", "technical")


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


# ── GET /health ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
