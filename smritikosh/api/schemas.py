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


# ── GET /health ───────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
