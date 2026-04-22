"""
Typed return values for the Smritikosh SDK.

These mirror the API response schemas but live in the SDK layer so callers
don't need to depend on the server-side `smritikosh.api.schemas` module.
"""

from dataclasses import dataclass, field


@dataclass
class EncodedEvent:
    """Returned by SmritikoshClient.encode() after storing a memory."""
    event_id: str
    user_id: str
    importance_score: float
    facts_extracted: int
    extraction_failed: bool


@dataclass
class RecentEvent:
    """One event item from SmritikoshClient.get_recent()."""
    event_id: str
    raw_text: str
    importance_score: float
    consolidated: bool
    created_at: str


@dataclass
class MemoryContext:
    """
    Returned by SmritikoshClient.build_context().

    context_text  — ready to drop into any LLM system prompt.
    messages      — OpenAI-style [{"role": "system", "content": "..."}].
    """
    user_id: str
    query: str
    context_text: str
    messages: list[dict]
    total_memories: int
    embedding_failed: bool

    def is_empty(self) -> bool:
        return self.total_memories == 0


@dataclass
class FeedbackRecord:
    """Returned by SmritikoshClient.submit_feedback()."""

    feedback_id: str
    event_id: str
    new_importance_score: float


@dataclass
class BeliefItem:
    """One inferred belief from SmritikoshClient.get_identity()."""

    statement: str
    category: str
    confidence: float
    evidence_count: int


@dataclass
class IdentityDimensionItem:
    """One category slice of a user's identity (e.g. all 'role' facts)."""

    category: str
    dominant_value: str
    confidence: float
    fact_count: int


@dataclass
class IdentityProfile:
    """
    Returned by SmritikoshClient.get_identity().

    summary      — LLM-generated narrative description of the user.
    dimensions   — per-category breakdown of identity facts.
    """

    user_id: str
    app_id: str
    summary: str
    dimensions: list[IdentityDimensionItem]
    beliefs: list[BeliefItem]
    total_facts: int
    computed_at: str
    is_empty: bool


@dataclass
class ProcedureRecord:
    """One behavioral rule from SmritikoshClient.list_procedures()."""
    procedure_id: str
    trigger: str
    instruction: str
    category: str
    priority: int
    is_active: bool
    hit_count: int


@dataclass
class ProcedureCreated:
    """Returned by SmritikoshClient.store_procedure()."""
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


@dataclass
class DeleteProcedureResult:
    """Returned by SmritikoshClient.delete_procedure()."""
    deleted: bool
    procedure_id: str


@dataclass
class DeleteUserProceduresResult:
    """Returned by SmritikoshClient.delete_user_procedures()."""
    procedures_deleted: int
    user_id: str
    app_id: str


@dataclass
class DeleteEventResult:
    """Returned by SmritikoshClient.delete_event()."""
    deleted: bool
    event_id: str


@dataclass
class DeleteUserMemoryResult:
    """Returned by SmritikoshClient.delete_user_memory()."""
    events_deleted: int
    user_id: str
    app_id: str


@dataclass
class AdminJobResult:
    """Summary of one user's admin job outcome."""
    user_id: str
    app_id: str
    skipped: bool
    detail: str


@dataclass
class AdminJobResponse:
    """Returned by SmritikoshClient admin methods."""
    job: str
    users_processed: int
    results: list[AdminJobResult]


@dataclass
class ReconsolidationResult:
    """Returned by SmritikoshClient.reconsolidate()."""
    event_id: str
    user_id: str
    updated: bool
    skipped: bool
    skip_reason: str
    old_summary: str
    new_summary: str


@dataclass
class HealthStatus:
    status: str
    version: str
    postgres: str = "unknown"
    neo4j: str = "unknown"


@dataclass
class SearchResultItem:
    """One scored event from SmritikoshClient.search()."""
    event_id: str
    raw_text: str
    importance_score: float
    hybrid_score: float
    similarity_score: float
    recency_score: float
    consolidated: bool
    created_at: str


@dataclass
class SearchResult:
    """Returned by SmritikoshClient.search()."""
    user_id: str
    query: str
    results: list[SearchResultItem]
    total: int
    embedding_failed: bool


@dataclass
class SessionIngestResult:
    """Returned by SmritikoshClient.ingest_session()."""
    session_id: str
    turns_processed: int
    facts_extracted: int
    extraction_skipped: bool
    already_processed: bool
    partial: bool


@dataclass
class IngestResult:
    """Returned by SmritikoshClient ingest methods."""
    source: str
    events_ingested: int
    events_failed: int
    event_ids: list[str]
