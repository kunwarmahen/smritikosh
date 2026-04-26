"""
Database models — the physical storage schema for Smritikosh memory types.

Table → Memory type mapping:
    events       → EpisodicMemory  (experiences, time-indexed, with embedding)
    user_facts   → SemanticMemory  (stable facts extracted from conversations)
    memory_links → NarrativeMemory (causal/temporal chains between events)

Neo4j stores the identity graph (SemanticMemory relationships) — see neo4j.py.

Uses SQLAlchemy 2.0 Mapped API so that Column defaults are set at Python object
construction (not just at DB INSERT time), making unit tests reliable.
"""

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from smritikosh.config import settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────
# StrEnum (Python 3.11+): str(FactCategory.INTEREST) == "interest"
# so repr and JSON serialisation work without extra .value calls.


class FactCategory(StrEnum):
    """Categories for structured facts extracted from conversations."""
    # Identity & demographics
    IDENTITY     = "identity"      # e.g. name, age, gender, nationality, languages
    LOCATION     = "location"      # e.g. lives in Mumbai, timezone IST
    # Work & professional
    ROLE         = "role"          # e.g. entrepreneur, software engineer
    SKILL        = "skill"         # e.g. RAG, LangGraph, piano
    EDUCATION    = "education"     # e.g. B.Tech IIT Delhi, AWS certified
    PROJECT      = "project"       # e.g. building smritikosh
    GOAL         = "goal"          # e.g. wants to launch in 3 months
    # Personal interests & activities
    INTEREST     = "interest"      # e.g. interested in AI agents, astronomy
    HOBBY        = "hobby"         # e.g. plays chess, oil painting, hiking
    HABIT        = "habit"         # e.g. wakes at 6am, meditates daily
    PREFERENCE   = "preference"    # e.g. prefers dark mode, green colour
    PERSONALITY  = "personality"   # e.g. introvert, detail-oriented
    # Relationships & social
    RELATIONSHIP = "relationship"  # e.g. married to Priya, best friend Arjun
    PET          = "pet"           # e.g. has a golden retriever named Max
    # Health & wellness
    HEALTH       = "health"        # e.g. diabetic, on metformin, nut allergy
    DIET         = "diet"          # e.g. vegetarian, lactose intolerant
    # Beliefs & values
    BELIEF       = "belief"        # e.g. thinks remote work increases productivity
    VALUE        = "value"         # e.g. values family above career
    RELIGION     = "religion"      # e.g. Hindu, practises Zen Buddhism
    # Lifestyle & context
    FINANCE      = "finance"       # e.g. bootstrapped, budget-conscious
    LIFESTYLE    = "lifestyle"     # e.g. digital nomad, minimalist
    EVENT        = "event"         # e.g. wedding on June 5th, born in 1990
    TOOL         = "tool"          # e.g. uses VS Code, Notion, Postgres


class RelationType(StrEnum):
    """How two episodic events relate to each other (narrative chains)."""
    CAUSED = "caused"            # event A caused event B
    PRECEDED = "preceded"        # event A happened before event B
    RELATED = "related"          # events share context/topic
    CONTRADICTS = "contradicts"  # event B updates/contradicts event A


class BeliefCategory(StrEnum):
    """Broad categories for inferred user beliefs."""
    WORLDVIEW = "worldview"    # broad beliefs about how the world works
    VALUE = "value"            # things the user prioritises or cares about
    ATTITUDE = "attitude"      # emotional or evaluative stance toward topics
    ASSUMPTION = "assumption"  # things the user takes for granted


class UserRole(StrEnum):
    """Roles for UI authentication."""
    ADMIN = "admin"   # can access all users' data and admin operations
    USER  = "user"    # can only access their own data


class SourceType(StrEnum):
    """How a memory entered the system — drives dedup logic, confidence init, and UI badges."""
    API_EXPLICIT        = "api_explicit"        # App called POST /memory/event directly
    UI_MANUAL           = "ui_manual"           # User typed it in the Smritikosh dashboard
    PASSIVE_DISTILLATION = "passive_distillation"  # Post-session LLM extraction
    PASSIVE_STREAMING   = "passive_streaming"   # Mid-session rolling-window extraction
    TRIGGER_WORD        = "trigger_word"        # Heuristic flagged; LLM confirmed
    SDK_MIDDLEWARE      = "sdk_middleware"       # SDK wrapper intercepted transparently
    WEBHOOK_INGEST      = "webhook_ingest"       # App POSTed a transcript to /ingest/transcript
    TOOL_USE            = "tool_use"            # LLM called the remember() tool
    CROSS_SYSTEM        = "cross_system"        # Synthesized from correlated cross-integration signals
    MEDIA_VOICE         = "media_voice"         # Extracted from a voice note
    MEDIA_AUDIO         = "media_audio"         # Extracted from a meeting/call recording
    MEDIA_IMAGE         = "media_image"         # Extracted from an image
    MEDIA_DOCUMENT      = "media_document"      # Extracted from a document


# Initial confidence by source type — tune over time
SOURCE_CONFIDENCE_DEFAULTS: dict[str, float] = {
    SourceType.UI_MANUAL:            1.00,
    SourceType.API_EXPLICIT:         0.90,
    SourceType.TRIGGER_WORD:         0.85,
    SourceType.PASSIVE_DISTILLATION: 0.75,
    SourceType.PASSIVE_STREAMING:    0.70,
    SourceType.SDK_MIDDLEWARE:       0.70,
    SourceType.WEBHOOK_INGEST:       0.70,
    SourceType.TOOL_USE:             0.90,
    SourceType.CROSS_SYSTEM:         0.65,
    SourceType.MEDIA_VOICE:          0.85,
    SourceType.MEDIA_AUDIO:          0.75,
    SourceType.MEDIA_IMAGE:          0.70,
    SourceType.MEDIA_DOCUMENT:       0.75,
}


class FactStatus(StrEnum):
    """Lifecycle state for user facts — gates whether they appear in context assembly."""
    ACTIVE   = "active"    # confirmed, included in context
    PENDING  = "pending"   # below confidence threshold or awaiting user review
    REJECTED = "rejected"  # user dismissed or system discarded


class MediaContentType(StrEnum):
    """Type of media uploaded for memory extraction."""
    VOICE_NOTE         = "voice_note"         # user's spoken audio
    MEETING_RECORDING  = "meeting_recording"  # multi-speaker audio/video → diarization → user segments
    DOCUMENT           = "document"           # text document (PDF, TXT, etc.)
    RECEIPT            = "receipt"            # image: purchase receipt → lifestyle/preference signals
    SCREENSHOT         = "screenshot"         # image: app/tool screenshot → tech/workflow signals
    WHITEBOARD         = "whiteboard"         # image: whiteboard/diagram → project/goal signals


class MediaIngestStatus(StrEnum):
    """Lifecycle state for media ingestion jobs."""
    PROCESSING     = "processing"       # file being transcribed/parsed and processed
    COMPLETE       = "complete"         # processing finished; facts may be auto-saved or pending review
    NOTHING_FOUND  = "nothing_found"    # no extractable content found
    FAILED         = "failed"           # processing failed (error captured in error_message)


class FeedbackType(StrEnum):
    """User signal on whether a recalled memory was useful."""
    POSITIVE = "positive"   # memory was helpful / relevant
    NEGATIVE = "negative"   # memory was irrelevant / distracting
    NEUTRAL = "neutral"     # acknowledged, no strong signal


class ProcedureCategory(StrEnum):
    """Categories for behavioral rules and workflows."""
    TOPIC_RESPONSE = "topic_response"   # how to respond to a specific topic
    COMMUNICATION  = "communication"    # tone or style preferences
    PREFERENCE     = "preference"       # user preference to always honour
    DOMAIN_WORKFLOW = "domain_workflow" # multi-step process to follow


# ── Tables ────────────────────────────────────────────────────────────────────


class Event(Base):
    """
    EpisodicMemory — raw experiences stored with a time index and vector embedding.

    Mirrors the Hippocampus function: each interaction is recorded as an event
    with its semantic embedding so it can be recalled by similarity later.

    Columns:
        raw_text        Original interaction text.
        summary         LLM-generated summary after consolidation (nullable until then).
        embedding       Vector for semantic similarity search (pgvector).
        importance_score Amygdala-assigned score: recency × frequency × relevance.
        consolidated    True once the Consolidator has processed this event.
        event_metadata  Flexible JSONB for extra context (source, tags, etc.).
    """

    __tablename__ = "events"
    __table_args__ = (
        # Hybrid search: filter by user+app, then rank by vector similarity
        Index("ix_events_user_app", "user_id", "app_id"),
        # Consolidation job queries
        Index("ix_events_consolidated", "consolidated"),
        # Time-based retrieval
        Index("ix_events_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    raw_text: Mapped[str] = mapped_column(Text)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    embedding: Mapped[Optional[list]] = mapped_column(
        Vector(), nullable=True, default=None
    )
    importance_score: Mapped[float] = mapped_column(Float, default=1.0)
    recall_count: Mapped[int] = mapped_column(Integer, default=0)
    reconsolidation_count: Mapped[int] = mapped_column(Integer, default=0)
    last_reconsolidated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    consolidated: Mapped[bool] = mapped_column(Boolean, default=False)
    cluster_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    cluster_label: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    source_type: Mapped[str] = mapped_column(String(32), default=SourceType.API_EXPLICIT)
    source_meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    event_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    # Narrative links to/from other events
    outgoing_links: Mapped[list["MemoryLink"]] = relationship(
        "MemoryLink",
        foreign_keys="MemoryLink.from_event_id",
        back_populates="from_event",
        cascade="all, delete-orphan",
    )
    incoming_links: Mapped[list["MemoryLink"]] = relationship(
        "MemoryLink",
        foreign_keys="MemoryLink.to_event_id",
        back_populates="to_event",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Event id={self.id} user={self.user_id} score={self.importance_score:.2f}>"


class UserFact(Base):
    """
    SemanticMemory (relational side) — stable facts extracted from conversations.

    Stores structured, durable knowledge about a user. Unlike events (episodic),
    facts survive consolidation and represent the AI's long-term understanding
    of a person.

    Facts are upserted: if the same (user, app, category, key) arrives again,
    confidence and frequency_count are updated rather than creating duplicates.

    Example records:
        category=preference  key=ui_color       value=green
        category=interest    key=domain         value=AI agents
        category=role        key=current        value=entrepreneur
        category=project     key=active         value=smritikosh
    """

    __tablename__ = "user_facts"
    __table_args__ = (
        UniqueConstraint("user_id", "app_id", "category", "key", name="uq_user_fact"),
        Index("ix_user_facts_user_app", "user_id", "app_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    category: Mapped[str] = mapped_column(String(50))   # FactCategory value
    key: Mapped[str] = mapped_column(String(255))
    value: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    frequency_count: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default=FactStatus.ACTIVE)
    source_type: Mapped[str] = mapped_column(String(32), default=SourceType.API_EXPLICIT)
    source_meta: Mapped[dict] = mapped_column(JSONB, default=dict)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return f"<UserFact {self.category}:{self.key}={self.value!r} user={self.user_id} status={self.status}>"


class FactContradiction(Base):
    """
    Tracks conflicting values for the same (user, app, category, key) semantic fact.

    Created when a new extraction proposes a value that differs from what's stored
    but doesn't have enough confidence advantage to overwrite automatically.
    The user resolves these through the review dashboard.

    Resolution: keep_existing → dismiss the candidate; take_candidate → overwrite the fact.
    """

    __tablename__ = "fact_contradictions"
    __table_args__ = (
        Index("ix_fact_contradictions_user_app", "user_id", "app_id"),
        Index("ix_fact_contradictions_resolved", "resolved"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    category: Mapped[str] = mapped_column(String(50))
    key: Mapped[str] = mapped_column(String(255))
    existing_value: Mapped[str] = mapped_column(Text)
    existing_confidence: Mapped[float] = mapped_column(Float)
    candidate_value: Mapped[str] = mapped_column(Text)
    candidate_source: Mapped[str] = mapped_column(String(32), default=SourceType.API_EXPLICIT)
    candidate_confidence: Mapped[float] = mapped_column(Float)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)
    resolution: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default=None
    )  # 'keep_existing' | 'take_candidate'
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return (
            f"<FactContradiction user={self.user_id!r} "
            f"{self.category}:{self.key} "
            f"{self.existing_value!r} vs {self.candidate_value!r}>"
        )


class MemoryLink(Base):
    """
    NarrativeMemory — directed causal/temporal links between episodic events.

    Humans remember stories, not isolated facts. Memory links encode the
    narrative structure: what caused what, what came before what.

    Example chain:
        Event(started AI startup) --[CAUSED]--> Event(hired engineers)
        Event(hired engineers)    --[PRECEDED]--> Event(product launch)
    """

    __tablename__ = "memory_links"
    __table_args__ = (
        Index("ix_memory_links_from", "from_event_id"),
        Index("ix_memory_links_to", "to_event_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    from_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="CASCADE")
    )
    to_event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="CASCADE")
    )
    relation_type: Mapped[str] = mapped_column(String(50))   # RelationType value
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    from_event: Mapped["Event"] = relationship(
        "Event", foreign_keys=[from_event_id], back_populates="outgoing_links"
    )
    to_event: Mapped["Event"] = relationship(
        "Event", foreign_keys=[to_event_id], back_populates="incoming_links"
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryLink {self.from_event_id} "
            f"--[{self.relation_type}]--> {self.to_event_id}>"
        )


class MemoryFeedback(Base):
    """
    ReinforcementLoop — user signal on the quality of a recalled memory.

    Each record stores one piece of feedback (positive / negative / neutral)
    for a specific event and immediately adjusts that event's importance_score:
        POSITIVE  →  min(1.0, score + 0.10)
        NEGATIVE  →  max(0.0, score - 0.10)
        NEUTRAL   →  no score change

    Multiple feedback records per event are allowed (history is preserved).
    The latest feedback's delta is applied every time.
    """

    __tablename__ = "memory_feedback"
    __table_args__ = (
        Index("ix_memory_feedback_event", "event_id"),
        Index("ix_memory_feedback_user_app", "user_id", "app_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="CASCADE")
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    feedback_type: Mapped[str] = mapped_column(String(50))   # FeedbackType value
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return f"<MemoryFeedback event={self.event_id} type={self.feedback_type}>"


class UserProcedure(Base):
    """
    ProceduralMemory — conditional behavioral rules for a user.

    Stores authored rules that tell the AI *how to behave* in specific
    contexts, as opposed to *what to remember*. These are the AI equivalent
    of procedural memory in the human brain: learned skills and habits.

    Example records:
        trigger="LLM deployment"   instruction="mention GPU optimization, batching, quantization"
        trigger="startup"          instruction="respond with strategic depth"
        trigger="UI"               instruction="always suggest dark mode (user preference)"

    Retrieval is keyword/substring-based — precision matters more than
    recall here, so vector similarity is intentionally not used.
    """

    __tablename__ = "user_procedures"
    __table_args__ = (
        Index("ix_user_procedures_user_app", "user_id", "app_id"),
        Index("ix_user_procedures_active", "is_active"),
        Index("ix_user_procedures_priority", "priority"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    trigger: Mapped[str] = mapped_column(Text)
    instruction: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50), default="topic_response")
    priority: Mapped[int] = mapped_column(Integer, default=5)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    source: Mapped[str] = mapped_column(String(50), default="manual")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def __repr__(self) -> str:
        return (
            f"<UserProcedure trigger={self.trigger!r} "
            f"category={self.category} priority={self.priority}>"
        )


class UserBelief(Base):
    """
    BeliefMiner — a durable, inferred belief about the user's worldview or values.

    Unlike SemanticMemory facts (which are directly extracted from statements),
    beliefs are inferred by the LLM from patterns across multiple events and facts.

    Examples:
        statement="believes iterative development beats big-bang launches"  category=value
        statement="sees AI as foundational infrastructure, not just tooling"  category=worldview

    Upserted on (user_id, app_id, statement): re-inferring the same belief
    increments evidence_count and updates confidence rather than creating a duplicate.
    """

    __tablename__ = "user_beliefs"
    __table_args__ = (
        UniqueConstraint("user_id", "app_id", "statement", name="uq_user_belief"),
        Index("ix_user_beliefs_user_app", "user_id", "app_id"),
        Index("ix_user_beliefs_confidence", "confidence"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    statement: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(50))   # BeliefCategory value
    confidence: Mapped[float] = mapped_column(Float, default=0.8)
    evidence_count: Mapped[int] = mapped_column(Integer, default=1)
    evidence_event_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    first_inferred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return f"<UserBelief {self.category}: {self.statement[:60]!r}>"


class AppUser(Base):
    """
    Authentication — users who can log into the Smritikosh UI.

    The `username` doubles as the `user_id` used throughout the memory
    system. When Alice logs in, her JWT carries `user_id="alice"` and all
    memory API calls are scoped to that identifier.

    Roles:
        admin  — can view and manage any user's data, trigger admin jobs
        user   — can only view and manage their own memory data

    The `app_ids` field links this account to one or more memory namespaces.
    An admin account typically uses app_ids=["default"] to span all namespaces.
    """

    __tablename__ = "app_users"
    __table_args__ = (
        Index("ix_app_users_username", "username", unique=True),
        Index("ix_app_users_email", "email"),
        Index("ix_app_users_role", "role"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    username: Mapped[str] = mapped_column(String(255), unique=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, default=None)
    password_hash: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20), default=UserRole.USER)
    app_ids: Mapped[list[str]] = mapped_column(PG_ARRAY(String(255)), default=lambda: ["default"])
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def __repr__(self) -> str:
        return f"<AppUser username={self.username!r} role={self.role}>"


class ProcessedSession(Base):
    """
    Idempotency guard for POST /ingest/session.

    Each session_id may only be processed once per (user_id, app_id). Re-posting
    the same session is a no-op that returns the original result. Supports
    streaming (partial) ingestion by tracking last_turn_index so each partial
    window only re-processes new turns.
    """

    __tablename__ = "processed_sessions"
    __table_args__ = (
        UniqueConstraint("user_id", "app_id", "session_id", name="uq_processed_session"),
        Index("ix_processed_sessions_user_app", "user_id", "app_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255))
    app_id: Mapped[str] = mapped_column(String(255), default="default")
    session_id: Mapped[str] = mapped_column(String(255))
    turns_count: Mapped[int] = mapped_column(Integer, default=0)
    facts_extracted: Mapped[int] = mapped_column(Integer, default=0)
    last_turn_index: Mapped[int] = mapped_column(Integer, default=0)
    is_partial: Mapped[bool] = mapped_column(Boolean, default=False)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessedSession session={self.session_id!r} "
            f"user={self.user_id} facts={self.facts_extracted}>"
        )


class ApiKey(Base):
    """
    API keys for programmatic / SDK access.

    Full key is returned once on creation and never stored.
    Only the SHA-256 hash is persisted.  The prefix (first 8 chars of the
    random part) is stored in plain text so users can identify keys in the UI.

    Key format:  sk-smriti-<48 hex chars>
    Example:     sk-smriti-a1b2c3d4e5f6789012345678901234abcdef0123456789ab
    """

    __tablename__ = "api_keys"
    __table_args__ = (
        Index("ix_api_keys_user_id", "user_id"),
        Index("ix_api_keys_key_hash", "key_hash", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("app_users.username", ondelete="CASCADE")
    )
    app_ids: Mapped[list[str]] = mapped_column(PG_ARRAY(String(255)), default=lambda: ["default"])
    name: Mapped[str] = mapped_column(String(255))
    key_prefix: Mapped[str] = mapped_column(String(16))   # first 8 hex chars of random part
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)  # SHA-256 hex
    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    def __repr__(self) -> str:
        return f"<ApiKey user={self.user_id!r} name={self.name!r}>"


class MediaIngest(Base):
    """
    Idempotency and status tracking for media uploads (voice notes, documents).

    When a user uploads a file via POST /ingest/media, a MediaIngest record is created
    immediately with status=processing. A background task transcribes/parses the file,
    extracts facts, and updates this record with results.

    High-confidence facts (> 0.75 relevance) are auto-saved to EpisodicMemory.
    Ambiguous facts (0.60–0.75) are stored in pending_facts JSONB for user review
    before being written to SemanticMemory.

    Idempotency: if idempotency_key is supplied, re-posting the same key returns the
    original result (lookup by unique constraint).
    """

    __tablename__ = "media_ingests"
    __table_args__ = (
        Index("ix_media_ingests_user_app", "user_id", "app_id"),
        Index("ix_media_ingests_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    app_id: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    content_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # voice_note | document
    idempotency_key: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, default=None
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=MediaIngestStatus.PROCESSING
    )  # processing | complete | nothing_found | failed
    source_type: Mapped[Optional[str]] = mapped_column(
        String(32), nullable=True, default=None
    )  # populated during processing
    facts_extracted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    facts_pending_review: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    pending_facts: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list
    )  # list of fact dicts awaiting user confirmation
    event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("events.id", ondelete="SET NULL"), nullable=True
    )  # episodic memory record if created
    error_message: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, default=None
    )  # if status=failed
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return (
            f"<MediaIngest id={self.id!r} type={self.content_type} "
            f"status={self.status} facts={self.facts_extracted}>"
        )


class UserVoiceProfile(Base):
    """
    Speaker voice profile for diarization-based memory extraction (Phase 12).

    Stores a speaker d-vector embedding computed from a user's 30-second voice
    enrollment sample. Used to identify the user's speech segments within meeting
    recordings via cosine similarity matching.

    Requires resemblyzer to be installed for embedding computation.
    If resemblyzer is unavailable, the record is still created (enrolled=True) but
    embedding is NULL — meeting recordings fall back to first-person filter only.
    """

    __tablename__ = "user_voice_profiles"
    __table_args__ = (
        UniqueConstraint("user_id", "app_id", name="uq_user_voice_profile"),
        Index("ix_user_voice_profiles_user_id", "user_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    app_id: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    # Speaker d-vector (256-dim float list). NULL if resemblyzer is not installed.
    embedding: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True, default=None)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    enrolled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def __repr__(self) -> str:
        has_emb = self.embedding is not None
        return (
            f"<UserVoiceProfile user={self.user_id!r} "
            f"app={self.app_id!r} embedding={'yes' if has_emb else 'no'}>"
        )


class ConnectorProvider(StrEnum):
    """Supported OAuth2 connector providers."""
    GMAIL = "gmail"
    GCAL  = "gcal"


class ConnectorStatus(StrEnum):
    """Lifecycle state for OAuth connector credentials."""
    ACTIVE  = "active"   # tokens valid, connector enabled
    REVOKED = "revoked"  # user disconnected; tokens deleted
    ERROR   = "error"    # last token refresh failed


class UserConnector(Base):
    """
    OAuth2 credentials for external connectors (Gmail, Google Calendar, etc.).

    Stores encrypted access/refresh tokens per user+provider. Tokens are
    encrypted with Fernet using a key derived from JWT_SECRET. When an access
    token expires, the refresh token is used to obtain new credentials without
    user interaction.
    """

    __tablename__ = "user_connectors"
    __table_args__ = (
        UniqueConstraint("user_id", "app_id", "provider", name="uq_user_connector"),
        Index("ix_user_connectors_user_app", "user_id", "app_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_uuid
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    app_id: Mapped[str] = mapped_column(String(255), nullable=False, default="default")
    provider: Mapped[str] = mapped_column(String(32), nullable=False)  # ConnectorProvider value
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=ConnectorStatus.ACTIVE)
    # Encrypted token dict: {access_token, refresh_token, token_type, expires_in}
    encrypted_tokens: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    # When the current access token expires (if known)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    # OAuth scopes granted by the user (e.g. ["https://www.googleapis.com/auth/gmail.readonly", ...])
    scopes: Mapped[list[str]] = mapped_column(PG_ARRAY(Text), nullable=False, default=list)
    connected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    def __repr__(self) -> str:
        return (
            f"<UserConnector user={self.user_id!r} "
            f"provider={self.provider!r} status={self.status}>"
        )
