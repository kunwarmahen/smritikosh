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
from sqlalchemy.dialects.postgresql import JSONB, UUID
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
    PREFERENCE = "preference"      # e.g. prefers dark mode, green color
    INTEREST = "interest"          # e.g. interested in AI agents
    ROLE = "role"                  # e.g. entrepreneur, engineer
    PROJECT = "project"            # e.g. building smritikosh
    SKILL = "skill"                # e.g. RAG, LangGraph
    GOAL = "goal"                  # e.g. wants to launch in 3 months
    RELATIONSHIP = "relationship"  # e.g. works with Alice


class RelationType(StrEnum):
    """How two episodic events relate to each other (narrative chains)."""
    CAUSED = "caused"            # event A caused event B
    PRECEDED = "preceded"        # event A happened before event B
    RELATED = "related"          # events share context/topic
    CONTRADICTS = "contradicts"  # event B updates/contradicts event A


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
        Vector(settings.embedding_dimensions), nullable=True, default=None
    )
    importance_score: Mapped[float] = mapped_column(Float, default=1.0)
    consolidated: Mapped[bool] = mapped_column(Boolean, default=False)
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
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )

    def __repr__(self) -> str:
        return f"<UserFact {self.category}:{self.key}={self.value!r} user={self.user_id}>"


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
