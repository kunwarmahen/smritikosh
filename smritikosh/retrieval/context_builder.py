"""
ContextBuilder — retrieval and assembly of memory context before LLM calls.

Mirrors the prefrontal cortex function: given a current query, pull the most
relevant fragments from all memory systems and assemble them into a coherent
context block that gets injected into the LLM prompt.

Pipeline:
    query text
        │
        └─► LLMAdapter.embed(query)                          (async, step 1)
                │
                ├─► EpisodicMemory.hybrid_search(embedding)  ┐
                ├─► SemanticMemory.get_user_profile()         ├── concurrent (step 2)
                └─► EpisodicMemory.get_recent()              ┘
                        │
                        └─► MemoryContext  (assembled, returned to caller)

MemoryContext.as_prompt_text() renders everything as a structured string
ready to prepend to the LLM system prompt.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from neo4j import AsyncSession as NeoSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event, UserProcedure
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory, SearchResult
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.memory.procedural import ProceduralMemory
from smritikosh.memory.semantic import SemanticMemory, UserProfile
from smritikosh.retrieval.intent_classifier import IntentClassifier, QueryIntent

logger = logging.getLogger(__name__)


# ── Return type ───────────────────────────────────────────────────────────────


@dataclass
class MemoryContext:
    """
    Assembled memory context for a single user query.

    Contains three memory slices:
        similar_events  — past events semantically close to the current query
        user_profile    — structured facts (preferences, interests, roles …)
        recent_events   — the most recent interactions, regardless of topic

    Use as_prompt_text() to get a single string for LLM injection, or
    as_messages() for a structured OpenAI-style message list.
    """

    user_id: str
    query: str
    similar_events: list[SearchResult] = field(default_factory=list)
    user_profile: UserProfile | None = None
    recent_events: list[Event] = field(default_factory=list)
    # True if query embedding failed — context assembled without vector recall
    embedding_failed: bool = False
    # Detected query intent — used to tune retrieval weights
    intent: str = QueryIntent.GENERAL
    # Narrative chains originating from the top similar event (Phase 3)
    narrative_chains: list[list[Event]] = field(default_factory=list)
    # Procedural rules that fired for this query
    procedures: list[UserProcedure] = field(default_factory=list)

    # ── Rendering ─────────────────────────────────────────────────────────

    def as_prompt_text(self) -> str:
        """
        Render assembled memory as a structured string for LLM prompt injection.

        Example output:
            ## User Memory Context

            ### Who this user is:
            Role: current=entrepreneur
            Interests: domain=AI agents, topic=LLM infrastructure
            Preferences: ui_color=green

            ### Recent activity (last 5 interactions):
            - [2026-03-15] User discussed building an AI memory startup
            - [2026-03-14] User asked about RAG pipeline optimisation

            ### Relevant past memories:
            - User decided to build smritikosh as core product  [score: 0.91]
            - User has experience with LangGraph and vector DBs  [score: 0.84]
        """
        sections: list[str] = ["## User Memory Context\n"]

        # ── Behavioral rules (procedural memory) ──────────────────────────
        if self.procedures:
            active = sorted(self.procedures, key=lambda p: p.priority, reverse=True)
            sections.append("### Behavioral rules for this user:")
            for proc in active:
                sections.append(
                    f"- [{proc.category}, priority {proc.priority}]"
                    f" When discussing \"{proc.trigger}\": {proc.instruction}"
                )
            sections.append("")

        # ── Identity / semantic facts ──────────────────────────────────────
        if self.user_profile and self.user_profile.facts:
            sections.append("### Who this user is:")
            sections.append(self.user_profile.as_text_summary())
            sections.append("")

        # ── Recent timeline ────────────────────────────────────────────────
        if self.recent_events:
            sections.append(f"### Recent activity (last {len(self.recent_events)} interactions):")
            for event in self.recent_events:
                date_str = _format_date(event.created_at)
                text = event.summary or event.raw_text
                sections.append(f"- [{date_str}] {_truncate(text, 120)}")
            sections.append("")

        # ── Semantically similar past memories ────────────────────────────
        if self.similar_events:
            sections.append("### Relevant past memories:")
            for sr in self.similar_events:
                text = sr.event.summary or sr.event.raw_text
                sections.append(
                    f"- {_truncate(text, 120)}  [score: {sr.hybrid_score:.2f}]"
                )
            sections.append("")

        # ── Narrative chains ───────────────────────────────────────────────
        if self.narrative_chains:
            sections.append("### Memory chains (how events unfolded):")
            for chain in self.narrative_chains:
                parts = [
                    f"[{_format_date(e.created_at)}] {_truncate(e.summary or e.raw_text, 80)}"
                    for e in chain
                ]
                sections.append(" → ".join(parts))
            sections.append("")

        if len(sections) == 1:
            # Only the header — no memory found
            sections.append("(no memory stored for this user yet)")

        return "\n".join(sections)

    def as_messages(self) -> list[dict[str, str]]:
        """
        Return memory context as an OpenAI-style system message.
        Append to your message list before the user turn.

        Example:
            messages = context.as_messages()
            messages.append({"role": "user", "content": user_query})
            response = await llm.complete(messages)
        """
        return [{"role": "system", "content": self.as_prompt_text()}]

    def is_empty(self) -> bool:
        """True if no memory was found for this user."""
        no_profile = not self.user_profile or not self.user_profile.facts
        return (
            not self.similar_events
            and no_profile
            and not self.recent_events
            and not self.procedures
        )

    def total_memories(self) -> int:
        """Total number of memory fragments assembled."""
        profile_count = len(self.user_profile.facts) if self.user_profile else 0
        return (
            len(self.similar_events)
            + profile_count
            + len(self.recent_events)
            + len(self.procedures)
        )


# ── ContextBuilder ────────────────────────────────────────────────────────────


class ContextBuilder:
    """
    Retrieves and assembles memory context for a given user query.

    All three retrieval operations (hybrid search, profile fetch, recent events)
    run concurrently after the query embedding is generated — minimising latency.

    Usage:
        builder = ContextBuilder(llm=llm, episodic=episodic, semantic=semantic)

        async with db_session() as pg, neo4j_session() as neo:
            ctx = await builder.build(pg, neo, user_id="u1", query="What should I build?")
            prompt_text = ctx.as_prompt_text()
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        *,
        top_k_similar: int = 5,
        recent_limit: int = 5,
        min_profile_confidence: float = 0.5,
        intent_classifier: IntentClassifier | None = None,
        narrative: NarrativeMemory | None = None,
        include_chains: bool = False,
        procedural: ProceduralMemory | None = None,
        top_k_procedures: int = 5,
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.top_k_similar = top_k_similar
        self.recent_limit = recent_limit
        self.min_profile_confidence = min_profile_confidence
        self.intent_classifier = intent_classifier
        self.narrative = narrative
        self.include_chains = include_chains
        self.procedural = procedural
        self.top_k_procedures = top_k_procedures

    # ── Primary entry point ────────────────────────────────────────────────

    async def build(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        query: str,
        app_id: str = "default",
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> MemoryContext:
        """
        Build a MemoryContext for the given query.

        Steps:
            1. Embed the query (required for vector recall).
            2. Concurrently fetch: similar events, user profile, recent events.
            3. Assemble and return MemoryContext.

        If embedding fails, similar_events will be empty but profile and
        recent events are still included (partial context is better than none).
        """
        # ── 1. Classify intent (sync, no I/O) ────────────────────────────
        intent_result = (
            self.intent_classifier.classify(query)
            if self.intent_classifier is not None
            else None
        )
        detected_intent = intent_result.intent if intent_result else QueryIntent.GENERAL

        # ── 2. Embed query ────────────────────────────────────────────────
        embedding, embedding_failed = await self._embed_query(query, user_id)

        # ── 3. Concurrent retrieval ───────────────────────────────────────
        similar_task = (
            self.episodic.hybrid_search(
                pg_session,
                user_id,
                embedding,
                app_id=app_id,
                top_k=self.top_k_similar,
                weights_override=intent_result.weights if intent_result else None,
                from_date=from_date,
                to_date=to_date,
            )
            if embedding is not None
            else _empty()
        )
        profile_task = self.semantic.get_user_profile(
            neo_session, user_id, app_id,
            min_confidence=self.min_profile_confidence,
        )
        recent_task = self.episodic.get_recent(
            pg_session, user_id, app_id, limit=self.recent_limit,
            from_date=from_date, to_date=to_date,
        )
        procedural_task = (
            self.procedural.search_by_query(
                pg_session, user_id, query, app_id=app_id, top_k=self.top_k_procedures
            )
            if self.procedural is not None
            else _empty()
        )

        similar, profile, recent, procedures_raw = await asyncio.gather(
            similar_task, profile_task, recent_task, procedural_task,
            return_exceptions=True,
        )

        # ── 4. Handle partial failures gracefully ─────────────────────────
        similar_events = _safe_result(similar, [], "hybrid_search", user_id)
        user_profile   = _safe_result(profile, None, "get_user_profile", user_id)
        recent_events  = _safe_result(recent,  [], "get_recent", user_id)
        procedures     = _safe_result(procedures_raw, [], "search_by_query", user_id)

        # Track recall so the frequency signal stays current
        if similar_events:
            await self.episodic.increment_recall(
                pg_session, [sr.event.id for sr in similar_events]
            )

        # Track procedure hits
        if procedures and self.procedural is not None:
            await self.procedural.increment_hit_count(
                pg_session, [p.id for p in procedures]
            )

        # Deduplicate: don't show the same event in both similar and recent
        similar_ids = {sr.event.id for sr in similar_events}
        recent_events = [e for e in recent_events if e.id not in similar_ids]

        # ── 5. Narrative chain traversal (opt-in) ────────────────────────
        narrative_chains: list[list[Event]] = []
        if self.narrative and self.include_chains and similar_events:
            try:
                anchor_id = similar_events[0].event.id
                chain_links = await self.narrative.get_chain_forward(pg_session, anchor_id)
                if chain_links:
                    linked_ids = [lnk.to_event_id for lnk in chain_links]
                    result = await pg_session.execute(
                        select(Event).where(Event.id.in_(linked_ids))
                    )
                    events_by_id = {e.id: e for e in result.scalars().all()}
                    chain_events = [similar_events[0].event] + [
                        events_by_id[lnk.to_event_id]
                        for lnk in chain_links
                        if lnk.to_event_id in events_by_id
                    ]
                    if len(chain_events) > 1:
                        narrative_chains = [chain_events]
            except Exception as exc:
                logger.warning(
                    "Narrative chain traversal failed",
                    extra={"user_id": user_id, "error": str(exc)},
                )

        ctx = MemoryContext(
            user_id=user_id,
            query=query,
            similar_events=similar_events,
            user_profile=user_profile,
            recent_events=recent_events,
            embedding_failed=embedding_failed,
            intent=str(detected_intent),
            narrative_chains=narrative_chains,
            procedures=procedures,
        )

        logger.info(
            "MemoryContext assembled",
            extra={
                "user_id": user_id,
                "intent": str(detected_intent),
                "similar": len(similar_events),
                "facts": len(user_profile.facts) if user_profile else 0,
                "recent": len(recent_events),
                "procedures": len(procedures),
                "total": ctx.total_memories(),
            },
        )
        return ctx

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _embed_query(
        self, query: str, user_id: str
    ) -> tuple[list[float] | None, bool]:
        """Embed the query. Returns (embedding, failed_flag)."""
        try:
            embedding = await self.llm.embed(query)
            return embedding, False
        except Exception as exc:
            logger.warning(
                "Query embedding failed — context assembled without vector recall",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return None, True


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _empty() -> list:
    """No-op coroutine that returns an empty list (used when embedding is unavailable)."""
    return []


def _safe_result(result, fallback, operation: str, user_id: str):
    """Return result if not an exception, else log and return fallback."""
    if isinstance(result, Exception):
        logger.warning(
            f"ContextBuilder: {operation} failed",
            extra={"user_id": user_id, "error": str(result)},
        )
        return fallback
    return result


def _format_date(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"
