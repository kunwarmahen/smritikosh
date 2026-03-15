"""
EpisodicMemory — stores and retrieves experiences from PostgreSQL + pgvector.

Mirrors the Hippocampus/episodic memory function in the human brain:
  - Every interaction is recorded as a timestamped event with an embedding.
  - Recall is by similarity (like human associative recall), not keyword search.
  - Hybrid scoring combines semantic similarity + recency + importance + frequency
    so that recent, important, frequently-recalled, and contextually relevant
    memories surface first.

Hybrid search formula:
    score = similarity_weight  * cosine_similarity(query, event.embedding)
          + recency_weight     * exp(-days_since_event / decay_days)
          + importance_weight  * event.importance_score
          + frequency_weight   * min(recall_count, freq_cap) / freq_cap

Weights (similarity + recency + importance + frequency + contextual_match)
must sum to 1.0 and are tunable via HybridWeights.
contextual_match is reserved for Phase 2 intent-aware retrieval (defaults to 0.0).
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event


@dataclass
class SearchResult:
    """An event returned from hybrid search, augmented with its score breakdown."""
    event: Event
    similarity_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass
class HybridWeights:
    """
    Tunable weights for the hybrid search scoring formula.

    All five weights must sum to 1.0.
    contextual_match defaults to 0.0 and is activated in Phase 2 (intent classification).
    """
    similarity: float = 0.40        # semantic closeness to the query
    recency: float = 0.30           # exponential decay based on event age
    importance: float = 0.15        # Amygdala-assigned importance score
    frequency: float = 0.15         # normalised recall_count — how often retrieved
    contextual_match: float = 0.0   # reserved: intent-aware boost (Phase 2)
    decay_days: float = 30.0        # half-life for recency decay
    frequency_cap: int = 50         # recall_count normalisation ceiling

    def __post_init__(self) -> None:
        total = (
            self.similarity
            + self.recency
            + self.importance
            + self.frequency
            + self.contextual_match
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"HybridWeights must sum to 1.0, got {total:.3f}. "
                f"similarity={self.similarity}, recency={self.recency}, "
                f"importance={self.importance}, frequency={self.frequency}, "
                f"contextual_match={self.contextual_match}"
            )


class EpisodicMemory:
    """
    Persistent episodic store backed by PostgreSQL + pgvector.

    All methods accept an AsyncSession so callers (Hippocampus, API routes,
    background jobs) control transaction boundaries — EpisodicMemory never
    commits on its own.

    Usage:
        episodic = EpisodicMemory()

        async with db_session() as session:
            event = await episodic.store(session, user_id="u1", raw_text="...")
            results = await episodic.hybrid_search(session, "u1", query_vec)
    """

    def __init__(self, weights: HybridWeights | None = None) -> None:
        self.weights = weights or HybridWeights()

    # ── Write ──────────────────────────────────────────────────────────────

    async def store(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        raw_text: str,
        app_id: str = "default",
        embedding: list[float] | None = None,
        importance_score: float = 1.0,
        metadata: dict | None = None,
    ) -> Event:
        """
        Persist a new episodic event.

        The caller is responsible for generating the embedding before calling
        this method (via LLMAdapter.embed). Storing without an embedding is
        allowed — the event won't appear in vector searches until one is added.
        """
        event = Event(
            user_id=user_id,
            app_id=app_id,
            raw_text=raw_text,
            embedding=embedding,
            importance_score=importance_score,
            consolidated=False,
            event_metadata=metadata or {},
        )
        session.add(event)
        await session.flush()   # get the auto-generated id without committing
        return event

    async def update_embedding(
        self,
        session: AsyncSession,
        event_id: uuid.UUID,
        embedding: list[float],
    ) -> None:
        """Attach an embedding to an already-stored event."""
        await session.execute(
            update(Event)
            .where(Event.id == event_id)
            .values(embedding=embedding, updated_at=datetime.now(timezone.utc))
        )

    async def mark_consolidated(
        self,
        session: AsyncSession,
        event_ids: list[uuid.UUID],
        summary: str | None = None,
    ) -> None:
        """
        Flag events as consolidated after the Consolidator has processed them.
        Optionally attach the generated summary.
        """
        values: dict = {"consolidated": True, "updated_at": datetime.now(timezone.utc)}
        if summary is not None:
            values["summary"] = summary

        await session.execute(
            update(Event).where(Event.id.in_(event_ids)).values(**values)
        )

    async def delete(self, session: AsyncSession, event_id: uuid.UUID) -> bool:
        """Delete an event. Returns True if it existed."""
        event = await session.get(Event, event_id)
        if event is None:
            return False
        await session.delete(event)
        return True

    async def increment_recall(
        self,
        session: AsyncSession,
        event_ids: list[uuid.UUID],
    ) -> None:
        """
        Increment recall_count for events surfaced by a search.

        Called by ContextBuilder after every hybrid_search so that
        frequently-retrieved memories get a higher frequency_score
        in future searches.
        """
        if not event_ids:
            return
        await session.execute(
            update(Event)
            .where(Event.id.in_(event_ids))
            .values(
                recall_count=Event.recall_count + 1,
                updated_at=datetime.now(timezone.utc),
            )
        )

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_recent(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        limit: int = 10,
        include_consolidated: bool = True,
    ) -> list[Event]:
        """Return the most recent events for a user, newest first."""
        q = (
            select(Event)
            .where(Event.user_id == user_id, Event.app_id == app_id)
            .order_by(Event.created_at.desc())
            .limit(limit)
        )
        if not include_consolidated:
            q = q.where(Event.consolidated.is_(False))

        result = await session.execute(q)
        return list(result.scalars().all())

    async def get_unconsolidated(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        limit: int = 100,
    ) -> list[Event]:
        """Return events not yet processed by the Consolidator (oldest first)."""
        result = await session.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.consolidated.is_(False),
            )
            .order_by(Event.created_at.asc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def search_similar(
        self,
        session: AsyncSession,
        user_id: str,
        query_embedding: list[float],
        app_id: str = "default",
        top_k: int = 5,
    ) -> list[Event]:
        """
        Pure vector similarity search using pgvector cosine distance.
        Returns events ordered by closeness to the query embedding.
        Use hybrid_search for most retrieval tasks — this is exposed for
        cases where you want raw similarity ranking only.
        """
        vec_literal = _embedding_literal(query_embedding)
        result = await session.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.embedding.is_not(None),
            )
            .order_by(text(f"embedding <=> '{vec_literal}'"))
            .limit(top_k)
        )
        return list(result.scalars().all())

    async def hybrid_search(
        self,
        session: AsyncSession,
        user_id: str,
        query_embedding: list[float],
        app_id: str = "default",
        top_k: int = 5,
        weights_override: "HybridWeights | None" = None,
    ) -> list[SearchResult]:
        """
        Hybrid retrieval: semantic similarity + recency decay + importance score.

        This is the primary retrieval method used by the ContextBuilder.
        Returns SearchResult objects so the caller can inspect score breakdown.

        SQL formula (inline for pgvector compatibility):
            hybrid_score =
                similarity_weight  * (1 - cosine_distance)
              + recency_weight     * exp(-age_in_days / decay_days)
              + importance_weight  * importance_score
              + frequency_weight   * min(recall_count, freq_cap) / freq_cap
        """
        w = weights_override if weights_override is not None else self.weights
        vec_literal = _embedding_literal(query_embedding)

        sql = text(f"""
            SELECT
                id,
                (1.0 - (embedding <=> '{vec_literal}'))                     AS similarity_score,
                EXP(
                    -EXTRACT(EPOCH FROM (NOW() - created_at))
                    / 86400.0 / :decay_days
                )                                                            AS recency_score,
                importance_score,
                LEAST(recall_count, :freq_cap)::float / :freq_cap           AS frequency_score
            FROM events
            WHERE
                user_id = :user_id
                AND app_id = :app_id
                AND embedding IS NOT NULL
            ORDER BY
                (
                    :w_sim  * (1.0 - (embedding <=> '{vec_literal}'))
                  + :w_rec  * EXP(-EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 / :decay_days)
                  + :w_imp  * importance_score
                  + :w_freq * (LEAST(recall_count, :freq_cap)::float / :freq_cap)
                ) DESC
            LIMIT :top_k
        """)

        rows = await session.execute(
            sql,
            {
                "user_id": user_id,
                "app_id": app_id,
                "decay_days": w.decay_days,
                "w_sim": w.similarity,
                "w_rec": w.recency,
                "w_imp": w.importance,
                "w_freq": w.frequency,
                "freq_cap": w.frequency_cap,
                "top_k": top_k,
            },
        )

        results: list[SearchResult] = []
        for row in rows:
            event = await session.get(Event, row.id)
            if event is not None:
                results.append(
                    SearchResult(
                        event=event,
                        similarity_score=float(row.similarity_score),
                        recency_score=float(row.recency_score),
                        frequency_score=float(row.frequency_score),
                        hybrid_score=(
                            w.similarity * float(row.similarity_score)
                            + w.recency * float(row.recency_score)
                            + w.importance * float(row.importance_score)
                            + w.frequency * float(row.frequency_score)
                        ),
                    )
                )
        return results


# ── Helpers ───────────────────────────────────────────────────────────────────


def _embedding_literal(embedding: list[float]) -> str:
    """Format a float list as a pgvector literal: [0.1,0.2,...]"""
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
