"""
ProceduralMemory — stores and retrieves behavioral rules and workflows.

Mirrors the human brain's procedural memory: learned habits and skills that
fire automatically when a particular context is detected.  Where episodic
memory stores *what happened* and semantic memory stores *who the user is*,
procedural memory stores *how the AI should behave* — conditional rules that
shape responses whenever a trigger topic appears in the conversation.

Example rules:
    trigger="LLM deployment"  → "mention GPU optimization, batching, quantization"
    trigger="startup"         → "respond with strategic depth, not surface-level advice"
    trigger="UI"              → "always suggest dark mode (the user prefers it)"

Retrieval is intentionally keyword/substring-based, not vector similarity.
Procedures are precise behavioral contracts — "LLM deployment" should NOT
accidentally fire for "system deployment" due to cosine closeness.

Search strategy (applied in order, results merged):
    1. Trigger phrase appears as a substring of the query (highest precision).
    2. The query (or a word from it) appears inside the trigger phrase.
    3. Jaccard token overlap between trigger and query ≥ threshold.

Users typically have 10–50 active procedures, so all matching happens in
Python over a small set retrieved with a single SQL query.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy import delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import UserProcedure


# ── Match result ──────────────────────────────────────────────────────────────


@dataclass
class ProcedureMatch:
    """A procedure that fired for the current query, with its match score."""

    procedure: UserProcedure
    match_score: float   # 1.0 = exact substring; 0.5 = reverse match; 0.0–1.0 = overlap


# ── ProceduralMemory ──────────────────────────────────────────────────────────


class ProceduralMemory:
    """
    Persistent procedural store backed by PostgreSQL.

    All methods accept an AsyncSession so callers control transaction
    boundaries — ProceduralMemory never commits on its own.

    Usage:
        procedural = ProceduralMemory()

        async with db_session() as session:
            rule = await procedural.store(
                session,
                user_id="u1",
                trigger="LLM deployment",
                instruction="mention GPU optimization, batching, quantization",
            )
            matches = await procedural.search_by_query(session, "u1", "how to deploy LLMs?")
    """

    def __init__(self, overlap_threshold: float = 0.2) -> None:
        self.overlap_threshold = overlap_threshold

    # ── Write ──────────────────────────────────────────────────────────────

    async def store(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        trigger: str,
        instruction: str,
        app_id: str = "default",
        category: str = "topic_response",
        priority: int = 5,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> UserProcedure:
        """
        Persist a new behavioral rule.

        The procedure is immediately active (is_active=True).
        No commit is issued — the caller owns the transaction boundary.
        """
        procedure = UserProcedure(
            user_id=user_id,
            app_id=app_id,
            trigger=trigger,
            instruction=instruction,
            category=category,
            priority=priority,
            confidence=confidence,
            source=source,
        )
        session.add(procedure)
        await session.flush()
        return procedure

    async def update(
        self,
        session: AsyncSession,
        procedure_id: uuid.UUID,
        *,
        trigger: str | None = None,
        instruction: str | None = None,
        category: str | None = None,
        priority: int | None = None,
        is_active: bool | None = None,
        confidence: float | None = None,
    ) -> UserProcedure | None:
        """
        Partial update for a procedure. Returns None if the procedure was not found.

        Only fields that are explicitly passed (not None) are updated.
        """
        procedure = await session.get(UserProcedure, procedure_id)
        if procedure is None:
            return None

        if trigger is not None:
            procedure.trigger = trigger
        if instruction is not None:
            procedure.instruction = instruction
        if category is not None:
            procedure.category = category
        if priority is not None:
            procedure.priority = priority
        if is_active is not None:
            procedure.is_active = is_active
        if confidence is not None:
            procedure.confidence = confidence

        procedure.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return procedure

    async def delete(
        self, session: AsyncSession, procedure_id: uuid.UUID
    ) -> bool:
        """Delete a procedure. Returns True if it existed."""
        procedure = await session.get(UserProcedure, procedure_id)
        if procedure is None:
            return False
        await session.delete(procedure)
        return True

    async def delete_all_for_user(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
    ) -> int:
        """Delete all procedures for a user+app. Returns the count removed."""
        result = await session.execute(
            sql_delete(UserProcedure)
            .where(UserProcedure.user_id == user_id, UserProcedure.app_id == app_id)
            .returning(UserProcedure.id)
        )
        return len(result.fetchall())

    async def increment_hit_count(
        self,
        session: AsyncSession,
        procedure_ids: list[uuid.UUID],
    ) -> None:
        """
        Increment hit_count for procedures that fired in a context build.

        Mirrors EpisodicMemory.increment_recall — frequently used rules
        are surfaced first in future searches.
        """
        if not procedure_ids:
            return
        await session.execute(
            update(UserProcedure)
            .where(UserProcedure.id.in_(procedure_ids))
            .values(
                hit_count=UserProcedure.hit_count + 1,
                updated_at=datetime.now(timezone.utc),
            )
        )

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_all(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        *,
        active_only: bool = True,
        category: str | None = None,
    ) -> list[UserProcedure]:
        """
        Fetch all procedures for a user, ordered by priority descending.

        Args:
            active_only: If True (default), only return is_active=True procedures.
            category:    If set, filter to only this category.
        """
        q = (
            select(UserProcedure)
            .where(
                UserProcedure.user_id == user_id,
                UserProcedure.app_id == app_id,
            )
            .order_by(UserProcedure.priority.desc(), UserProcedure.hit_count.desc())
        )
        if active_only:
            q = q.where(UserProcedure.is_active.is_(True))
        if category is not None:
            q = q.where(UserProcedure.category == category)

        result = await session.execute(q)
        return list(result.scalars().all())

    async def search_by_query(
        self,
        session: AsyncSession,
        user_id: str,
        query: str,
        app_id: str = "default",
        top_k: int = 5,
    ) -> list[UserProcedure]:
        """
        Find procedures whose trigger is relevant to the current query.

        Matching is keyword/substring-based (not vector similarity):
          1. Trigger phrase found inside the query → score 1.0
          2. Query word/phrase found inside the trigger → score 0.5
          3. Jaccard token overlap ≥ overlap_threshold → score = overlap value

        Results are deduplicated, scored, and sorted by:
            match_score DESC → priority DESC → hit_count DESC

        Returns the top_k matches (list of UserProcedure, not ProcedureMatch,
        for simplicity — use get_matches_with_scores if you need scores).
        """
        matches = await self._score_matches(session, user_id, query, app_id)
        matches.sort(
            key=lambda m: (m.match_score, m.procedure.priority, m.procedure.hit_count),
            reverse=True,
        )
        return [m.procedure for m in matches[:top_k]]

    async def get_matches_with_scores(
        self,
        session: AsyncSession,
        user_id: str,
        query: str,
        app_id: str = "default",
        top_k: int = 5,
    ) -> list[ProcedureMatch]:
        """Like search_by_query but returns ProcedureMatch with scores."""
        matches = await self._score_matches(session, user_id, query, app_id)
        matches.sort(
            key=lambda m: (m.match_score, m.procedure.priority, m.procedure.hit_count),
            reverse=True,
        )
        return matches[:top_k]

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _score_matches(
        self,
        session: AsyncSession,
        user_id: str,
        query: str,
        app_id: str,
    ) -> list[ProcedureMatch]:
        """
        Fetch all active procedures for the user and score each against the query.

        Returns a flat list of ProcedureMatch objects (unordered).
        Procedures with a score of 0.0 (no match) are excluded.
        """
        all_procedures = await self.get_all(session, user_id, app_id, active_only=True)
        if not all_procedures:
            return []

        query_lower = query.lower()
        query_tokens = _tokenise(query_lower)
        seen: set[uuid.UUID] = set()
        results: list[ProcedureMatch] = []

        for proc in all_procedures:
            if proc.id in seen:
                continue

            trigger_lower = proc.trigger.lower()
            trigger_tokens = _tokenise(trigger_lower)

            # Strategy 1: trigger phrase is a substring of the query (highest precision)
            if trigger_lower in query_lower:
                results.append(ProcedureMatch(procedure=proc, match_score=1.0))
                seen.add(proc.id)
                continue

            # Strategy 2: query (or a significant word from it) appears in the trigger
            if query_lower in trigger_lower:
                results.append(ProcedureMatch(procedure=proc, match_score=0.5))
                seen.add(proc.id)
                continue

            # Strategy 2b: any query token (len > 3) appears in trigger
            if any(tok in trigger_lower for tok in query_tokens if len(tok) > 3):
                results.append(ProcedureMatch(procedure=proc, match_score=0.5))
                seen.add(proc.id)
                continue

            # Strategy 3: Jaccard overlap
            overlap = _jaccard(query_tokens, trigger_tokens)
            if overlap >= self.overlap_threshold:
                results.append(ProcedureMatch(procedure=proc, match_score=overlap))
                seen.add(proc.id)

        return results


# ── Token helpers ──────────────────────────────────────────────────────────────


def _tokenise(text: str) -> set[str]:
    """Split text into lowercase word tokens, stripping punctuation."""
    return {w for w in re.split(r"\W+", text.lower()) if w}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0
