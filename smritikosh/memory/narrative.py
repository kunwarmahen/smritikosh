"""
NarrativeMemory — directed causal/temporal chains between episodic events.

Mirrors the brain's ability to encode narratives, not just isolated facts.
Humans remember stories: what caused what, what led to what.

This module owns all reads and writes to the memory_links table.
The MemoryLink table and RelationType enum are defined in db/models.py —
this class is the application layer on top of that schema.

Chain traversal uses a recursive CTE in PostgreSQL, so depth is bounded
by max_depth to prevent runaway queries on cyclic or very deep graphs.

Usage:
    narrative = NarrativeMemory()

    async with db_session() as session:
        # Write a link
        link = await narrative.create_link(
            session,
            from_event_id=event_a.id,
            to_event_id=event_b.id,
            relation_type=RelationType.CAUSED,
        )

        # Traverse forward from an anchor
        chain = await narrative.get_chain_forward(session, event_a.id)

        # Load all events related to an anchor (any direction)
        related = await narrative.get_related_events(session, event_a.id)
"""

import uuid
import logging

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event, MemoryLink, RelationType

logger = logging.getLogger(__name__)


class NarrativeMemory:
    """
    Reads and writes narrative memory links between episodic events.

    All methods accept an AsyncSession — callers own transaction boundaries.
    """

    # ── Write ──────────────────────────────────────────────────────────────

    async def create_link(
        self,
        session: AsyncSession,
        *,
        from_event_id: uuid.UUID,
        to_event_id: uuid.UUID,
        relation_type: RelationType,
    ) -> MemoryLink:
        """
        Create a directed link between two events.

        Does not guard against duplicates — callers should check for
        existing links if idempotency is needed.
        """
        link = MemoryLink(
            from_event_id=from_event_id,
            to_event_id=to_event_id,
            relation_type=str(relation_type),
        )
        session.add(link)
        await session.flush()  # assign id without committing
        return link

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_chain_forward(
        self,
        session: AsyncSession,
        event_id: uuid.UUID,
        max_depth: int = 5,
    ) -> list[MemoryLink]:
        """
        Traverse links forward from an anchor event (what happened next).

        Returns links in depth order: direct links first, deeper links after.
        Bounded by max_depth to prevent runaway queries.

        Example:
            A --[CAUSED]--> B --[PRECEDED]--> C
            get_chain_forward(A) → [A→B link, B→C link]
        """
        sql = text("""
            WITH RECURSIVE chain AS (
                SELECT id, to_event_id, 1 AS depth
                FROM memory_links
                WHERE from_event_id = :anchor_id

                UNION ALL

                SELECT ml.id, ml.to_event_id, chain.depth + 1
                FROM memory_links ml
                JOIN chain ON ml.from_event_id = chain.to_event_id
                WHERE chain.depth < :max_depth
            )
            SELECT id FROM chain ORDER BY depth ASC
        """)

        rows = (
            await session.execute(sql, {"anchor_id": event_id, "max_depth": max_depth})
        ).all()

        if not rows:
            return []

        link_ids = [row.id for row in rows]
        result = await session.execute(
            select(MemoryLink).where(MemoryLink.id.in_(link_ids))
        )
        link_map = {link.id: link for link in result.scalars().all()}
        # Return in original depth order
        return [link_map[lid] for lid in link_ids if lid in link_map]

    async def get_chain_backward(
        self,
        session: AsyncSession,
        event_id: uuid.UUID,
        max_depth: int = 5,
    ) -> list[MemoryLink]:
        """
        Traverse links backward from an anchor event (what caused this).

        Returns links in depth order: direct predecessors first.

        Example:
            A --[CAUSED]--> B --[PRECEDED]--> C
            get_chain_backward(C) → [B→C link, A→B link]
        """
        sql = text("""
            WITH RECURSIVE chain AS (
                SELECT id, from_event_id, 1 AS depth
                FROM memory_links
                WHERE to_event_id = :anchor_id

                UNION ALL

                SELECT ml.id, ml.from_event_id, chain.depth + 1
                FROM memory_links ml
                JOIN chain ON ml.to_event_id = chain.from_event_id
                WHERE chain.depth < :max_depth
            )
            SELECT id FROM chain ORDER BY depth ASC
        """)

        rows = (
            await session.execute(sql, {"anchor_id": event_id, "max_depth": max_depth})
        ).all()

        if not rows:
            return []

        link_ids = [row.id for row in rows]
        result = await session.execute(
            select(MemoryLink).where(MemoryLink.id.in_(link_ids))
        )
        link_map = {link.id: link for link in result.scalars().all()}
        return [link_map[lid] for lid in link_ids if lid in link_map]

    async def get_related_events(
        self,
        session: AsyncSession,
        event_id: uuid.UUID,
        user_id: str,
        app_id: str = "default",
    ) -> list[Event]:
        """
        Return all events directly linked to this event in either direction.

        Useful for surfacing context without committing to a traversal direction.
        Only returns events belonging to the same user+app (multi-tenant isolation).
        """
        result = await session.execute(
            select(Event)
            .join(
                MemoryLink,
                or_(
                    (MemoryLink.from_event_id == event_id) & (MemoryLink.to_event_id == Event.id),
                    (MemoryLink.to_event_id == event_id) & (MemoryLink.from_event_id == Event.id),
                ),
            )
            .where(Event.id != event_id, Event.user_id == user_id, Event.app_id == app_id)
            .distinct()
        )
        return list(result.scalars().all())
