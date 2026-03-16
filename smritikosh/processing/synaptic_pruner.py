"""
SynapticPruner — removes low-value memories to prevent unbounded growth.

Mirrors the brain's synaptic pruning: neural connections that are rarely
used and carry little signal are weakened and eventually eliminated,
keeping the memory system efficient.

Prune condition (all three must be true):
    importance_score < importance_threshold   (low signal value)
    recall_count     < min_recall_count       (never or rarely retrieved)
    age              > min_age_days           (old enough to safely drop)

Design decisions:
  - Only consolidated events are pruned — raw unconsolidated events are
    preserved until the Consolidator has processed them.
  - The conjunction of all three conditions is conservative: a memory that
    has been recalled even once is kept regardless of age or importance.
  - Returns a PruningResult so the Scheduler can log what was removed.

Run: after each Consolidation cycle via the Scheduler.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_IMPORTANCE_THRESHOLD = 0.2   # prune only if importance is below this
DEFAULT_MIN_RECALL_COUNT = 2         # prune only if recalled fewer times than this
DEFAULT_MIN_AGE_DAYS = 90            # never prune events younger than this


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PruningResult:
    user_id: str
    app_id: str
    events_evaluated: int = 0
    events_pruned: int = 0
    skipped: bool = False


# ── SynapticPruner ────────────────────────────────────────────────────────────

class SynapticPruner:
    """
    Deletes consolidated events that meet all three prune conditions:
    low importance, low recall frequency, and sufficient age.

    Usage:
        pruner = SynapticPruner(episodic=episodic)

        async with db_session() as session:
            result = await pruner.prune(session, user_id="u1")
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
        min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
        min_age_days: int = DEFAULT_MIN_AGE_DAYS,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.episodic = episodic
        self.importance_threshold = importance_threshold
        self.min_recall_count = min_recall_count
        self.min_age_days = min_age_days
        self.audit = audit

    async def prune(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> PruningResult:
        """
        Evaluate and delete consolidated events that meet all prune conditions.

        Pre-filters in SQL (importance + recall_count + age) so that only
        true candidates are loaded into Python.
        """
        result = PruningResult(user_id=user_id, app_id=app_id)

        candidates = await self._get_prune_candidates(session, user_id, app_id)
        result.events_evaluated = len(candidates)

        if not candidates:
            result.skipped = True
            return result

        now = datetime.now(timezone.utc)
        pruned = 0

        for event in candidates:
            if self._should_prune(event, now):
                deleted = await self.episodic.delete(session, event.id)
                if deleted:
                    pruned += 1
                    logger.debug(
                        "Pruned low-value event",
                        extra={
                            "event_id": str(event.id),
                            "user_id": user_id,
                            "importance": round(event.importance_score or 0.0, 3),
                            "recall_count": event.recall_count or 0,
                        },
                    )
                    if self.audit:
                        from smritikosh.audit.logger import AuditEvent, EventType
                        created = event.created_at
                        if created and created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        age_days = round((now - created).total_seconds() / 86400.0, 1) if created else None
                        await self.audit.emit(AuditEvent(
                            event_type=EventType.MEMORY_PRUNED,
                            user_id=user_id,
                            app_id=app_id,
                            event_id=str(event.id),
                            payload={
                                "importance_score": event.importance_score,
                                "recall_count": event.recall_count or 0,
                                "age_days": age_days,
                                "importance_threshold": self.importance_threshold,
                                "min_recall_count": self.min_recall_count,
                                "min_age_days": self.min_age_days,
                                "raw_text_preview": (event.raw_text or "")[:200],
                            },
                        ))

        result.events_pruned = pruned
        logger.info(
            "Pruning complete",
            extra={
                "user_id": user_id,
                "evaluated": result.events_evaluated,
                "pruned": pruned,
            },
        )
        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _get_prune_candidates(
        self, session: AsyncSession, user_id: str, app_id: str
    ) -> list[Event]:
        """
        Fetch consolidated events that pass all three pre-filter conditions.
        SQL-level filtering keeps the Python loop small.
        """
        cutoff_sql = text(f"NOW() - INTERVAL '{self.min_age_days} days'")
        result = await session.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.consolidated.is_(True),
                Event.created_at < cutoff_sql,
                Event.importance_score < self.importance_threshold,
                Event.recall_count < self.min_recall_count,
            )
            .order_by(Event.importance_score.asc())  # lowest importance first
        )
        return list(result.scalars().all())

    def _should_prune(self, event: Event, now: datetime) -> bool:
        """
        Return True only if ALL three conditions are met:
          - importance_score below threshold
          - recall_count below minimum
          - age exceeds minimum days
        """
        created = event.created_at
        if created is None:
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).total_seconds() / 86400.0
        return (
            (event.importance_score or 0.0) < self.importance_threshold
            and (event.recall_count or 0) < self.min_recall_count
            and age_days > self.min_age_days
        )


# ── Module-level decision helper (for tests and CLI) ─────────────────────────

def compute_prune_decision(
    importance: float,
    recall_count: int,
    age_days: float,
    importance_threshold: float = DEFAULT_IMPORTANCE_THRESHOLD,
    min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
    min_age_days: int = DEFAULT_MIN_AGE_DAYS,
) -> bool:
    """Stateless prune decision — useful for previewing what would be pruned."""
    return (
        importance < importance_threshold
        and recall_count < min_recall_count
        and age_days > min_age_days
    )
