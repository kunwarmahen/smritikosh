"""
SynapticPruner — removes low-value memories to prevent unbounded growth.

Mirrors the brain's synaptic pruning: neural connections that are rarely
used and carry little signal are weakened and eventually eliminated,
keeping the memory system efficient.

Score formula (per event):
    prune_score = importance_score × exp(-age_days / decay_days)

Events below the prune_threshold AND older than min_age_days are deleted.

Design decisions:
  - Only consolidated events are pruned — raw unconsolidated events are
    preserved until the Consolidator has processed them.
  - Events younger than min_age_days are always kept, regardless of score,
    to prevent deleting recent context that hasn't been consolidated yet.
  - Returns a PruningResult so the Scheduler can log what was removed.

Run: after each Consolidation cycle via the Scheduler.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PRUNE_THRESHOLD = 0.15    # events scoring below this are deleted
DEFAULT_MIN_AGE_DAYS = 7          # never prune events younger than this
DEFAULT_DECAY_DAYS = 30.0         # recency decay half-life


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
    Deletes consolidated events whose prune_score falls below the threshold.

    Usage:
        pruner = SynapticPruner(episodic=episodic)

        async with db_session() as session:
            result = await pruner.prune(session, user_id="u1")
    """

    def __init__(
        self,
        episodic: EpisodicMemory,
        prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
        min_age_days: int = DEFAULT_MIN_AGE_DAYS,
        decay_days: float = DEFAULT_DECAY_DAYS,
    ) -> None:
        self.episodic = episodic
        self.prune_threshold = prune_threshold
        self.min_age_days = min_age_days
        self.decay_days = decay_days

    async def prune(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> PruningResult:
        """
        Evaluate and delete low-scoring consolidated events for a user.

        Only consolidated events older than min_age_days are eligible.
        Each is scored and deleted if below prune_threshold.
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
            score = self._prune_score(event, now)
            if score < self.prune_threshold:
                deleted = await self.episodic.delete(session, event.id)
                if deleted:
                    pruned += 1
                    logger.debug(
                        "Pruned low-score event",
                        extra={
                            "event_id": str(event.id),
                            "user_id": user_id,
                            "score": round(score, 3),
                        },
                    )

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
        Fetch consolidated events older than min_age_days.
        These are the only candidates for pruning.
        """
        cutoff_sql = text(
            f"NOW() - INTERVAL '{self.min_age_days} days'"
        )
        result = await session.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.consolidated.is_(True),
                Event.created_at < cutoff_sql,
            )
            .order_by(Event.importance_score.asc())   # lowest importance first
        )
        return list(result.scalars().all())

    def _prune_score(self, event: Event, now: datetime) -> float:
        """
        Compute the prune score for one event.
        Score = importance_score × recency_factor
        """
        importance = event.importance_score or 0.0
        created = event.created_at
        if created is None:
            return 0.0
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        age_days = (now - created).total_seconds() / 86400.0
        recency = math.exp(-age_days / self.decay_days)
        return importance * recency


# ── Module-level scoring helper (for tests and CLI) ──────────────────────────

def compute_prune_score(
    importance: float,
    age_days: float,
    decay_days: float = DEFAULT_DECAY_DAYS,
) -> float:
    """Stateless scoring function — useful for previewing what would be pruned."""
    return importance * math.exp(-age_days / decay_days)
