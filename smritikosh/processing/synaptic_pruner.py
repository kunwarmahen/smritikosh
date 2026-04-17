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
  - Thresholds are adaptive: a high-volume user (>1000 events) gets a
    tighter prune than a light user (<50 events). The instance defaults
    act as the baseline for the "normal" volume tier.
  - Returns a PruningResult so the Scheduler can log what was removed.

Run: after each Consolidation cycle via the Scheduler.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_IMPORTANCE_THRESHOLD = 0.2   # prune only if importance is below this
DEFAULT_MIN_RECALL_COUNT = 2         # prune only if recalled fewer times than this
DEFAULT_MIN_AGE_DAYS = 90            # never prune events younger than this

# Volume tiers that trigger threshold adjustment
HIGH_VOLUME_EVENT_COUNT = 1000       # tighten thresholds above this
LOW_VOLUME_EVENT_COUNT = 50          # loosen thresholds below this


# ── Threshold type ────────────────────────────────────────────────────────────

@dataclass
class PruningThresholds:
    importance_threshold: float
    min_recall_count: int
    min_age_days: int


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class PruningResult:
    user_id: str
    app_id: str
    events_evaluated: int = 0
    events_pruned: int = 0
    facts_purged: int = 0
    skipped: bool = False
    thresholds: PruningThresholds | None = field(default=None)


# ── Adaptive threshold computation ───────────────────────────────────────────

def compute_adaptive_thresholds(
    event_count: int,
    base_importance: float = DEFAULT_IMPORTANCE_THRESHOLD,
    base_min_recall: int = DEFAULT_MIN_RECALL_COUNT,
    base_min_age: int = DEFAULT_MIN_AGE_DAYS,
) -> PruningThresholds:
    """
    Return pruning thresholds scaled to the user's event volume.

    High-volume users accumulate memories faster, so the prune bar is raised
    (more aggressive).  Low-volume users have sparse memory, so the bar is
    lowered (more conservative).

    Tier         | event_count     | effect
    -------------|-----------------|----------------------------------------------
    High volume  | > 1000          | importance +50%, age −33%  (tighten)
    Normal       | 50 – 1000       | base values unchanged
    Low volume   | < 50            | importance −25%, age +100% (loosen)
    """
    if event_count > HIGH_VOLUME_EVENT_COUNT:
        return PruningThresholds(
            importance_threshold=round(base_importance * 1.5, 4),
            min_recall_count=base_min_recall,
            min_age_days=max(1, round(base_min_age * 0.67)),
        )
    if event_count < LOW_VOLUME_EVENT_COUNT:
        return PruningThresholds(
            importance_threshold=round(base_importance * 0.75, 4),
            min_recall_count=base_min_recall,
            min_age_days=round(base_min_age * 2.0),
        )
    return PruningThresholds(
        importance_threshold=base_importance,
        min_recall_count=base_min_recall,
        min_age_days=base_min_age,
    )


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
        semantic=None,  # SemanticMemory | None — enables fact GC after pruning
        audit=None,     # AuditLogger | None
    ) -> None:
        self.episodic = episodic
        self.importance_threshold = importance_threshold
        self.min_recall_count = min_recall_count
        self.min_age_days = min_age_days
        self.semantic = semantic
        self.audit = audit

    async def prune(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str = "default",
        neo_session=None,   # neo4j.AsyncSession | None — for fact GC
        override_thresholds: "PruningThresholds | None" = None,
    ) -> PruningResult:
        """
        Evaluate and delete consolidated events that meet all prune conditions.

        Computes adaptive thresholds based on the user's total event count
        before loading any candidates, so the SQL pre-filter already uses the
        scaled values.

        Pass override_thresholds to bypass adaptive computation — useful for
        testing (e.g. min_age_days=0 to prune fresh events immediately).
        """
        result = PruningResult(user_id=user_id, app_id=app_id)

        # Compute adaptive thresholds (or use caller-supplied overrides)
        event_count = await self._count_user_events(session, user_id, app_id)
        if override_thresholds is not None:
            thresholds = override_thresholds
        else:
            thresholds = compute_adaptive_thresholds(
                event_count,
                base_importance=self.importance_threshold,
                base_min_recall=self.min_recall_count,
                base_min_age=self.min_age_days,
            )
        result.thresholds = thresholds

        if event_count != DEFAULT_MIN_RECALL_COUNT:  # log only when tier changed
            logger.debug(
                "Adaptive pruning thresholds",
                extra={
                    "user_id": user_id,
                    "event_count": event_count,
                    "importance_threshold": thresholds.importance_threshold,
                    "min_age_days": thresholds.min_age_days,
                },
            )

        candidates = await self._get_prune_candidates(session, user_id, app_id, thresholds)
        result.events_evaluated = len(candidates)

        if not candidates:
            result.skipped = True
            if self.audit:
                from smritikosh.audit.logger import AuditEvent, EventType
                await self.audit.emit(AuditEvent(
                    event_type=EventType.MEMORY_PRUNE_RUN,
                    user_id=user_id,
                    app_id=app_id,
                    payload={
                        "events_evaluated": 0,
                        "events_pruned": 0,
                        "facts_purged": 0,
                        "skipped": True,
                        "skip_reason": "no candidates — events too young or too important",
                        "importance_threshold": thresholds.importance_threshold,
                        "min_age_days": thresholds.min_age_days,
                        "min_recall_count": thresholds.min_recall_count,
                    },
                ))
            return result

        now = datetime.now(timezone.utc)
        pruned = 0

        for event in candidates:
            if self._should_prune(event, now, thresholds):
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
                                "importance_threshold": thresholds.importance_threshold,
                                "min_recall_count": thresholds.min_recall_count,
                                "min_age_days": thresholds.min_age_days,
                                "event_count": event_count,
                                "raw_text_preview": (event.raw_text or "")[:200],
                            },
                        ))

        result.events_pruned = pruned

        # ── Fact garbage collection ────────────────────────────────────────
        # If any events were pruned and we have a SemanticMemory + neo4j
        # session, remove facts that were last reinforced within the same
        # age window as the deleted events — they are orphaned in practice.
        if pruned and self.semantic and neo_session is not None:
            try:
                facts_purged = await self.semantic.purge_unseen_facts(
                    neo_session,
                    user_id=user_id,
                    not_seen_since_days=thresholds.min_age_days,
                )
                result.facts_purged = facts_purged
                logger.debug(
                    "Fact GC complete",
                    extra={"user_id": user_id, "facts_purged": facts_purged},
                )
            except Exception as exc:
                logger.warning(
                    "Fact GC failed — skipping: %s",
                    exc,
                    extra={"user_id": user_id},
                )

        logger.info(
            "Pruning complete",
            extra={
                "user_id": user_id,
                "evaluated": result.events_evaluated,
                "pruned": pruned,
                "facts_purged": result.facts_purged,
            },
        )

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.MEMORY_PRUNE_RUN,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "events_evaluated": result.events_evaluated,
                    "events_pruned": pruned,
                    "facts_purged": result.facts_purged,
                    "skipped": result.skipped,
                    "importance_threshold": thresholds.importance_threshold,
                    "min_age_days": thresholds.min_age_days,
                    "min_recall_count": thresholds.min_recall_count,
                },
            ))

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _count_user_events(
        self, session: AsyncSession, user_id: str, app_id: str
    ) -> int:
        """Count total consolidated events for this user to determine volume tier."""
        row = await session.execute(
            select(func.count()).where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.consolidated.is_(True),
            )
        )
        return row.scalar_one() or 0

    async def _get_prune_candidates(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str,
        thresholds: PruningThresholds,
    ) -> list[Event]:
        """
        Fetch consolidated events that pass all three pre-filter conditions.
        SQL-level filtering keeps the Python loop small.
        Uses the adaptive thresholds computed for this user.
        """
        cutoff_sql = text(f"NOW() - INTERVAL '{thresholds.min_age_days} days'")
        result = await session.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.consolidated.is_(True),
                Event.created_at < cutoff_sql,
                Event.importance_score < thresholds.importance_threshold,
                Event.recall_count < thresholds.min_recall_count,
            )
            .order_by(Event.importance_score.asc())  # lowest importance first
        )
        return list(result.scalars().all())

    def _should_prune(self, event: Event, now: datetime, thresholds: PruningThresholds) -> bool:
        """
        Return True only if ALL three conditions are met:
          - importance_score below threshold
          - recall_count below minimum
          - age exceeds minimum days
        Uses the adaptive thresholds computed for this user.
        """
        created = event.created_at
        if created is None:
            return False
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = (now - created).total_seconds() / 86400.0
        return (
            (event.importance_score or 0.0) < thresholds.importance_threshold
            and (event.recall_count or 0) < thresholds.min_recall_count
            and age_days > thresholds.min_age_days
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
