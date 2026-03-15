"""
Scheduler — background job runner for consolidation and pruning.

Wires together Consolidator and SynapticPruner into periodic jobs using
APScheduler. Runs inside the same async event loop as the FastAPI app.

Jobs:
    consolidation_job  — runs every hour, processes all active users
    pruning_job        — runs daily after consolidation

Active users = users who have unconsolidated events in the last 24 hours.
This avoids scanning all users in the DB on every tick.

Integration with FastAPI:
    Call scheduler.start() inside the lifespan context to start background jobs.
    Call scheduler.shutdown() on app teardown.

Usage (standalone / testing):
    scheduler = MemoryScheduler(consolidator=..., pruner=..., ...)
    await scheduler.run_consolidation_now(user_id="u1")
"""

import logging
from datetime import timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select, text

from smritikosh.db.neo4j import neo4j_session
from smritikosh.db.postgres import db_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.processing.consolidator import Consolidator, ConsolidationResult
from smritikosh.processing.synaptic_pruner import PruningResult, SynapticPruner

logger = logging.getLogger(__name__)


class MemoryScheduler:
    """
    Manages periodic memory maintenance jobs.

    Both jobs can also be triggered manually (useful for testing or admin CLI).

    Args:
        consolidator:         Consolidator instance.
        pruner:               SynapticPruner instance.
        episodic:             EpisodicMemory — used to discover active users.
        consolidation_hours:  How often to run consolidation (default: 1).
        pruning_hours:        How often to run pruning (default: 24).
    """

    def __init__(
        self,
        consolidator: Consolidator,
        pruner: SynapticPruner,
        episodic: EpisodicMemory,
        consolidation_hours: int = 1,
        pruning_hours: int = 24,
    ) -> None:
        self.consolidator = consolidator
        self.pruner = pruner
        self.episodic = episodic
        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            self.run_consolidation_for_all_users,
            trigger="interval",
            hours=consolidation_hours,
            id="consolidation_job",
            name="Memory Consolidation",
            max_instances=1,       # never overlap
        )
        self._scheduler.add_job(
            self.run_pruning_for_all_users,
            trigger="interval",
            hours=pruning_hours,
            id="pruning_job",
            name="Synaptic Pruning",
            max_instances=1,
        )

    def start(self) -> None:
        self._scheduler.start()
        logger.info(
            "MemoryScheduler started",
            extra={
                "jobs": [j.id for j in self._scheduler.get_jobs()],
            },
        )

    def shutdown(self) -> None:
        self._scheduler.shutdown(wait=False)
        logger.info("MemoryScheduler stopped.")

    # ── Manual triggers (also used internally by scheduler) ───────────────

    async def run_consolidation_for_all_users(self) -> list[ConsolidationResult]:
        """
        Discover users with unconsolidated events and consolidate each.
        Called automatically by the scheduler or manually via admin endpoints.
        """
        user_app_pairs = await self._get_active_users()
        results: list[ConsolidationResult] = []

        for user_id, app_id in user_app_pairs:
            result = await self.run_consolidation_now(user_id=user_id, app_id=app_id)
            results.append(result)

        logger.info(
            "Batch consolidation complete",
            extra={"users_processed": len(results)},
        )
        return results

    async def run_pruning_for_all_users(self) -> list[PruningResult]:
        """Run pruning for all users that have consolidated events."""
        user_app_pairs = await self._get_all_users()
        results: list[PruningResult] = []

        for user_id, app_id in user_app_pairs:
            result = await self.run_pruning_now(user_id=user_id, app_id=app_id)
            results.append(result)

        logger.info(
            "Batch pruning complete",
            extra={"users_processed": len(results)},
        )
        return results

    async def run_consolidation_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> ConsolidationResult:
        """Run one consolidation cycle immediately for a specific user."""
        try:
            async with db_session() as pg, neo4j_session() as neo:
                return await self.consolidator.run(
                    pg, neo, user_id=user_id, app_id=app_id
                )
        except Exception as exc:
            logger.error(
                "Consolidation failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = ConsolidationResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = str(exc)
            return result

    async def run_pruning_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> PruningResult:
        """Run pruning immediately for a specific user."""
        try:
            async with db_session() as session:
                return await self.pruner.prune(
                    session, user_id=user_id, app_id=app_id
                )
        except Exception as exc:
            logger.error(
                "Pruning failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = PruningResult(user_id=user_id, app_id=app_id, skipped=True)
            return result

    # ── User discovery ─────────────────────────────────────────────────────

    async def _get_active_users(self) -> list[tuple[str, str]]:
        """
        Return (user_id, app_id) pairs that have unconsolidated events.
        'Active' = had at least one event in the last 24 hours.
        """
        from smritikosh.db.models import Event
        async with db_session() as session:
            result = await session.execute(
                select(Event.user_id, Event.app_id)
                .where(
                    Event.consolidated.is_(False),
                    Event.created_at >= text("NOW() - INTERVAL '24 hours'"),
                )
                .distinct()
            )
            return [(row.user_id, row.app_id) for row in result]

    async def _get_all_users(self) -> list[tuple[str, str]]:
        """Return all distinct (user_id, app_id) pairs in the events table."""
        from smritikosh.db.models import Event
        async with db_session() as session:
            result = await session.execute(
                select(Event.user_id, Event.app_id).distinct()
            )
            return [(row.user_id, row.app_id) for row in result]
