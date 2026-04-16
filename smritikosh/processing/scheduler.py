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
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select, text

from smritikosh.db.neo4j import neo4j_session
from smritikosh.db.postgres import db_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.processing.belief_miner import BeliefMiner, MiningResult
from smritikosh.processing.consolidator import Consolidator, ConsolidationResult
from smritikosh.processing.fact_decayer import DecayResult, FactDecayer
from smritikosh.processing.memory_clusterer import ClusterResult, MemoryClusterer
from smritikosh.processing.synaptic_pruner import PruningResult, SynapticPruner

logger = logging.getLogger(__name__)


class MemoryScheduler:
    """
    Manages periodic memory maintenance jobs.

    Both jobs can also be triggered manually (useful for testing or admin CLI).

    Args:
        consolidator:            Consolidator instance.
        pruner:                  SynapticPruner instance.
        episodic:                EpisodicMemory — used to discover active users.
        consolidation_cron:      Cron expression for consolidation (default: hourly).
        pruning_cron:            Cron expression for pruning (default: daily 02:00 UTC).
        clusterer:               Optional MemoryClusterer instance.
        clustering_cron:         Cron expression for clustering (default: every 6 hours).
        belief_miner:            Optional BeliefMiner instance.
        belief_mining_cron:      Cron expression for belief mining (default: every 12 hours).
        fact_decayer:            Optional FactDecayer instance.
        fact_decay_cron:         Cron expression for fact decay (default: weekly Sunday 03:00 UTC).

    Cron format: standard 5-field UTC — "minute hour day-of-month month day-of-week"
    Examples:
        "0 * * * *"   — every hour on the hour
        "0 2 * * *"   — daily at 02:00 UTC
        "0 3 * * 0"   — every Sunday at 03:00 UTC
    """

    def __init__(
        self,
        consolidator: Consolidator,
        pruner: SynapticPruner,
        episodic: EpisodicMemory,
        consolidation_cron: str = "0 * * * *",
        pruning_cron: str = "0 2 * * *",
        clusterer: MemoryClusterer | None = None,
        clustering_cron: str = "0 */6 * * *",
        belief_miner: BeliefMiner | None = None,
        belief_mining_cron: str = "0 */12 * * *",
        fact_decayer: FactDecayer | None = None,
        fact_decay_cron: str = "0 3 * * 0",
    ) -> None:
        self.consolidator = consolidator
        self.pruner = pruner
        self.episodic = episodic
        self.clusterer = clusterer
        self.belief_miner = belief_miner
        self.fact_decayer = fact_decayer
        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            self.run_consolidation_for_all_users,
            trigger=CronTrigger.from_crontab(consolidation_cron),
            id="consolidation_job",
            name="Memory Consolidation",
            max_instances=1,       # never overlap
        )
        self._scheduler.add_job(
            self.run_pruning_for_all_users,
            trigger=CronTrigger.from_crontab(pruning_cron),
            id="pruning_job",
            name="Synaptic Pruning",
            max_instances=1,
        )
        if self.clusterer is not None:
            self._scheduler.add_job(
                self.run_clustering_for_all_users,
                trigger=CronTrigger.from_crontab(clustering_cron),
                id="clustering_job",
                name="Memory Clustering",
                max_instances=1,
            )
        if self.belief_miner is not None:
            self._scheduler.add_job(
                self.run_belief_mining_for_all_users,
                trigger=CronTrigger.from_crontab(belief_mining_cron),
                id="belief_mining_job",
                name="Belief Mining",
                max_instances=1,
            )
        if self.fact_decayer is not None:
            self._scheduler.add_job(
                self.run_fact_decay,
                trigger=CronTrigger.from_crontab(fact_decay_cron),
                id="fact_decay_job",
                name="Semantic Fact Decay",
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

    async def run_pruning_for_all_users(self, override_thresholds=None) -> list[PruningResult]:
        """Run pruning for all users that have consolidated events."""
        user_app_pairs = await self._get_all_users()
        results: list[PruningResult] = []

        for user_id, app_id in user_app_pairs:
            result = await self.run_pruning_now(
                user_id=user_id, app_id=app_id, override_thresholds=override_thresholds
            )
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
        self,
        *,
        user_id: str,
        app_id: str = "default",
        override_thresholds=None,  # PruningThresholds | None
    ) -> PruningResult:
        """Run pruning immediately for a specific user."""
        try:
            async with db_session() as pg, neo4j_session() as neo:
                return await self.pruner.prune(
                    pg, user_id=user_id, app_id=app_id, neo_session=neo,
                    override_thresholds=override_thresholds,
                )
        except Exception as exc:
            logger.error(
                "Pruning failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = PruningResult(user_id=user_id, app_id=app_id, skipped=True)
            return result

    async def run_belief_mining_for_all_users(self) -> list[MiningResult]:
        """Run belief mining for all users that have consolidated events."""
        if self.belief_miner is None:
            return []
        user_app_pairs = await self._get_all_users()
        results: list[MiningResult] = []

        for user_id, app_id in user_app_pairs:
            result = await self.run_belief_mining_now(user_id=user_id, app_id=app_id)
            results.append(result)

        logger.info(
            "Batch belief mining complete",
            extra={"users_processed": len(results)},
        )
        return results

    async def run_belief_mining_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> MiningResult:
        """Run belief mining immediately for a specific user."""
        if self.belief_miner is None:
            result = MiningResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = "No belief_miner configured."
            return result
        try:
            async with db_session() as pg, neo4j_session() as neo:
                return await self.belief_miner.mine(
                    pg, neo, user_id=user_id, app_id=app_id
                )
        except Exception as exc:
            logger.error(
                "Belief mining failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = MiningResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = str(exc)
            return result

    async def run_clustering_for_all_users(self) -> list[ClusterResult]:
        """Run clustering for all users that have events with embeddings."""
        if self.clusterer is None:
            return []
        user_app_pairs = await self._get_all_users()
        results: list[ClusterResult] = []

        for user_id, app_id in user_app_pairs:
            result = await self.run_clustering_now(user_id=user_id, app_id=app_id)
            results.append(result)

        logger.info(
            "Batch clustering complete",
            extra={"users_processed": len(results)},
        )
        return results

    async def run_clustering_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> ClusterResult:
        """Run clustering immediately for a specific user."""
        if self.clusterer is None:
            result = ClusterResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = "No clusterer configured."
            return result
        try:
            async with db_session() as pg:
                return await self.clusterer.run(pg, user_id=user_id, app_id=app_id)
        except Exception as exc:
            logger.error(
                "Clustering failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = ClusterResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = str(exc)
            return result

    async def run_fact_decay(self) -> DecayResult:
        """
        Run one full fact decay cycle across all Neo4j facts.
        Called automatically by the weekly scheduler or manually via admin.
        """
        if self.fact_decayer is None:
            result = DecayResult(skipped=True)
            result.skip_reason = "No fact_decayer configured."
            return result
        try:
            async with neo4j_session() as neo:
                return await self.fact_decayer.run(neo)
        except Exception as exc:
            logger.error("Fact decay failed", extra={"error": str(exc)})
            result = DecayResult(skipped=True)
            result.skip_reason = str(exc)
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
