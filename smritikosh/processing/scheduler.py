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

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import or_, select, text

from smritikosh.config import settings
from smritikosh.db.activity import mark_job_done
from smritikosh.db.models import UserActivity
from smritikosh.db.neo4j import neo4j_session
from smritikosh.db.postgres import db_session
from smritikosh.llm.usage import llm_context
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.metrics import JOB_USER_ERRORS, track_job
from smritikosh.processing.belief_miner import BeliefMiner, MiningResult
from smritikosh.processing.consolidator import Consolidator, ConsolidationResult
from smritikosh.processing.cross_system_synthesizer import CrossSystemSynthesizer, SynthesisResult
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
        synthesizer:             Optional CrossSystemSynthesizer instance.
        synthesis_cron:          Cron expression for cross-system synthesis (default: daily 01:00 UTC).
        reflection_agent:        Optional ReflectionAgent instance (E4).
        reflection_cron:         Cron expression for reflection cycles (default: daily 05:00 UTC).

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
        synthesizer: CrossSystemSynthesizer | None = None,
        synthesis_cron: str = "0 1 * * *",
        reflection_agent=None,   # cognition.reflection.ReflectionAgent | None
        reflection_cron: str = "0 5 * * *",
    ) -> None:
        self.consolidator = consolidator
        self.pruner = pruner
        self.episodic = episodic
        self.clusterer = clusterer
        self.belief_miner = belief_miner
        self.fact_decayer = fact_decayer
        self.synthesizer = synthesizer
        self.reflection_agent = reflection_agent
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
        if self.synthesizer is not None:
            self._scheduler.add_job(
                self.run_synthesis_for_all_users,
                trigger=CronTrigger.from_crontab(synthesis_cron),
                id="cross_system_synthesis_job",
                name="Cross-System Synthesis",
                max_instances=1,
            )
        if self.reflection_agent is not None:
            self._scheduler.add_job(
                self.run_reflection_for_all_users,
                trigger=CronTrigger.from_crontab(reflection_cron),
                id="reflection_job",
                name="Reflection Cycles",
                max_instances=1,
            )

    @property
    def running(self) -> bool:
        """True once start() has been called and the scheduler is active."""
        return self._scheduler.running

    def start(self) -> None:
        if self._scheduler.running:
            return
        self._scheduler.start()
        logger.info(
            "MemoryScheduler started",
            extra={
                "jobs": [j.id for j in self._scheduler.get_jobs()],
            },
        )

    def shutdown(self) -> None:
        # Idempotent: a standby process may never have started the scheduler.
        if not self._scheduler.running:
            return
        self._scheduler.shutdown(wait=False)
        logger.info("MemoryScheduler stopped.")

    # ── Manual triggers (also used internally by scheduler) ───────────────

    async def run_consolidation_for_all_users(self) -> list[ConsolidationResult]:
        """
        Discover users with unconsolidated events and consolidate each.
        Called automatically by the scheduler or manually via admin endpoints.
        """
        with track_job("consolidation"):
            user_app_pairs = await self._get_active_users()
            logger.info(
                "Consolidation job triggered — %d active user(s) found",
                len(user_app_pairs),
            )
            results: list[ConsolidationResult] = await self._run_for_users(
                user_app_pairs, self.run_consolidation_now
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Consolidation job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
            )
            return results

    async def run_pruning_for_all_users(self, override_thresholds=None) -> list[PruningResult]:
        """Run pruning for all users that have consolidated events."""
        with track_job("pruning"):
            user_app_pairs = await self._get_all_users("last_pruned_at")
            logger.info(
                "Pruning job triggered — %d user(s) found",
                len(user_app_pairs),
            )

            async def _prune_one(*, user_id: str, app_id: str) -> PruningResult:
                return await self.run_pruning_now(
                    user_id=user_id, app_id=app_id, override_thresholds=override_thresholds
                )

            results: list[PruningResult] = await self._run_for_users(
                user_app_pairs, _prune_one
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Pruning job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
            )
            return results

    async def run_consolidation_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> ConsolidationResult:
        """Run one consolidation cycle immediately for a specific user."""
        try:
            with llm_context(user_id=user_id, app_id=app_id, source="consolidation"):
                async with db_session() as pg, neo4j_session() as neo:
                    result = await self.consolidator.run(
                        pg, neo, user_id=user_id, app_id=app_id
                    )
                    await mark_job_done(pg, user_id, app_id, "consolidated")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="consolidation").inc()
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
            with llm_context(user_id=user_id, app_id=app_id, source="pruning"):
                async with db_session() as pg, neo4j_session() as neo:
                    result = await self.pruner.prune(
                        pg, user_id=user_id, app_id=app_id, neo_session=neo,
                        override_thresholds=override_thresholds,
                    )
                    await mark_job_done(pg, user_id, app_id, "pruned")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="pruning").inc()
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
        with track_job("belief_mining"):
            user_app_pairs = await self._get_all_users("last_belief_mined_at")
            logger.info(
                "Belief mining job triggered — %d user(s) found",
                len(user_app_pairs),
            )
            results: list[MiningResult] = await self._run_for_users(
                user_app_pairs, self.run_belief_mining_now
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Belief mining job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
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
            with llm_context(user_id=user_id, app_id=app_id, source="belief_mining"):
                async with db_session() as pg, neo4j_session() as neo:
                    result = await self.belief_miner.mine(
                        pg, neo, user_id=user_id, app_id=app_id
                    )
                    await mark_job_done(pg, user_id, app_id, "belief_mined")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="belief_mining").inc()
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
        with track_job("clustering"):
            user_app_pairs = await self._get_all_users("last_clustered_at")
            logger.info(
                "Clustering job triggered — %d user(s) found",
                len(user_app_pairs),
            )
            results: list[ClusterResult] = await self._run_for_users(
                user_app_pairs, self.run_clustering_now
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Clustering job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
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
            with llm_context(user_id=user_id, app_id=app_id, source="clustering"):
                async with db_session() as pg:
                    result = await self.clusterer.run(pg, user_id=user_id, app_id=app_id)
                    await mark_job_done(pg, user_id, app_id, "clustered")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="clustering").inc()
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
        logger.info("Fact decay job triggered")
        try:
            with track_job("fact_decay"), llm_context(source="fact_decay"):
                async with neo4j_session() as neo:
                    result = await self.fact_decayer.run(neo)
            logger.info("Fact decay job complete")
            return result
        except Exception as exc:
            logger.error("Fact decay failed: %s", exc)
            result = DecayResult(skipped=True)
            result.skip_reason = str(exc)
            return result

    async def run_synthesis_for_all_users(self) -> list[SynthesisResult]:
        """Run cross-system synthesis for all users. Called daily by the scheduler."""
        if self.synthesizer is None:
            return []
        with track_job("synthesis"):
            user_app_pairs = await self._get_all_users("last_synthesized_at")
            logger.info(
                "Cross-system synthesis job triggered — %d user(s) found",
                len(user_app_pairs),
            )
            results: list[SynthesisResult] = await self._run_for_users(
                user_app_pairs, self.run_synthesis_now
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Cross-system synthesis job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
            )
            return results

    async def run_synthesis_now(
        self, *, user_id: str, app_id: str = "default"
    ) -> SynthesisResult:
        """Run cross-system synthesis immediately for a specific user."""
        if self.synthesizer is None:
            result = SynthesisResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = "No synthesizer configured."
            return result
        try:
            with llm_context(user_id=user_id, app_id=app_id, source="synthesis"):
                async with db_session() as pg, neo4j_session() as neo:
                    result = await self.synthesizer.run(
                        pg, neo, user_id=user_id, app_id=app_id
                    )
                    await mark_job_done(pg, user_id, app_id, "synthesized")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="synthesis").inc()
            logger.error(
                "Cross-system synthesis failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = SynthesisResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = str(exc)
            return result

    async def run_reflection_for_all_users(self) -> list:
        """Run reflection cycles for all users, stalest first (E4)."""
        if self.reflection_agent is None:
            return []
        with track_job("reflection"):
            user_app_pairs = await self._get_all_users("last_reflected_at")
            logger.info(
                "Reflection job triggered — %d user(s) found",
                len(user_app_pairs),
            )
            results = await self._run_for_users(
                user_app_pairs, self.run_reflection_now
            )
            skipped = sum(1 for r in results if r.skipped)
            logger.info(
                "Reflection job complete — processed=%d skipped=%d",
                len(results) - skipped,
                skipped,
            )
            return results

    async def run_reflection_now(
        self, *, user_id: str, app_id: str = "default"
    ):
        """Run one reflection cycle immediately for a specific user."""
        from smritikosh.cognition.reflection import ReflectionResult

        if self.reflection_agent is None:
            result = ReflectionResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = "No reflection_agent configured."
            return result
        try:
            with llm_context(user_id=user_id, app_id=app_id, source="reflection"):
                async with db_session() as pg, neo4j_session() as neo:
                    result = await self.reflection_agent.reflect(
                        pg, neo, user_id=user_id, app_id=app_id
                    )
                    await mark_job_done(pg, user_id, app_id, "reflected")
                    return result
        except Exception as exc:
            JOB_USER_ERRORS.labels(job="reflection").inc()
            logger.error(
                "Reflection failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result = ReflectionResult(user_id=user_id, app_id=app_id, skipped=True)
            result.skip_reason = str(exc)
            return result

    # ── User discovery ─────────────────────────────────────────────────────

    async def _get_active_users(self) -> list[tuple[str, str]]:
        """
        Return (user_id, app_id) pairs worth consolidating: active in the last
        24 hours AND with events newer than their last consolidation.

        Indexed lookup on user_activity (item A5) — no events-table scan. The
        consolidation watermark means tenants with nothing new are skipped
        entirely, which the old DISTINCT query could not do.
        """
        async with db_session() as session:
            result = await session.execute(
                select(UserActivity.user_id, UserActivity.app_id).where(
                    UserActivity.last_event_at >= text("NOW() - INTERVAL '24 hours'"),
                    or_(
                        UserActivity.last_consolidated_at.is_(None),
                        UserActivity.last_consolidated_at < UserActivity.last_event_at,
                    ),
                )
            )
            pairs = [(row.user_id, row.app_id) for row in result]
            if pairs:
                return pairs
            # Nothing matched. If the activity table has rows, that's a real
            # "no one needs work" — but an empty table means a deployment that
            # predates it (created via create_all, no backfill): fall back to
            # the legacy scan so existing tenants are not silently dropped.
            if await self._activity_table_populated(session):
                return []
            return await self._legacy_active_users(session)

    async def _get_all_users(
        self, stale_watermark: str | None = None
    ) -> list[tuple[str, str]]:
        """All known (user_id, app_id) pairs from user_activity, stalest first.

        `stale_watermark` names a UserActivity column (e.g. "last_pruned_at");
        tenants never processed sort first, so a job that is killed mid-cycle
        makes progress across restarts instead of restarting from the same end.
        """
        async with db_session() as session:
            query = select(UserActivity.user_id, UserActivity.app_id)
            if stale_watermark is not None:
                column = getattr(UserActivity, stale_watermark)
                query = query.order_by(column.asc().nulls_first())
            result = await session.execute(query)
            pairs = [(row.user_id, row.app_id) for row in result]
            if pairs:
                return pairs
            return await self._legacy_all_users(session)

    @staticmethod
    async def _activity_table_populated(session) -> bool:
        result = await session.execute(select(UserActivity.id).limit(1))
        return result.first() is not None

    @staticmethod
    async def _legacy_active_users(session) -> list[tuple[str, str]]:
        """Pre-A5 discovery: DISTINCT scan over events. Fallback only."""
        from smritikosh.db.models import Event

        result = await session.execute(
            select(Event.user_id, Event.app_id)
            .where(
                Event.consolidated.is_(False),
                Event.created_at >= text("NOW() - INTERVAL '24 hours'"),
            )
            .distinct()
        )
        return [(row.user_id, row.app_id) for row in result]

    @staticmethod
    async def _legacy_all_users(session) -> list[tuple[str, str]]:
        """Pre-A5 discovery: DISTINCT scan over events. Fallback only."""
        from smritikosh.db.models import Event

        result = await session.execute(
            select(Event.user_id, Event.app_id).distinct()
        )
        return [(row.user_id, row.app_id) for row in result]

    # ── Bounded-concurrency fan-out ─────────────────────────────────────────

    async def _run_for_users(self, pairs, runner) -> list:
        """Run `runner(user_id=…, app_id=…)` for every pair, at most
        SCHEDULER_JOB_CONCURRENCY at a time. Results keep the order of `pairs`.

        Replaces the sequential per-user loop: one slow tenant no longer
        blocks every tenant behind it. Each runner swallows its own errors
        (returning a skipped result), so gather never aborts the batch.
        """
        limit = max(1, settings.scheduler_job_concurrency)
        semaphore = asyncio.Semaphore(limit)

        async def _one(user_id: str, app_id: str):
            async with semaphore:
                return await runner(user_id=user_id, app_id=app_id)

        return list(await asyncio.gather(*(_one(u, a) for u, a in pairs)))


# ── Scheduler bootstrap helpers ─────────────────────────────────────────────────


def build_scheduler() -> "MemoryScheduler":
    """Construct a MemoryScheduler wired with every processing job.

    Shared by the API process (in-process scheduler) and the standalone worker,
    so both build an identically-configured scheduler. Dependency getters are
    imported lazily to avoid a circular import at module load time.
    """
    from smritikosh.api.deps import (
        get_belief_miner,
        get_clusterer,
        get_consolidator,
        get_episodic,
        get_fact_decayer,
        get_pruner,
        get_reflection_agent,
        get_synthesizer,
    )
    from smritikosh.config import settings

    return MemoryScheduler(
        consolidator=get_consolidator(),
        pruner=get_pruner(),
        episodic=get_episodic(),
        clusterer=get_clusterer(),
        belief_miner=get_belief_miner(),
        fact_decayer=get_fact_decayer(),
        synthesizer=get_synthesizer(),
        reflection_agent=get_reflection_agent(),
        consolidation_cron=settings.scheduler_consolidation_cron,
        pruning_cron=settings.scheduler_pruning_cron,
        clustering_cron=settings.scheduler_clustering_cron,
        belief_mining_cron=settings.scheduler_belief_mining_cron,
        fact_decay_cron=settings.scheduler_fact_decay_cron,
        reflection_cron=settings.scheduler_reflection_cron,
    )


async def elect_and_start_scheduler(
    scheduler: "MemoryScheduler",
    leader_lock,  # processing.leader.LeaderLock
    *,
    poll_interval: float = 30.0,
) -> bool:
    """Block until this process wins leader election, then start the scheduler.

    Polls every `poll_interval` seconds while another process holds the lock, so
    a standby starts the scheduler automatically if the current leader dies.
    Returns True once this process has started the scheduler. Cancel the awaiting
    task to stop waiting (e.g. on shutdown).
    """
    announced = False
    while True:
        if await leader_lock.try_acquire():
            scheduler.start()
            logger.info("Won scheduler leader election — background jobs run in this process.")
            return True
        if not announced:
            logger.info(
                "Another process holds the scheduler lock — standing by "
                "(re-checking every %.0fs).",
                poll_interval,
            )
            announced = True
        await asyncio.sleep(poll_interval)
