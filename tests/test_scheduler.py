"""
Tests for MemoryScheduler.

Unit tests verify job registration, manual triggers, and user discovery logic.
No real DB or LLM connections are used.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.processing.consolidator import ConsolidationResult
from smritikosh.processing.synaptic_pruner import PruningResult
from smritikosh.processing.scheduler import MemoryScheduler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_consolidator():
    c = AsyncMock()
    c.run = AsyncMock(
        return_value=ConsolidationResult(user_id="u1", app_id="default")
    )
    return c


@pytest.fixture
def mock_pruner():
    p = AsyncMock()
    p.prune = AsyncMock(
        return_value=PruningResult(user_id="u1", app_id="default")
    )
    return p


@pytest.fixture
def mock_episodic():
    return AsyncMock()


@pytest.fixture
def scheduler(mock_consolidator, mock_pruner, mock_episodic):
    with patch("smritikosh.processing.scheduler.AsyncIOScheduler") as mock_sched_cls:
        mock_sched = MagicMock()
        mock_sched.get_jobs.return_value = [
            MagicMock(id="consolidation_job"),
            MagicMock(id="pruning_job"),
        ]
        mock_sched_cls.return_value = mock_sched

        s = MemoryScheduler(
            consolidator=mock_consolidator,
            pruner=mock_pruner,
            episodic=mock_episodic,
        )
        s._scheduler = mock_sched  # keep reference for assertions
        return s


# ── Scheduler construction ────────────────────────────────────────────────────

class TestSchedulerConstruction:
    def test_registers_consolidation_job(self, mock_consolidator, mock_pruner, mock_episodic):
        with patch("smritikosh.processing.scheduler.AsyncIOScheduler") as MockSched:
            instance = MagicMock()
            instance.get_jobs.return_value = []
            MockSched.return_value = instance

            MemoryScheduler(
                consolidator=mock_consolidator,
                pruner=mock_pruner,
                episodic=mock_episodic,
            )

            calls = instance.add_job.call_args_list
            job_ids = [c.kwargs.get("id") or c[1].get("id") for c in calls]
            assert "consolidation_job" in job_ids

    def test_registers_pruning_job(self, mock_consolidator, mock_pruner, mock_episodic):
        with patch("smritikosh.processing.scheduler.AsyncIOScheduler") as MockSched:
            instance = MagicMock()
            instance.get_jobs.return_value = []
            MockSched.return_value = instance

            MemoryScheduler(
                consolidator=mock_consolidator,
                pruner=mock_pruner,
                episodic=mock_episodic,
            )

            calls = instance.add_job.call_args_list
            job_ids = [c.kwargs.get("id") or c[1].get("id") for c in calls]
            assert "pruning_job" in job_ids

    def test_custom_intervals(self, mock_consolidator, mock_pruner, mock_episodic):
        with patch("smritikosh.processing.scheduler.AsyncIOScheduler") as MockSched:
            instance = MagicMock()
            instance.get_jobs.return_value = []
            MockSched.return_value = instance

            MemoryScheduler(
                consolidator=mock_consolidator,
                pruner=mock_pruner,
                episodic=mock_episodic,
                consolidation_hours=2,
                pruning_hours=48,
            )

            calls = instance.add_job.call_args_list
            # First call is consolidation, second is pruning
            consol_kwargs = calls[0][1]
            prune_kwargs = calls[1][1]
            assert consol_kwargs["hours"] == 2
            assert prune_kwargs["hours"] == 48


# ── start / shutdown ──────────────────────────────────────────────────────────

class TestStartShutdown:
    def test_start_calls_scheduler_start(self, scheduler):
        scheduler.start()
        scheduler._scheduler.start.assert_called_once()

    def test_shutdown_calls_scheduler_shutdown(self, scheduler):
        scheduler.shutdown()
        scheduler._scheduler.shutdown.assert_called_once_with(wait=False)


# ── run_consolidation_now ─────────────────────────────────────────────────────

class TestRunConsolidationNow:
    async def test_calls_consolidator_run(self, scheduler, mock_consolidator):
        with (
            patch("smritikosh.processing.scheduler.db_session") as mock_db,
            patch("smritikosh.processing.scheduler.neo4j_session") as mock_neo,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_neo.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_neo.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await scheduler.run_consolidation_now(user_id="u1", app_id="default")

        mock_consolidator.run.assert_called_once()
        assert result.user_id == "u1"

    async def test_returns_skipped_result_on_error(self, scheduler, mock_consolidator):
        mock_consolidator.run.side_effect = RuntimeError("DB error")
        with (
            patch("smritikosh.processing.scheduler.db_session") as mock_db,
            patch("smritikosh.processing.scheduler.neo4j_session") as mock_neo,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_neo.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_neo.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await scheduler.run_consolidation_now(user_id="u1", app_id="default")

        assert result.skipped is True
        assert "DB error" in result.skip_reason


# ── run_pruning_now ───────────────────────────────────────────────────────────

class TestRunPruningNow:
    async def test_calls_pruner_prune(self, scheduler, mock_pruner):
        with patch("smritikosh.processing.scheduler.db_session") as mock_db:
            mock_db.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await scheduler.run_pruning_now(user_id="u1", app_id="default")

        mock_pruner.prune.assert_called_once()
        assert result.user_id == "u1"

    async def test_returns_skipped_result_on_error(self, scheduler, mock_pruner):
        mock_pruner.prune.side_effect = RuntimeError("Neo4j down")
        with patch("smritikosh.processing.scheduler.db_session") as mock_db:
            mock_db.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await scheduler.run_pruning_now(user_id="u1", app_id="default")

        assert result.skipped is True


# ── run_consolidation_for_all_users ──────────────────────────────────────────

class TestBatchConsolidation:
    async def test_processes_all_active_users(self, scheduler, mock_consolidator):
        user_pairs = [("u1", "app1"), ("u2", "app2")]

        with patch.object(scheduler, "_get_active_users", AsyncMock(return_value=user_pairs)):
            with patch.object(
                scheduler, "run_consolidation_now",
                AsyncMock(side_effect=lambda **kw: ConsolidationResult(
                    user_id=kw["user_id"], app_id=kw["app_id"]
                ))
            ):
                results = await scheduler.run_consolidation_for_all_users()

        assert len(results) == 2
        assert {r.user_id for r in results} == {"u1", "u2"}

    async def test_empty_active_users(self, scheduler):
        with patch.object(scheduler, "_get_active_users", AsyncMock(return_value=[])):
            results = await scheduler.run_consolidation_for_all_users()

        assert results == []


# ── run_pruning_for_all_users ─────────────────────────────────────────────────

class TestBatchPruning:
    async def test_processes_all_users(self, scheduler):
        user_pairs = [("u1", "default"), ("u3", "app3")]

        with patch.object(scheduler, "_get_all_users", AsyncMock(return_value=user_pairs)):
            with patch.object(
                scheduler, "run_pruning_now",
                AsyncMock(side_effect=lambda **kw: PruningResult(
                    user_id=kw["user_id"], app_id=kw["app_id"]
                ))
            ):
                results = await scheduler.run_pruning_for_all_users()

        assert len(results) == 2

    async def test_empty_users(self, scheduler):
        with patch.object(scheduler, "_get_all_users", AsyncMock(return_value=[])):
            results = await scheduler.run_pruning_for_all_users()

        assert results == []
