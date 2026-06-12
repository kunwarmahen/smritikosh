"""Tests for user_activity discovery + per-user job fan-out (item A5).

Covers:
- touch_user_activity / mark_job_done — upsert construction, watermark guard
- EpisodicMemory.store — touches user_activity in the same transaction
- MemoryScheduler._get_active_users — indexed query, legacy fallback rules
- MemoryScheduler._get_all_users — staleness ordering (NULLS FIRST)
- MemoryScheduler._run_for_users — bounded concurrency, order preservation
- run_consolidation_now — stamps the consolidation watermark on success
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.dialects import postgresql

from smritikosh.db.activity import JOB_WATERMARKS, mark_job_done, touch_user_activity
from smritikosh.processing.consolidator import ConsolidationResult
from smritikosh.processing.scheduler import MemoryScheduler


# ── Helpers ───────────────────────────────────────────────────────────────────


def compiled(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def rows_result(rows: list) -> MagicMock:
    result = MagicMock()
    result.__iter__ = MagicMock(return_value=iter(rows))
    return result


def make_scheduler(**kwargs) -> MemoryScheduler:
    with patch("smritikosh.processing.scheduler.AsyncIOScheduler"):
        return MemoryScheduler(
            consolidator=kwargs.get("consolidator", AsyncMock()),
            pruner=kwargs.get("pruner", AsyncMock()),
            episodic=AsyncMock(),
        )


def patch_db_session(pg: AsyncMock):
    ctx = patch("smritikosh.processing.scheduler.db_session")
    mock_db = ctx.start()
    mock_db.return_value.__aenter__ = AsyncMock(return_value=pg)
    mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ── Activity helpers ──────────────────────────────────────────────────────────


class TestTouchUserActivity:
    @pytest.mark.asyncio
    async def test_upserts_last_event_at(self):
        session = AsyncMock()
        await touch_user_activity(session, "u1", "myapp")

        session.execute.assert_awaited_once()
        stmt = session.execute.call_args.args[0]
        sql = compiled(stmt)
        assert "user_activity" in sql
        assert "ON CONFLICT" in sql
        params = stmt.compile().params
        assert params["user_id"] == "u1"
        assert params["app_id"] == "myapp"


class TestMarkJobDone:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("job,column", sorted(JOB_WATERMARKS.items()))
    async def test_stamps_named_watermark(self, job, column):
        session = AsyncMock()
        await mark_job_done(session, "u1", "default", job)

        stmt = session.execute.call_args.args[0]
        assert column in stmt.compile().params
        assert "ON CONFLICT" in compiled(stmt)

    @pytest.mark.asyncio
    async def test_unknown_job_raises(self):
        with pytest.raises(ValueError, match="Unknown job watermark"):
            await mark_job_done(AsyncMock(), "u1", "default", "definitely_not_a_job")


class TestStoreTouchesActivity:
    @pytest.mark.asyncio
    async def test_store_upserts_activity_row(self):
        from smritikosh.memory.episodic import EpisodicMemory

        session = AsyncMock()
        session.add = MagicMock()
        await EpisodicMemory().store(session, user_id="u1", raw_text="hello")

        session.execute.assert_awaited_once()
        assert "user_activity" in compiled(session.execute.call_args.args[0])


# ── Discovery queries ─────────────────────────────────────────────────────────


class TestGetActiveUsers:
    @pytest.mark.asyncio
    async def test_returns_activity_rows(self):
        pg = AsyncMock()
        pg.execute = AsyncMock(
            return_value=rows_result([SimpleNamespace(user_id="u1", app_id="a1")])
        )
        ctx = patch_db_session(pg)
        try:
            pairs = await make_scheduler()._get_active_users()
        finally:
            ctx.stop()

        assert pairs == [("u1", "a1")]
        assert pg.execute.await_count == 1  # no fallback queries

    @pytest.mark.asyncio
    async def test_empty_but_populated_table_means_no_work(self):
        pg = AsyncMock()
        populated_check = MagicMock()
        populated_check.first.return_value = ("some-row",)
        pg.execute = AsyncMock(side_effect=[rows_result([]), populated_check])
        ctx = patch_db_session(pg)
        try:
            pairs = await make_scheduler()._get_active_users()
        finally:
            ctx.stop()

        assert pairs == []
        assert pg.execute.await_count == 2  # no legacy scan

    @pytest.mark.asyncio
    async def test_empty_table_falls_back_to_legacy_scan(self):
        pg = AsyncMock()
        empty_check = MagicMock()
        empty_check.first.return_value = None
        legacy = rows_result([SimpleNamespace(user_id="legacy", app_id="default")])
        pg.execute = AsyncMock(side_effect=[rows_result([]), empty_check, legacy])
        ctx = patch_db_session(pg)
        try:
            pairs = await make_scheduler()._get_active_users()
        finally:
            ctx.stop()

        assert pairs == [("legacy", "default")]


class TestGetAllUsers:
    @pytest.mark.asyncio
    async def test_orders_by_watermark_nulls_first(self):
        pg = AsyncMock()
        pg.execute = AsyncMock(
            return_value=rows_result([SimpleNamespace(user_id="u1", app_id="a1")])
        )
        ctx = patch_db_session(pg)
        try:
            pairs = await make_scheduler()._get_all_users("last_pruned_at")
        finally:
            ctx.stop()

        assert pairs == [("u1", "a1")]
        sql = compiled(pg.execute.call_args.args[0])
        assert "ORDER BY" in sql
        assert "NULLS FIRST" in sql
        assert "last_pruned_at" in sql

    @pytest.mark.asyncio
    async def test_empty_table_falls_back_to_legacy_scan(self):
        pg = AsyncMock()
        legacy = rows_result([SimpleNamespace(user_id="legacy", app_id="default")])
        pg.execute = AsyncMock(side_effect=[rows_result([]), legacy])
        ctx = patch_db_session(pg)
        try:
            pairs = await make_scheduler()._get_all_users("last_pruned_at")
        finally:
            ctx.stop()

        assert pairs == [("legacy", "default")]
        assert "DISTINCT" in compiled(pg.execute.call_args.args[0])


# ── Bounded-concurrency fan-out ───────────────────────────────────────────────


class TestRunForUsers:
    @pytest.mark.asyncio
    async def test_concurrency_is_bounded(self, monkeypatch):
        from smritikosh.config import settings

        monkeypatch.setattr(settings, "scheduler_job_concurrency", 2)
        current = 0
        peak = 0

        async def runner(*, user_id: str, app_id: str) -> str:
            nonlocal current, peak
            current += 1
            peak = max(peak, current)
            await asyncio.sleep(0.01)
            current -= 1
            return user_id

        pairs = [(f"u{i}", "default") for i in range(6)]
        results = await make_scheduler()._run_for_users(pairs, runner)

        assert peak <= 2
        assert results == [f"u{i}" for i in range(6)]  # order preserved

    @pytest.mark.asyncio
    async def test_empty_pairs(self):
        assert await make_scheduler()._run_for_users([], AsyncMock()) == []


# ── Watermark stamping on job success ─────────────────────────────────────────


class TestWatermarkStamping:
    @pytest.mark.asyncio
    async def test_consolidation_stamps_watermark(self):
        consolidator = AsyncMock()
        consolidator.run = AsyncMock(
            return_value=ConsolidationResult(user_id="u1", app_id="default")
        )
        scheduler = make_scheduler(consolidator=consolidator)

        pg = AsyncMock()
        with (
            patch("smritikosh.processing.scheduler.db_session") as mock_db,
            patch("smritikosh.processing.scheduler.neo4j_session") as mock_neo,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=pg)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_neo.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_neo.return_value.__aexit__ = AsyncMock(return_value=False)

            await scheduler.run_consolidation_now(user_id="u1", app_id="default")

        stamped = [
            call.args[0]
            for call in pg.execute.call_args_list
            if "user_activity" in compiled(call.args[0])
        ]
        assert len(stamped) == 1
        assert "last_consolidated_at" in stamped[0].compile().params

    @pytest.mark.asyncio
    async def test_failed_run_does_not_stamp(self):
        consolidator = AsyncMock()
        consolidator.run = AsyncMock(side_effect=RuntimeError("boom"))
        scheduler = make_scheduler(consolidator=consolidator)

        pg = AsyncMock()
        with (
            patch("smritikosh.processing.scheduler.db_session") as mock_db,
            patch("smritikosh.processing.scheduler.neo4j_session") as mock_neo,
        ):
            mock_db.return_value.__aenter__ = AsyncMock(return_value=pg)
            mock_db.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_neo.return_value.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_neo.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await scheduler.run_consolidation_now(user_id="u1")

        assert result.skipped is True
        pg.execute.assert_not_awaited()
