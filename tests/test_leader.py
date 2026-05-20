"""Tests for scheduler leader election and the standalone worker (item A1).

Covers:
- LeaderLock — Postgres advisory-lock acquire / idempotency / release
- elect_and_start_scheduler — starts the scheduler once leadership is won
- build_scheduler — constructs a fully-wired, not-yet-running scheduler
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.processing import leader as leader_mod
from smritikosh.processing.leader import LeaderLock
from smritikosh.processing.scheduler import MemoryScheduler, build_scheduler, elect_and_start_scheduler


def _mock_conn(acquired: bool) -> AsyncMock:
    """An AsyncMock connection whose advisory-lock query returns `acquired`."""
    conn = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = acquired
    conn.execute.return_value = result
    return conn


class TestLeaderLock:
    async def test_acquire_succeeds_and_sets_leader(self):
        conn = _mock_conn(acquired=True)
        with patch.object(leader_mod, "engine") as eng:
            eng.connect = AsyncMock(return_value=conn)
            lock = LeaderLock()
            assert await lock.try_acquire() is True
            assert lock.is_leader is True
            conn.close.assert_not_awaited()  # connection kept — the lock lives on it

    async def test_acquire_fails_when_lock_held_elsewhere(self):
        conn = _mock_conn(acquired=False)
        with patch.object(leader_mod, "engine") as eng:
            eng.connect = AsyncMock(return_value=conn)
            lock = LeaderLock()
            assert await lock.try_acquire() is False
            assert lock.is_leader is False
            conn.close.assert_awaited()  # losing process releases its connection

    async def test_acquire_is_idempotent_once_leader(self):
        conn = _mock_conn(acquired=True)
        with patch.object(leader_mod, "engine") as eng:
            eng.connect = AsyncMock(return_value=conn)
            lock = LeaderLock()
            await lock.try_acquire()
            await lock.try_acquire()  # second call must not open a new connection
            eng.connect.assert_awaited_once()

    async def test_acquire_handles_db_error_gracefully(self):
        with patch.object(leader_mod, "engine") as eng:
            eng.connect = AsyncMock(side_effect=OSError("postgres unreachable"))
            lock = LeaderLock()
            assert await lock.try_acquire() is False
            assert lock.is_leader is False

    async def test_release_unlocks_and_closes(self):
        conn = _mock_conn(acquired=True)
        with patch.object(leader_mod, "engine") as eng:
            eng.connect = AsyncMock(return_value=conn)
            lock = LeaderLock()
            await lock.try_acquire()
            await lock.release()
            assert lock.is_leader is False
            conn.close.assert_awaited()

    async def test_release_is_safe_when_never_acquired(self):
        lock = LeaderLock()
        await lock.release()  # must not raise
        assert lock.is_leader is False


class TestElectAndStartScheduler:
    async def test_starts_scheduler_when_leadership_won_immediately(self):
        scheduler = MagicMock()
        lock = MagicMock()
        lock.try_acquire = AsyncMock(return_value=True)

        result = await elect_and_start_scheduler(scheduler, lock)

        assert result is True
        scheduler.start.assert_called_once()

    async def test_waits_then_starts_when_leadership_won_later(self):
        scheduler = MagicMock()
        lock = MagicMock()
        # Lose the first election, win the second.
        lock.try_acquire = AsyncMock(side_effect=[False, True])

        result = await elect_and_start_scheduler(scheduler, lock, poll_interval=0)

        assert result is True
        assert lock.try_acquire.await_count == 2
        scheduler.start.assert_called_once()


class TestBuildScheduler:
    def test_builds_a_not_running_scheduler(self):
        scheduler = build_scheduler()
        assert isinstance(scheduler, MemoryScheduler)
        assert scheduler.running is False

    def test_shutdown_is_idempotent_before_start(self):
        scheduler = build_scheduler()
        scheduler.shutdown()  # never started — must be a safe no-op
        assert scheduler.running is False
