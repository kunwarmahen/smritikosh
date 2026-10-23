"""
Tests for memories in use not ageing: mark_used, the "active" order,
recency from last activity, the pruner's age, and fact decay that adds up
to its half-life instead of compounding.

THE BIAS these encode: time is the enemy of a memory nobody touches, not
of one an answer leans on every week. Each test is built against the
failure where a statement written long ago and used yesterday is ranked,
listed, pruned or decayed as if it had sat untouched since it was
written -- and against the opposite leak, one user's "used" refreshing
somebody else's memories.
"""

import math
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.dialects import postgresql

from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.synaptic_pruner import PruningThresholds, SynapticPruner


def _session(rowcount: int = 0) -> AsyncMock:
    session = AsyncMock()
    result = MagicMock()
    result.rowcount = rowcount
    result.scalars.return_value.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    return session


def _sql(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


# ── mark_used ─────────────────────────────────────────────────────────────────


class TestMarkUsed:
    async def test_it_stamps_last_used_at_and_counts_the_rows(self):
        session = _session(rowcount=2)

        marked = await EpisodicMemory().mark_used(session, "u1", [uuid.uuid4(), uuid.uuid4()])

        assert marked == 2
        sql = _sql(session.execute.call_args.args[0])
        assert "last_used_at" in sql

    async def test_it_only_touches_the_named_users_events(self):
        session = _session()

        await EpisodicMemory().mark_used(session, "u1", [uuid.uuid4()])

        sql = _sql(session.execute.call_args.args[0])
        assert "events.user_id = " in sql

    async def test_nothing_to_mark_is_no_query(self):
        session = _session()

        assert await EpisodicMemory().mark_used(session, "u1", []) == 0
        session.execute.assert_not_called()


# ── the order a client tops its prompt up from ───────────────────────────────


class TestActiveOrder:
    async def test_active_orders_by_the_later_of_written_and_used(self):
        session = _session()

        await EpisodicMemory().get_recent(session, "u1", order="active")

        sql = _sql(session.execute.call_args.args[0])
        assert "ORDER BY greatest(events.created_at, coalesce(events.last_used_at" in sql

    async def test_the_default_is_still_newest_written(self):
        session = _session()

        await EpisodicMemory().get_recent(session, "u1")

        sql = _sql(session.execute.call_args.args[0])
        assert "ORDER BY events.created_at DESC" in sql


# ── recency in hybrid search ─────────────────────────────────────────────────


class TestRecencyFromLastActivity:
    async def test_recency_counts_from_the_later_of_written_and_used(self):
        session = _session()
        result = MagicMock()
        result.__iter__ = MagicMock(return_value=iter([]))
        session.execute = AsyncMock(return_value=result)

        await EpisodicMemory().hybrid_search(session, "u1", [0.1, 0.2, 0.3])

        search_sql = str(session.execute.call_args_list[-1].args[0])
        assert "GREATEST(created_at, COALESCE(last_used_at, created_at)) AS active_at" in search_sql
        assert "NOW() - created_at" not in search_sql


# ── pruning ───────────────────────────────────────────────────────────────────


class TestPrunerAge:
    THRESHOLDS = PruningThresholds(importance_threshold=0.2, min_recall_count=2,
                                   min_age_days=90)

    def _event(self, written_days_ago: int, used_days_ago: int | None):
        now = datetime.now(timezone.utc)
        event = MagicMock()
        event.importance_score = 0.1
        event.recall_count = 0
        event.created_at = now - timedelta(days=written_days_ago)
        event.last_used_at = (None if used_days_ago is None
                              else now - timedelta(days=used_days_ago))
        return event

    def test_an_old_memory_used_last_week_is_kept(self):
        pruner = SynapticPruner(episodic=MagicMock())
        event = self._event(written_days_ago=400, used_days_ago=7)

        assert pruner._should_prune(event, datetime.now(timezone.utc), self.THRESHOLDS) is False

    def test_an_old_memory_never_used_is_still_pruned(self):
        pruner = SynapticPruner(episodic=MagicMock())
        event = self._event(written_days_ago=400, used_days_ago=None)

        assert pruner._should_prune(event, datetime.now(timezone.utc), self.THRESHOLDS) is True


# ── fact decay ────────────────────────────────────────────────────────────────


def _neo_session() -> AsyncMock:
    record = {"decayed_count": 0, "pending_count": 0, "deleted_count": 0,
              "orphans_deleted": 0}
    result = AsyncMock()
    result.single = AsyncMock(return_value=record)
    session = AsyncMock()
    session.run = AsyncMock(return_value=result)
    return session


class TestDecayAddsUp:
    async def test_each_pass_counts_from_the_last_pass_not_the_last_sighting(self):
        session = _neo_session()

        await SemanticMemory().decay_stale_facts(session)

        cypher = session.run.call_args_list[0].args[0]
        assert "r.decayed_at" in cypher
        assert "datetime(since)" in cypher
        assert "r.decayed_at = toString(datetime())" in cypher

    async def test_days_are_counted_whole_not_left_over_after_months(self):
        # duration.between(a, b).days is the remainder after whole months:
        # 56 days read as 25, and a 90-day cutoff could never be passed.
        session = _neo_session()
        semantic = SemanticMemory()

        await semantic.decay_stale_facts(session)
        await semantic.purge_unseen_facts(session, user_id="u1", not_seen_since_days=90)

        for call in session.run.call_args_list:
            assert "duration.between" not in call.args[0]

    def test_eight_weekly_passes_halve_a_60_day_fact_about_once(self):
        # The arithmetic the Cypher now does, pass by pass: each pass decays
        # by the days since the previous one. Eight weekly passes must land
        # where one 56-day decay would (about 52%), not near 5% as counting
        # from last_seen_at on every pass did.
        half_life, confidence = 60.0, 1.0
        for _ in range(8):
            confidence *= math.exp(-math.log(2) * 7 / half_life)
        assert confidence == pytest.approx(2 ** (-56 / 60))
        assert confidence > 0.5

        compounding = 1.0
        for week in range(1, 9):
            compounding *= math.exp(-math.log(2) * 7 * week / half_life)
        assert compounding < 0.1         # the old way: deleted at the floor
