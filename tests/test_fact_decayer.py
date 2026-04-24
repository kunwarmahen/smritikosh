"""
Tests for FactDecayer and SemanticMemory.decay_stale_facts().

Unit tests mock the Neo4j AsyncSession so no live database is required.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.fact_decayer import DecayResult, FactDecayer


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_neo_session(counts: tuple[int, int, int, int] = (10, 3, 2, 1)) -> AsyncMock:
    """
    Return a mock Neo4j AsyncSession whose run() returns records with the given
    (decayed_count, pending_promoted_count, deleted_count, orphans_deleted) values.
    Four passes: decay, staleness→pending, delete-below-floor, orphan cleanup.
    """
    decayed, pending_promoted, deleted, orphans = counts

    def _make_result(value: int):
        record = MagicMock()
        record.__getitem__ = lambda self, k: value
        result = AsyncMock()
        result.single = AsyncMock(return_value=record)
        return result

    session = AsyncMock()
    session.run = AsyncMock(side_effect=[
        _make_result(decayed),
        _make_result(pending_promoted),
        _make_result(deleted),
        _make_result(orphans),
    ])
    return session


# ── SemanticMemory.decay_stale_facts ─────────────────────────────────────────


class TestDecayStateFacts:
    @pytest.mark.asyncio
    async def test_returns_correct_counts(self):
        session = _make_neo_session(counts=(15, 5, 3, 2))
        semantic = SemanticMemory()

        decayed, pending_promoted, deleted, orphans = await semantic.decay_stale_facts(
            session, decay_half_life_days=60.0, confidence_floor=0.1
        )

        assert decayed == 15
        assert pending_promoted == 5
        assert deleted == 3
        assert orphans == 2

    @pytest.mark.asyncio
    async def test_runs_four_cypher_queries(self):
        """decay_stale_facts now runs four passes: decay, staleness→pending, delete, orphan."""
        session = _make_neo_session()
        semantic = SemanticMemory()

        await semantic.decay_stale_facts(session)

        assert session.run.call_count == 4

    @pytest.mark.asyncio
    async def test_passes_decay_params_to_first_query(self):
        session = _make_neo_session()
        semantic = SemanticMemory()

        await semantic.decay_stale_facts(
            session, decay_half_life_days=30.0, confidence_floor=0.05
        )

        first_call_kwargs = session.run.call_args_list[0]
        assert first_call_kwargs.kwargs.get("decay_days") == 30.0

    @pytest.mark.asyncio
    async def test_passes_floor_to_third_query(self):
        """Confidence floor is now used in the third pass (delete), not second."""
        session = _make_neo_session()
        semantic = SemanticMemory()

        await semantic.decay_stale_facts(session, confidence_floor=0.05)

        third_call_kwargs = session.run.call_args_list[2]
        assert third_call_kwargs.kwargs.get("confidence_floor") == 0.05

    @pytest.mark.asyncio
    async def test_passes_staleness_threshold_to_second_query(self):
        """Second pass receives the staleness_pending_threshold param."""
        session = _make_neo_session()
        semantic = SemanticMemory()

        await semantic.decay_stale_facts(session, staleness_pending_threshold=0.25)

        second_call_kwargs = session.run.call_args_list[1]
        assert second_call_kwargs.kwargs.get("staleness_threshold") == 0.25

    @pytest.mark.asyncio
    async def test_handles_none_records_gracefully(self):
        """If any query returns no rows (empty graph), counts default to 0."""
        session = AsyncMock()
        empty_result = AsyncMock()
        empty_result.single = AsyncMock(return_value=None)
        session.run = AsyncMock(return_value=empty_result)

        semantic = SemanticMemory()
        decayed, pending_promoted, deleted, orphans = await semantic.decay_stale_facts(session)

        assert decayed == 0
        assert pending_promoted == 0
        assert deleted == 0
        assert orphans == 0


# ── FactDecayer ───────────────────────────────────────────────────────────────


class TestFactDecayer:
    @pytest.mark.asyncio
    async def test_happy_path_returns_result(self):
        session = _make_neo_session(counts=(20, 3, 4, 1))
        semantic = SemanticMemory()
        decayer = FactDecayer(semantic=semantic, half_life_days=60.0, confidence_floor=0.1)

        result = await decayer.run(session)

        assert isinstance(result, DecayResult)
        assert result.skipped is False
        assert result.decayed_count == 20
        assert result.pending_promoted_count == 3
        assert result.deleted_count == 4
        assert result.orphans_deleted == 1

    @pytest.mark.asyncio
    async def test_uses_config_defaults_when_not_overridden(self):
        """FactDecayer reads from settings when half_life_days / confidence_floor are None."""
        from smritikosh.config import settings
        decayer = FactDecayer(semantic=SemanticMemory())
        assert decayer.half_life_days == settings.fact_decay_half_life_days
        assert decayer.confidence_floor == settings.fact_decay_floor

    @pytest.mark.asyncio
    async def test_skips_gracefully_on_neo4j_error(self):
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.decay_stale_facts = AsyncMock(side_effect=RuntimeError("neo4j down"))
        decayer = FactDecayer(semantic=semantic)

        session = AsyncMock()
        result = await decayer.run(session)

        assert result.skipped is True
        assert "neo4j down" in result.skip_reason

    @pytest.mark.asyncio
    async def test_zero_deletes_on_clean_graph(self):
        """A graph with all fresh, high-confidence facts produces no deletes."""
        session = _make_neo_session(counts=(50, 0, 0, 0))
        semantic = SemanticMemory()
        decayer = FactDecayer(semantic=semantic)

        result = await decayer.run(session)

        assert result.deleted_count == 0
        assert result.orphans_deleted == 0
        assert result.decayed_count == 50


# ── DecayResult defaults ──────────────────────────────────────────────────────


class TestDecayResult:
    def test_defaults(self):
        r = DecayResult()
        assert r.decayed_count == 0
        assert r.deleted_count == 0
        assert r.orphans_deleted == 0
        assert r.skipped is False
        assert r.skip_reason == ""

    def test_skipped_result(self):
        r = DecayResult(skipped=True, skip_reason="no decayer")
        assert r.skipped is True
        assert r.skip_reason == "no decayer"
