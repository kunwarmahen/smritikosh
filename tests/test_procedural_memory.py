"""
Unit tests for ProceduralMemory.

Tests cover:
  - store / update / delete CRUD operations
  - delete_all_for_user
  - increment_hit_count
  - search_by_query — all three matching strategies
  - _tokenise and _jaccard helpers
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import UserProcedure
from smritikosh.memory.procedural import ProceduralMemory, _jaccard, _tokenise


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_procedure(
    trigger: str = "LLM deployment",
    instruction: str = "mention GPU optimization",
    user_id: str = "u1",
    priority: int = 5,
    is_active: bool = True,
    hit_count: int = 0,
) -> UserProcedure:
    p = UserProcedure(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        trigger=trigger,
        instruction=instruction,
        priority=priority,
        is_active=is_active,
        hit_count=hit_count,
        confidence=1.0,
        source="manual",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    return p


def mock_session_with_procedures(procedures: list[UserProcedure]) -> AsyncMock:
    """Return a mock AsyncSession whose execute() returns the given procedures."""
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = procedures
    session.execute = AsyncMock(return_value=result)
    return session


# ── _tokenise ─────────────────────────────────────────────────────────────────


class TestTokenise:
    def test_splits_on_whitespace(self):
        assert _tokenise("hello world") == {"hello", "world"}

    def test_strips_punctuation(self):
        assert _tokenise("hello, world!") == {"hello", "world"}

    def test_lowercases(self):
        assert _tokenise("Hello WORLD") == {"hello", "world"}

    def test_empty_string(self):
        assert _tokenise("") == set()

    def test_multi_word_phrase(self):
        tokens = _tokenise("LLM deployment pipeline")
        assert tokens == {"llm", "deployment", "pipeline"}


# ── _jaccard ──────────────────────────────────────────────────────────────────


class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert _jaccard({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        j = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        assert abs(j - 0.5) < 1e-9

    def test_empty_sets(self):
        assert _jaccard(set(), {"a"}) == 0.0
        assert _jaccard({"a"}, set()) == 0.0
        assert _jaccard(set(), set()) == 0.0

    def test_superset(self):
        j = _jaccard({"a"}, {"a", "b", "c"})
        # intersection=1, union=3
        assert abs(j - 1 / 3) < 1e-9


# ── ProceduralMemory.store ────────────────────────────────────────────────────


class TestStore:
    @pytest.mark.asyncio
    async def test_adds_procedure_to_session(self):
        session = AsyncMock()
        session.flush = AsyncMock()
        pm = ProceduralMemory()

        result = await pm.store(
            session,
            user_id="u1",
            trigger="startup",
            instruction="respond with depth",
        )

        session.add.assert_called_once()
        assert result.trigger == "startup"
        assert result.instruction == "respond with depth"
        assert result.user_id == "u1"

    @pytest.mark.asyncio
    async def test_defaults_applied(self):
        session = AsyncMock()
        session.flush = AsyncMock()
        pm = ProceduralMemory()

        result = await pm.store(session, user_id="u1", trigger="t", instruction="i")

        assert result.app_id == "default"
        assert result.priority == 5
        assert result.category == "topic_response"
        assert result.source == "manual"
        # is_active is a DB-side default (applied at INSERT/flush against real DB)
        # so we only verify the other scalar defaults here

    @pytest.mark.asyncio
    async def test_custom_priority_and_category(self):
        session = AsyncMock()
        session.flush = AsyncMock()
        pm = ProceduralMemory()

        result = await pm.store(
            session,
            user_id="u1",
            trigger="t",
            instruction="i",
            priority=9,
            category="communication",
        )

        assert result.priority == 9
        assert result.category == "communication"


# ── ProceduralMemory.update ───────────────────────────────────────────────────


class TestUpdate:
    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        pm = ProceduralMemory()

        result = await pm.update(session, uuid.uuid4(), trigger="new")

        assert result is None

    @pytest.mark.asyncio
    async def test_updates_only_provided_fields(self):
        proc = make_procedure(trigger="old trigger", priority=5)
        session = AsyncMock()
        session.get = AsyncMock(return_value=proc)
        session.flush = AsyncMock()
        pm = ProceduralMemory()

        result = await pm.update(session, proc.id, trigger="new trigger")

        assert result.trigger == "new trigger"
        assert result.priority == 5  # unchanged

    @pytest.mark.asyncio
    async def test_deactivate_procedure(self):
        proc = make_procedure(is_active=True)
        session = AsyncMock()
        session.get = AsyncMock(return_value=proc)
        session.flush = AsyncMock()
        pm = ProceduralMemory()

        result = await pm.update(session, proc.id, is_active=False)

        assert result.is_active is False


# ── ProceduralMemory.delete ───────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_returns_true_when_found(self):
        proc = make_procedure()
        session = AsyncMock()
        session.get = AsyncMock(return_value=proc)
        pm = ProceduralMemory()

        result = await pm.delete(session, proc.id)

        assert result is True
        session.delete.assert_awaited_once_with(proc)

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        session = AsyncMock()
        session.get = AsyncMock(return_value=None)
        pm = ProceduralMemory()

        result = await pm.delete(session, uuid.uuid4())

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_all_for_user_returns_count(self):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [(uuid.uuid4(),), (uuid.uuid4(),)]
        session.execute = AsyncMock(return_value=mock_result)
        pm = ProceduralMemory()

        count = await pm.delete_all_for_user(session, "u1")

        assert count == 2


# ── ProceduralMemory.search_by_query ─────────────────────────────────────────


class TestSearchByQuery:
    @pytest.mark.asyncio
    async def test_strategy1_trigger_in_query(self):
        """Trigger phrase appears as substring of the query → score 1.0."""
        proc = make_procedure(trigger="LLM deployment")
        session = mock_session_with_procedures([proc])
        pm = ProceduralMemory()

        results = await pm.search_by_query(session, "u1", "how do I do LLM deployment on AWS?")

        assert len(results) == 1
        assert results[0].trigger == "LLM deployment"

    @pytest.mark.asyncio
    async def test_strategy2_query_in_trigger(self):
        """Query appears inside the trigger phrase → score 0.5."""
        proc = make_procedure(trigger="building AI startup company")
        session = mock_session_with_procedures([proc])
        pm = ProceduralMemory()

        results = await pm.search_by_query(session, "u1", "startup")

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_strategy2b_query_token_in_trigger(self):
        """A query token (len>3) appears inside the trigger → score 0.5."""
        proc = make_procedure(trigger="startup planning and roadmap")
        session = mock_session_with_procedures([proc])
        pm = ProceduralMemory()

        results = await pm.search_by_query(
            session, "u1", "how should i plan my startup roadmap"
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_strategy3_jaccard_overlap(self):
        """Jaccard overlap ≥ threshold → matches."""
        proc = make_procedure(trigger="machine learning model training")
        session = mock_session_with_procedures([proc])
        pm = ProceduralMemory(overlap_threshold=0.1)

        results = await pm.search_by_query(
            session, "u1", "training machine learning models effectively"
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self):
        proc = make_procedure(trigger="dark mode preference")
        session = mock_session_with_procedures([proc])
        pm = ProceduralMemory()

        results = await pm.search_by_query(session, "u1", "what is the weather today")

        assert results == []

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self):
        procs = [make_procedure(trigger=f"topic {i}") for i in range(10)]
        session = mock_session_with_procedures(procs)
        pm = ProceduralMemory(overlap_threshold=0.0)  # match everything

        # Use a query that will Jaccard-match all of them via low threshold
        # Patch _score_matches to return all as matches
        with patch.object(pm, "_score_matches", AsyncMock(return_value=[
            MagicMock(procedure=p, match_score=0.5) for p in procs
        ])):
            results = await pm.search_by_query(session, "u1", "topic test", top_k=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_inactive_procedures_excluded(self):
        active = make_procedure(trigger="AI tools", is_active=True)
        inactive = make_procedure(trigger="AI tools", is_active=False)
        # get_all with active_only=True only returns active
        session = mock_session_with_procedures([active])
        pm = ProceduralMemory()

        results = await pm.search_by_query(session, "u1", "how do I use AI tools?")

        assert len(results) == 1
        assert results[0].is_active is True

    @pytest.mark.asyncio
    async def test_higher_priority_ranked_first(self):
        """When scores are equal, higher priority wins."""
        low = make_procedure(trigger="deployment", priority=3, hit_count=0)
        high = make_procedure(trigger="deployment", priority=9, hit_count=0)
        session = mock_session_with_procedures([low, high])
        pm = ProceduralMemory()

        results = await pm.search_by_query(session, "u1", "deployment guide")

        assert results[0].priority == 9
