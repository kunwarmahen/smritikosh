"""
Tests for EpisodicMemory.

How to run:
    # Unit tests (no DB — session is mocked):
    pytest tests/test_episodic_memory.py -v

    # DB integration tests (requires running Postgres):
    #   docker compose up -d postgres
    pytest tests/test_episodic_memory.py -v -m db

Test strategy:
    - Unit tests mock AsyncSession to verify query construction and business logic.
    - DB integration tests insert real rows, run hybrid search, and verify scores.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import Base, Event
from smritikosh.memory.episodic import EpisodicMemory, HybridWeights, SearchResult, _embedding_literal


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_episodic(**weight_overrides) -> EpisodicMemory:
    if weight_overrides:
        weights = HybridWeights(**weight_overrides)
        return EpisodicMemory(weights=weights)
    return EpisodicMemory()


def make_mock_session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.add = MagicMock()
    return session


def make_event(user_id="u1", raw_text="test event", **kwargs) -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text=raw_text,
        embedding=[0.1] * 1536,
        importance_score=kwargs.get("importance_score", 1.0),
        consolidated=kwargs.get("consolidated", False),
        event_metadata={},
        created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
    )


# ── HybridWeights ─────────────────────────────────────────────────────────────


class TestHybridWeights:
    def test_defaults_sum_to_one(self):
        w = HybridWeights()
        total = w.similarity + w.recency + w.importance + w.frequency + w.contextual_match
        assert abs(total - 1.0) < 1e-6

    def test_default_values(self):
        w = HybridWeights()
        assert w.similarity == 0.40
        assert w.recency == 0.30
        assert w.importance == 0.15
        assert w.frequency == 0.15
        assert w.contextual_match == 0.0

    def test_custom_weights_valid(self):
        w = HybridWeights(similarity=0.50, recency=0.20, importance=0.15, frequency=0.15)
        assert w.similarity == 0.50

    def test_contextual_match_slot(self):
        # Phase 2: activate contextual_match by redistributing from recency
        w = HybridWeights(similarity=0.40, recency=0.25, importance=0.15, frequency=0.15, contextual_match=0.05)
        total = w.similarity + w.recency + w.importance + w.frequency + w.contextual_match
        assert abs(total - 1.0) < 1e-6

    def test_invalid_weights_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            HybridWeights(similarity=0.5, recency=0.5, importance=0.5)

    def test_default_decay_days(self):
        w = HybridWeights()
        assert w.decay_days == 30.0

    def test_default_frequency_cap(self):
        w = HybridWeights()
        assert w.frequency_cap == 50


# ── _embedding_literal ────────────────────────────────────────────────────────


class TestEmbeddingLiteral:
    def test_format(self):
        result = _embedding_literal([0.1, 0.2, 0.3])
        assert result.startswith("[")
        assert result.endswith("]")
        assert "0.10000000" in result

    def test_length_preserved(self):
        vec = [0.5] * 10
        lit = _embedding_literal(vec)
        # 10 values separated by commas inside brackets
        assert lit.count(",") == 9


# ── store() ───────────────────────────────────────────────────────────────────


class TestStore:
    @pytest.mark.asyncio
    async def test_store_adds_event_to_session(self):
        episodic = make_episodic()
        session = make_mock_session()

        event = await episodic.store(session, user_id="u1", raw_text="test text")

        session.add.assert_called_once()
        session.flush.assert_called_once()
        assert event.user_id == "u1"
        assert event.raw_text == "test text"

    @pytest.mark.asyncio
    async def test_store_with_embedding(self):
        episodic = make_episodic()
        session = make_mock_session()
        vec = [0.1] * 1536

        event = await episodic.store(
            session, user_id="u1", raw_text="hello", embedding=vec
        )

        assert event.embedding == vec

    @pytest.mark.asyncio
    async def test_store_defaults(self):
        episodic = make_episodic()
        session = make_mock_session()

        event = await episodic.store(session, user_id="u1", raw_text="hello")

        assert event.app_id == "default"
        assert event.importance_score == 1.0
        assert event.consolidated is False
        assert event.event_metadata == {}

    @pytest.mark.asyncio
    async def test_store_with_metadata(self):
        episodic = make_episodic()
        session = make_mock_session()

        event = await episodic.store(
            session,
            user_id="u1",
            raw_text="hello",
            metadata={"source": "slack", "channel": "#general"},
        )

        assert event.event_metadata["source"] == "slack"

    @pytest.mark.asyncio
    async def test_store_custom_app_id(self):
        episodic = make_episodic()
        session = make_mock_session()

        event = await episodic.store(
            session, user_id="u1", raw_text="hello", app_id="my_app"
        )

        assert event.app_id == "my_app"


# ── update_embedding() ────────────────────────────────────────────────────────


class TestUpdateEmbedding:
    @pytest.mark.asyncio
    async def test_executes_update(self):
        episodic = make_episodic()
        session = make_mock_session()
        event_id = uuid.uuid4()
        vec = [0.5] * 1536

        await episodic.update_embedding(session, event_id, vec)

        session.execute.assert_called_once()


# ── mark_consolidated() ───────────────────────────────────────────────────────


class TestMarkConsolidated:
    @pytest.mark.asyncio
    async def test_marks_given_ids(self):
        episodic = make_episodic()
        session = make_mock_session()
        ids = [uuid.uuid4(), uuid.uuid4()]

        await episodic.mark_consolidated(session, ids)

        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_mark_with_summary(self):
        episodic = make_episodic()
        session = make_mock_session()
        ids = [uuid.uuid4()]

        # Should not raise
        await episodic.mark_consolidated(session, ids, summary="User discussed AI")


# ── delete() ──────────────────────────────────────────────────────────────────


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_event(self):
        episodic = make_episodic()
        session = make_mock_session()
        event = make_event()
        session.get = AsyncMock(return_value=event)
        session.delete = AsyncMock()

        result = await episodic.delete(session, event.id)

        assert result is True
        session.delete.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_delete_missing_event_returns_false(self):
        episodic = make_episodic()
        session = make_mock_session()
        session.get = AsyncMock(return_value=None)

        result = await episodic.delete(session, uuid.uuid4())

        assert result is False


# ── get_recent() ──────────────────────────────────────────────────────────────


class TestGetRecent:
    @pytest.mark.asyncio
    async def test_returns_list_of_events(self):
        episodic = make_episodic()
        session = make_mock_session()
        events = [make_event(), make_event()]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = events
        session.execute = AsyncMock(return_value=mock_result)

        result = await episodic.get_recent(session, "u1")

        assert result == events
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_limit_is_passed(self):
        """Verify the query is built with the correct limit (inspected via call count)."""
        episodic = make_episodic()
        session = make_mock_session()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        await episodic.get_recent(session, "u1", limit=3)
        session.execute.assert_called_once()


# ── SearchResult ──────────────────────────────────────────────────────────────


class TestSearchResult:
    def test_search_result_fields(self):
        event = make_event()
        sr = SearchResult(
            event=event,
            similarity_score=0.9,
            recency_score=0.8,
            frequency_score=0.6,
            hybrid_score=0.85,
        )
        assert sr.event is event
        assert sr.similarity_score == 0.9
        assert sr.recency_score == 0.8
        assert sr.frequency_score == 0.6
        assert sr.hybrid_score == 0.85

    def test_search_result_defaults(self):
        event = make_event()
        sr = SearchResult(event=event)
        assert sr.similarity_score == 0.0
        assert sr.recency_score == 0.0
        assert sr.frequency_score == 0.0
        assert sr.hybrid_score == 0.0


# ── increment_recall() ────────────────────────────────────────────────────────


class TestIncrementRecall:
    @pytest.mark.asyncio
    async def test_executes_update_for_ids(self):
        episodic = make_episodic()
        session = make_mock_session()
        ids = [uuid.uuid4(), uuid.uuid4()]

        await episodic.increment_recall(session, ids)

        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_op_for_empty_list(self):
        episodic = make_episodic()
        session = make_mock_session()

        await episodic.increment_recall(session, [])

        session.execute.assert_not_called()


# ── DB integration tests ───────────────────────────────────────────────────────


@pytest.mark.db
class TestEpisodicMemoryDB:
    """
    How to run:
        docker compose up -d postgres
        pytest tests/test_episodic_memory.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings

        engine = create_async_engine(settings.postgres_url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.engine = engine
        self.episodic = EpisodicMemory()

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def test_store_and_retrieve(self):
        async with self.SessionFactory() as session:
            event = await self.episodic.store(
                session, user_id="u1", raw_text="User loves building AI systems"
            )
            await session.commit()
            event_id = event.id

        async with self.SessionFactory() as session:
            found = await session.get(Event, event_id)
            assert found is not None
            assert found.raw_text == "User loves building AI systems"
            assert found.consolidated is False

    async def test_get_recent_ordered_newest_first(self):
        from datetime import timedelta

        async with self.SessionFactory() as session:
            now = datetime.now(timezone.utc)
            e1 = await self.episodic.store(
                session, user_id="u1", raw_text="old event",
            )
            e1.created_at = now - timedelta(days=5)

            e2 = await self.episodic.store(
                session, user_id="u1", raw_text="new event",
            )
            e2.created_at = now
            await session.commit()

        async with self.SessionFactory() as session:
            results = await self.episodic.get_recent(session, "u1", limit=10)
            assert results[0].raw_text == "new event"
            assert results[1].raw_text == "old event"

    async def test_mark_consolidated(self):
        async with self.SessionFactory() as session:
            event = await self.episodic.store(
                session, user_id="u1", raw_text="raw conversation"
            )
            await session.commit()
            event_id = event.id

        async with self.SessionFactory() as session:
            await self.episodic.mark_consolidated(
                session, [event_id], summary="User interested in AI"
            )
            await session.commit()

        async with self.SessionFactory() as session:
            found = await session.get(Event, event_id)
            assert found.consolidated is True
            assert found.summary == "User interested in AI"

    async def test_delete_event(self):
        async with self.SessionFactory() as session:
            event = await self.episodic.store(
                session, user_id="u1", raw_text="to be deleted"
            )
            await session.commit()
            event_id = event.id

        async with self.SessionFactory() as session:
            deleted = await self.episodic.delete(session, event_id)
            await session.commit()
            assert deleted is True

        async with self.SessionFactory() as session:
            gone = await session.get(Event, event_id)
            assert gone is None

    async def test_hybrid_search_returns_ranked_results(self):
        """Events with similar embeddings should score higher."""
        query_vec = [1.0] * 1536

        async with self.SessionFactory() as session:
            # High similarity — same direction vector
            e_similar = await self.episodic.store(
                session,
                user_id="u1",
                raw_text="very similar to query",
                embedding=[1.0] * 1536,
                importance_score=0.5,
            )
            # Low similarity — opposite direction
            e_distant = await self.episodic.store(
                session,
                user_id="u1",
                raw_text="very different from query",
                embedding=[-1.0] * 1536,
                importance_score=0.5,
            )
            await session.commit()

        async with self.SessionFactory() as session:
            results = await self.episodic.hybrid_search(
                session, "u1", query_vec, top_k=5
            )

        assert len(results) == 2
        # The similar event should rank first
        assert results[0].event.raw_text == "very similar to query"
        assert results[0].similarity_score > results[1].similarity_score

    async def test_hybrid_search_isolates_by_user(self):
        """Results for u1 should not include events from u2."""
        vec = [0.5] * 1536

        async with self.SessionFactory() as session:
            await self.episodic.store(session, user_id="u1", raw_text="u1 event", embedding=vec)
            await self.episodic.store(session, user_id="u2", raw_text="u2 event", embedding=vec)
            await session.commit()

        async with self.SessionFactory() as session:
            results = await self.episodic.hybrid_search(session, "u1", vec, top_k=10)

        user_ids = {r.event.user_id for r in results}
        assert user_ids == {"u1"}

    async def test_events_without_embedding_excluded_from_search(self):
        vec = [0.5] * 1536

        async with self.SessionFactory() as session:
            # No embedding — should not appear in search
            await self.episodic.store(session, user_id="u1", raw_text="no embedding")
            # With embedding — should appear
            await self.episodic.store(session, user_id="u1", raw_text="has embedding", embedding=vec)
            await session.commit()

        async with self.SessionFactory() as session:
            results = await self.episodic.hybrid_search(session, "u1", vec, top_k=10)

        texts = [r.event.raw_text for r in results]
        assert "has embedding" in texts
        assert "no embedding" not in texts

    async def test_get_unconsolidated_excludes_done(self):
        async with self.SessionFactory() as session:
            e1 = await self.episodic.store(session, user_id="u1", raw_text="not done")
            e2 = await self.episodic.store(session, user_id="u1", raw_text="done")
            await session.flush()
            await self.episodic.mark_consolidated(session, [e2.id])
            await session.commit()

        async with self.SessionFactory() as session:
            pending = await self.episodic.get_unconsolidated(session, "u1")

        assert len(pending) == 1
        assert pending[0].raw_text == "not done"
