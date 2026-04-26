"""
Tests for ReinforcementLoop and the apply_delta helper.

Unit tests mock AsyncSession to verify feedback storage and score adjustment.
DB integration tests (gated behind @pytest.mark.db) test the full pipeline
against a live PostgreSQL database.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from smritikosh.db.models import Event, FeedbackType, MemoryFeedback
from smritikosh.processing.reinforcement import (
    NEGATIVE_DELTA,
    POSITIVE_DELTA,
    SCORE_MAX,
    SCORE_MIN,
    ReinforcementLoop,
    apply_delta,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(importance_score: float = 0.8) -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text="test event",
        importance_score=importance_score,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_mock_session(event: Event | None = None) -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.get = AsyncMock(return_value=event)
    session.execute = AsyncMock(return_value=MagicMock())
    return session


@pytest.fixture
def loop() -> ReinforcementLoop:
    return ReinforcementLoop()


# ── apply_delta ───────────────────────────────────────────────────────────────


class TestApplyDelta:
    def test_positive_increases_score(self):
        new = apply_delta(0.8, FeedbackType.POSITIVE)
        assert abs(new - 0.9) < 1e-9

    def test_negative_decreases_score(self):
        new = apply_delta(0.8, FeedbackType.NEGATIVE)
        assert abs(new - 0.7) < 1e-9

    def test_neutral_unchanged(self):
        new = apply_delta(0.8, FeedbackType.NEUTRAL)
        assert abs(new - 0.8) < 1e-9

    def test_positive_capped_at_one(self):
        new = apply_delta(0.95, FeedbackType.POSITIVE)
        assert new == SCORE_MAX

    def test_negative_floored_at_zero(self):
        new = apply_delta(0.05, FeedbackType.NEGATIVE)
        assert new == SCORE_MIN

    def test_positive_at_max_stays_max(self):
        assert apply_delta(1.0, FeedbackType.POSITIVE) == 1.0

    def test_negative_at_min_stays_min(self):
        assert apply_delta(0.0, FeedbackType.NEGATIVE) == 0.0

    def test_exact_delta_magnitude(self):
        assert abs(apply_delta(0.5, FeedbackType.POSITIVE) - (0.5 + POSITIVE_DELTA)) < 1e-9
        assert abs(apply_delta(0.5, FeedbackType.NEGATIVE) - (0.5 - NEGATIVE_DELTA)) < 1e-9

    def test_all_feedback_types_accepted(self):
        for ft in FeedbackType:
            result = apply_delta(0.5, ft)
            assert SCORE_MIN <= result <= SCORE_MAX


# ── ReinforcementLoop.submit ──────────────────────────────────────────────────


class TestReinforcementSubmit:
    @pytest.mark.asyncio
    async def test_raises_when_event_not_found(self, loop):
        session = make_mock_session(event=None)
        with pytest.raises(ValueError, match="not found"):
            await loop.submit(
                session,
                event_id=uuid.uuid4(),
                user_id="u1",
                feedback_type=FeedbackType.POSITIVE,
            )

    @pytest.mark.asyncio
    async def test_returns_feedback_and_new_score(self, loop):
        event = make_event(0.8)
        session = make_mock_session(event)

        feedback, new_score = await loop.submit(
            session,
            event_id=event.id,
            user_id="u1",
            feedback_type=FeedbackType.POSITIVE,
        )

        assert isinstance(feedback, MemoryFeedback)
        assert abs(new_score - 0.9) < 1e-9

    @pytest.mark.asyncio
    async def test_feedback_added_to_session(self, loop):
        event = make_event(0.8)
        session = make_mock_session(event)

        await loop.submit(
            session,
            event_id=event.id,
            user_id="u1",
            feedback_type=FeedbackType.NEGATIVE,
        )

        session.add.assert_called_once()
        added = session.add.call_args[0][0]
        assert isinstance(added, MemoryFeedback)

    @pytest.mark.asyncio
    async def test_feedback_fields_stored_correctly(self, loop):
        event = make_event(0.8)
        session = make_mock_session(event)

        feedback, _ = await loop.submit(
            session,
            event_id=event.id,
            user_id="u1",
            app_id="default",
            feedback_type=FeedbackType.NEGATIVE,
            comment="Not relevant at all",
        )

        assert feedback.event_id == event.id
        assert feedback.user_id == "u1"
        assert feedback.app_id == "default"
        assert feedback.feedback_type == "negative"
        assert feedback.comment == "Not relevant at all"

    @pytest.mark.asyncio
    async def test_execute_called_for_positive(self, loop):
        """importance_score should be updated in DB on POSITIVE feedback."""
        event = make_event(0.5)
        session = make_mock_session(event)

        await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.POSITIVE,
        )

        # 1 UPDATE for importance_score
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_called_for_negative(self, loop):
        event = make_event(0.5)
        session = make_mock_session(event)

        await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.NEGATIVE,
        )

        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_execute_for_neutral(self, loop):
        """NEUTRAL feedback should NOT trigger an importance_score UPDATE."""
        event = make_event(0.5)
        session = make_mock_session(event)

        await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.NEUTRAL,
        )

        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_always_called(self, loop):
        event = make_event(0.8)
        session = make_mock_session(event)

        await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.NEUTRAL,
        )

        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_neutral_returns_unchanged_score(self, loop):
        event = make_event(0.75)
        session = make_mock_session(event)

        _, new_score = await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.NEUTRAL,
        )

        assert abs(new_score - 0.75) < 1e-9

    @pytest.mark.asyncio
    async def test_score_clamped_at_max(self, loop):
        event = make_event(0.99)
        session = make_mock_session(event)

        _, new_score = await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.POSITIVE,
        )

        assert new_score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_at_min(self, loop):
        event = make_event(0.01)
        session = make_mock_session(event)

        _, new_score = await loop.submit(
            session, event_id=event.id, user_id="u1",
            feedback_type=FeedbackType.NEGATIVE,
        )

        assert new_score == 0.0


# ── ReinforcementLoop.get_feedback ────────────────────────────────────────────


class TestGetFeedback:
    @pytest.mark.asyncio
    async def test_returns_list_from_db(self, loop):
        fb1 = MemoryFeedback(
            id=uuid.uuid4(), event_id=uuid.uuid4(), user_id="u1",
            app_id="default", feedback_type="positive",
        )
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [fb1]
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)

        records = await loop.get_feedback(session, fb1.event_id)

        assert records == [fb1]
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_feedback(self, loop):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)

        records = await loop.get_feedback(session, uuid.uuid4())

        assert records == []


# ── ReinforcementLoop.get_user_feedback ───────────────────────────────────────


class TestGetUserFeedback:
    @pytest.mark.asyncio
    async def test_queries_by_user_and_app(self, loop):
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)

        await loop.get_user_feedback(session, user_id="u1", app_id="myapp")

        session.execute.assert_called_once()


# ── DB integration tests ──────────────────────────────────────────────────────


@pytest.mark.db
class TestReinforcementDB:
    """
    How to run:
        docker compose up -d postgres
        pytest tests/test_reinforcement.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings
        from smritikosh.db.models import Base
        from smritikosh.memory.episodic import EpisodicMemory

        engine = create_async_engine(settings.postgres_url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.engine = engine
        self.episodic = EpisodicMemory()
        self.loop = ReinforcementLoop()

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def _store_event(self, session, importance: float = 0.5) -> Event:
        return await self.episodic.store(
            session, user_id="u1", raw_text="test", importance_score=importance
        )

    async def test_positive_feedback_raises_score(self):
        async with self.SessionFactory() as session:
            event = await self._store_event(session, importance=0.5)
            await session.flush()

            _, new_score = await self.loop.submit(
                session, event_id=event.id, user_id="u1",
                feedback_type=FeedbackType.POSITIVE,
            )
            await session.commit()

        assert abs(new_score - 0.6) < 1e-6

    async def test_negative_feedback_lowers_score(self):
        async with self.SessionFactory() as session:
            event = await self._store_event(session, importance=0.5)
            await session.flush()

            _, new_score = await self.loop.submit(
                session, event_id=event.id, user_id="u1",
                feedback_type=FeedbackType.NEGATIVE,
            )
            await session.commit()

        assert abs(new_score - 0.4) < 1e-6

    async def test_feedback_record_persisted(self):
        async with self.SessionFactory() as session:
            event = await self._store_event(session)
            await session.flush()
            event_id = event.id

            feedback, _ = await self.loop.submit(
                session, event_id=event_id, user_id="u1",
                feedback_type=FeedbackType.POSITIVE, comment="Very helpful!",
            )
            await session.commit()

        async with self.SessionFactory() as session:
            records = await self.loop.get_feedback(session, event_id)

        assert len(records) == 1
        assert records[0].feedback_type == "positive"
        assert records[0].comment == "Very helpful!"

    async def test_multiple_feedback_accumulated(self):
        async with self.SessionFactory() as session:
            event = await self._store_event(session, importance=0.5)
            await session.flush()
            event_id = event.id

            await self.loop.submit(session, event_id=event_id, user_id="u1",
                                   feedback_type=FeedbackType.POSITIVE)
            await self.loop.submit(session, event_id=event_id, user_id="u1",
                                   feedback_type=FeedbackType.POSITIVE)
            await session.commit()

        async with self.SessionFactory() as session:
            records = await self.loop.get_feedback(session, event_id)

        assert len(records) == 2
