"""
Tests for NarrativeMemory.

Unit tests mock AsyncSession to verify write and read behaviour.
DB integration tests (gated behind @pytest.mark.db) verify the
recursive CTE chain traversal against a live PostgreSQL database.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.db.models import Event, MemoryLink, RelationType
from smritikosh.memory.narrative import NarrativeMemory


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


def make_event(raw_text: str = "test event") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text=raw_text,
        importance_score=0.8,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_link(from_id: uuid.UUID, to_id: uuid.UUID, rel: str = "preceded") -> MemoryLink:
    link = MemoryLink(
        id=uuid.uuid4(),
        from_event_id=from_id,
        to_event_id=to_id,
        relation_type=rel,
    )
    return link


@pytest.fixture
def narrative() -> NarrativeMemory:
    return NarrativeMemory()


# ── create_link ───────────────────────────────────────────────────────────────


class TestCreateLink:
    @pytest.mark.asyncio
    async def test_adds_to_session(self, narrative):
        session = make_mock_session()
        from_id, to_id = uuid.uuid4(), uuid.uuid4()

        link = await narrative.create_link(
            session,
            from_event_id=from_id,
            to_event_id=to_id,
            relation_type=RelationType.CAUSED,
        )

        session.add.assert_called_once()
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_memory_link(self, narrative):
        session = make_mock_session()

        link = await narrative.create_link(
            session,
            from_event_id=uuid.uuid4(),
            to_event_id=uuid.uuid4(),
            relation_type=RelationType.PRECEDED,
        )

        assert isinstance(link, MemoryLink)

    @pytest.mark.asyncio
    async def test_stores_correct_ids_and_relation(self, narrative):
        session = make_mock_session()
        from_id, to_id = uuid.uuid4(), uuid.uuid4()

        link = await narrative.create_link(
            session,
            from_event_id=from_id,
            to_event_id=to_id,
            relation_type=RelationType.RELATED,
        )

        assert link.from_event_id == from_id
        assert link.to_event_id == to_id
        assert link.relation_type == "related"

    @pytest.mark.asyncio
    async def test_all_relation_types_accepted(self, narrative):
        session = make_mock_session()
        for rel in RelationType:
            session.add.reset_mock()
            await narrative.create_link(
                session,
                from_event_id=uuid.uuid4(),
                to_event_id=uuid.uuid4(),
                relation_type=rel,
            )
            session.add.assert_called_once()


# ── get_chain_forward — empty result ─────────────────────────────────────────


class TestGetChainForwardEmpty:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_links(self, narrative):
        session = make_mock_session()
        # CTE returns no rows
        cte_result = MagicMock()
        cte_result.all.return_value = []
        session.execute = AsyncMock(return_value=cte_result)

        result = await narrative.get_chain_forward(session, uuid.uuid4())

        assert result == []

    @pytest.mark.asyncio
    async def test_only_one_db_call_when_no_chain(self, narrative):
        session = make_mock_session()
        cte_result = MagicMock()
        cte_result.all.return_value = []
        session.execute = AsyncMock(return_value=cte_result)

        await narrative.get_chain_forward(session, uuid.uuid4())

        # Only the CTE query fires; the link-loading query is skipped
        assert session.execute.call_count == 1


# ── get_chain_backward — empty result ────────────────────────────────────────


class TestGetChainBackwardEmpty:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_links(self, narrative):
        session = make_mock_session()
        cte_result = MagicMock()
        cte_result.all.return_value = []
        session.execute = AsyncMock(return_value=cte_result)

        result = await narrative.get_chain_backward(session, uuid.uuid4())

        assert result == []


# ── get_related_events ────────────────────────────────────────────────────────


class TestGetRelatedEvents:
    @pytest.mark.asyncio
    async def test_returns_events_from_query(self, narrative):
        session = make_mock_session()
        events = [make_event("related a"), make_event("related b")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = events
        session.execute = AsyncMock(return_value=mock_result)

        result = await narrative.get_related_events(session, uuid.uuid4(), "u1", "default")

        assert result == events
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_related(self, narrative):
        session = make_mock_session()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=mock_result)

        result = await narrative.get_related_events(session, uuid.uuid4(), "u1", "default")

        assert result == []


# ── DB integration tests ──────────────────────────────────────────────────────


@pytest.mark.db
class TestNarrativeMemoryDB:
    """
    How to run:
        docker compose up -d postgres
        pytest tests/test_narrative_memory.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings
        from smritikosh.db.models import Base

        engine = create_async_engine(settings.postgres_url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.engine = engine
        self.narrative = NarrativeMemory()

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def _store_event(self, session, raw_text: str) -> Event:
        from smritikosh.memory.episodic import EpisodicMemory
        episodic = EpisodicMemory()
        return await episodic.store(session, user_id="u1", raw_text=raw_text)

    async def test_create_and_retrieve_link(self):
        async with self.SessionFactory() as session:
            e1 = await self._store_event(session, "User started AI startup")
            e2 = await self._store_event(session, "User hired first engineer")
            await session.flush()

            link = await self.narrative.create_link(
                session,
                from_event_id=e1.id,
                to_event_id=e2.id,
                relation_type=RelationType.CAUSED,
            )
            await session.commit()
            link_id = link.id

        async with self.SessionFactory() as session:
            from sqlalchemy import select as sa_select
            found = (await session.execute(
                sa_select(MemoryLink).where(MemoryLink.id == link_id)
            )).scalar_one_or_none()
            assert found is not None
            assert found.relation_type == "caused"

    async def test_get_chain_forward(self):
        async with self.SessionFactory() as session:
            e1 = await self._store_event(session, "Startup founded")
            e2 = await self._store_event(session, "Engineers hired")
            e3 = await self._store_event(session, "Product launched")
            await session.flush()

            await self.narrative.create_link(session, from_event_id=e1.id, to_event_id=e2.id, relation_type=RelationType.CAUSED)
            await self.narrative.create_link(session, from_event_id=e2.id, to_event_id=e3.id, relation_type=RelationType.PRECEDED)
            await session.commit()
            e1_id = e1.id

        async with self.SessionFactory() as session:
            chain = await self.narrative.get_chain_forward(session, e1_id, max_depth=5)

        assert len(chain) == 2
        assert chain[0].from_event_id == e1_id

    async def test_get_chain_backward(self):
        async with self.SessionFactory() as session:
            e1 = await self._store_event(session, "Startup founded")
            e2 = await self._store_event(session, "Product launched")
            await session.flush()

            await self.narrative.create_link(session, from_event_id=e1.id, to_event_id=e2.id, relation_type=RelationType.PRECEDED)
            await session.commit()
            e2_id = e2.id

        async with self.SessionFactory() as session:
            chain = await self.narrative.get_chain_backward(session, e2_id)

        assert len(chain) == 1
        assert chain[0].to_event_id == e2_id

    async def test_get_related_events(self):
        async with self.SessionFactory() as session:
            e1 = await self._store_event(session, "Event A")
            e2 = await self._store_event(session, "Event B")
            e3 = await self._store_event(session, "Event C — unrelated")
            await session.flush()

            await self.narrative.create_link(session, from_event_id=e1.id, to_event_id=e2.id, relation_type=RelationType.RELATED)
            await session.commit()
            e1_id, e2_id, e3_id = e1.id, e2.id, e3.id

        async with self.SessionFactory() as session:
            related = await self.narrative.get_related_events(session, e1_id)

        related_ids = {e.id for e in related}
        assert e2_id in related_ids
        assert e3_id not in related_ids
        assert e1_id not in related_ids
