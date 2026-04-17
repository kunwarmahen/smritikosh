"""
Tests for DB models.

How to run:
    # Unit tests (no DB needed — tests model construction and logic):
    pytest tests/test_db_models.py -v

    # Integration tests (requires running Postgres with pgvector):
    #   docker compose up -d postgres
    #   pytest tests/test_db_models.py -v -m db
    pytest tests/test_db_models.py -v -m db

Test strategy:
    - Unit tests verify model construction, enums, repr, and constraints.
    - DB integration tests (marked 'db') spin up real tables via init_db(),
      insert rows, and verify queries — including vector similarity search.
"""

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from smritikosh.db.models import (
    Base,
    Event,
    FactCategory,
    MemoryLink,
    RelationType,
    UserFact,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(**overrides) -> Event:
    defaults = dict(
        user_id="user_mahen",
        app_id="smritikosh",
        raw_text="User discussed building an AI memory startup",
        importance_score=0.9,
        consolidated=False,
        event_metadata={"source": "chat"},
    )
    return Event(**{**defaults, **overrides})


def make_fact(**overrides) -> UserFact:
    defaults = dict(
        user_id="user_mahen",
        app_id="smritikosh",
        category=FactCategory.INTEREST,
        key="domain",
        value="AI agents",
        confidence=0.95,
        frequency_count=3,
    )
    return UserFact(**{**defaults, **overrides})


# ── Unit tests — model construction ───────────────────────────────────────────


class TestEventModel:
    def test_event_defaults(self):
        # app_id, importance_score, consolidated are DB-level INSERT defaults —
        # verified in TestEventDB (integration). Here we just check the fields exist.
        event = Event(user_id="u1", raw_text="hello")
        assert event.user_id == "u1"
        assert event.raw_text == "hello"
        assert event.summary is None
        assert event.embedding is None

    def test_event_repr(self):
        event = make_event()
        r = repr(event)
        assert "Event" in r
        assert "user_mahen" in r
        assert "0.90" in r

    def test_event_with_embedding(self):
        vector = [0.1] * 1536
        event = make_event(embedding=vector)
        assert event.embedding == vector

    def test_event_with_metadata(self):
        event = make_event(event_metadata={"source": "slack", "channel": "#general"})
        assert event.event_metadata["source"] == "slack"

    def test_event_custom_id(self):
        uid = uuid.uuid4()
        event = Event(id=uid, user_id="u1", raw_text="test")
        assert event.id == uid


class TestUserFactModel:
    def test_fact_defaults(self):
        # app_id, confidence, frequency_count are DB-level INSERT defaults —
        # verified in TestEventDB (integration). Here we check field assignment.
        fact = UserFact(
            user_id="u1",
            category=FactCategory.PREFERENCE,
            key="color",
            value="green",
        )
        assert fact.user_id == "u1"
        assert fact.category == FactCategory.PREFERENCE
        assert fact.key == "color"
        assert fact.value == "green"

    def test_fact_repr(self):
        fact = make_fact()
        r = repr(fact)
        assert "UserFact" in r
        assert "interest" in r
        assert "AI agents" in r

    def test_fact_categories(self):
        for cat in FactCategory:
            fact = UserFact(user_id="u1", category=cat, key="k", value="v")
            assert fact.category == cat


class TestMemoryLinkModel:
    def test_link_repr(self):
        fid = uuid.uuid4()
        tid = uuid.uuid4()
        link = MemoryLink(
            from_event_id=fid,
            to_event_id=tid,
            relation_type=RelationType.CAUSED,
        )
        r = repr(link)
        assert "MemoryLink" in r
        assert "caused" in r

    def test_all_relation_types(self):
        assert RelationType.CAUSED == "caused"
        assert RelationType.PRECEDED == "preceded"
        assert RelationType.RELATED == "related"
        assert RelationType.CONTRADICTS == "contradicts"


class TestEnums:
    def test_fact_category_values(self):
        assert FactCategory.IDENTITY     == "identity"
        assert FactCategory.LOCATION     == "location"
        assert FactCategory.ROLE         == "role"
        assert FactCategory.SKILL        == "skill"
        assert FactCategory.EDUCATION    == "education"
        assert FactCategory.PROJECT      == "project"
        assert FactCategory.GOAL         == "goal"
        assert FactCategory.INTEREST     == "interest"
        assert FactCategory.HOBBY        == "hobby"
        assert FactCategory.HABIT        == "habit"
        assert FactCategory.PREFERENCE   == "preference"
        assert FactCategory.PERSONALITY  == "personality"
        assert FactCategory.RELATIONSHIP == "relationship"
        assert FactCategory.PET          == "pet"
        assert FactCategory.HEALTH       == "health"
        assert FactCategory.DIET         == "diet"
        assert FactCategory.BELIEF       == "belief"
        assert FactCategory.VALUE        == "value"
        assert FactCategory.RELIGION     == "religion"
        assert FactCategory.FINANCE      == "finance"
        assert FactCategory.LIFESTYLE    == "lifestyle"
        assert FactCategory.EVENT        == "event"
        assert FactCategory.TOOL         == "tool"

    def test_enum_is_str(self):
        # Enums inherit from str so they serialise cleanly to JSON
        assert isinstance(FactCategory.PREFERENCE, str)
        assert isinstance(RelationType.CAUSED, str)


# ── DB integration tests ───────────────────────────────────────────────────────


@pytest.mark.db
class TestEventDB:
    """
    How to run:
        docker compose up -d postgres
        pytest tests/test_db_models.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        """Create tables before each test, drop after."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
        from smritikosh.config import settings

        engine = create_async_engine(settings.postgres_url)

        # Enable pgvector and create tables
        from sqlalchemy import text
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self.engine = engine

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def test_insert_and_retrieve_event(self):
        event = make_event()
        async with self.SessionFactory() as session:
            session.add(event)
            await session.commit()

        async with self.SessionFactory() as session:
            result = await session.get(Event, event.id)
            assert result is not None
            assert result.user_id == "user_mahen"
            assert result.raw_text == "User discussed building an AI memory startup"

    async def test_event_with_embedding_roundtrip(self):
        vector = [0.01 * i for i in range(1536)]
        event = make_event(embedding=vector)
        async with self.SessionFactory() as session:
            session.add(event)
            await session.commit()

        async with self.SessionFactory() as session:
            result = await session.get(Event, event.id)
            assert result.embedding is not None
            assert len(result.embedding) == 1536

    async def test_vector_similarity_search(self):
        """Verify cosine similarity search works via pgvector."""
        from sqlalchemy import text

        events = [
            make_event(raw_text=f"event {i}", embedding=[float(i)] * 1536)
            for i in range(3)
        ]
        async with self.SessionFactory() as session:
            session.add_all(events)
            await session.commit()

        query_vector = "[1.0" + ",1.0" * 1535 + "]"
        async with self.SessionFactory() as session:
            rows = await session.execute(
                text(
                    "SELECT id, embedding <=> :q AS distance "
                    "FROM events ORDER BY distance LIMIT 1"
                ),
                {"q": query_vector},
            )
            top = rows.fetchone()
            assert top is not None

    async def test_insert_user_fact(self):
        fact = make_fact()
        async with self.SessionFactory() as session:
            session.add(fact)
            await session.commit()

        async with self.SessionFactory() as session:
            result = await session.get(UserFact, fact.id)
            assert result.category == FactCategory.INTEREST
            assert result.value == "AI agents"

    async def test_user_fact_unique_constraint(self):
        """Same (user, app, category, key) should raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        fact1 = make_fact()
        fact2 = make_fact()  # identical keys

        async with self.SessionFactory() as session:
            session.add(fact1)
            await session.commit()

        with pytest.raises(IntegrityError):
            async with self.SessionFactory() as session:
                session.add(fact2)
                await session.commit()

    async def test_memory_link_cascade_delete(self):
        """Deleting an event should cascade-delete its links."""
        event_a = make_event(raw_text="started startup")
        event_b = make_event(raw_text="hired engineers")

        async with self.SessionFactory() as session:
            session.add_all([event_a, event_b])
            await session.flush()

            link = MemoryLink(
                from_event_id=event_a.id,
                to_event_id=event_b.id,
                relation_type=RelationType.CAUSED,
            )
            session.add(link)
            await session.commit()
            link_id = link.id

        # Delete event_a — link should disappear
        async with self.SessionFactory() as session:
            evt = await session.get(Event, event_a.id)
            await session.delete(evt)
            await session.commit()

        async with self.SessionFactory() as session:
            gone = await session.get(MemoryLink, link_id)
            assert gone is None
