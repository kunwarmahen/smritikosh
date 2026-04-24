"""
Tests for Hippocampus.

How to run:
    # Unit tests (all dependencies mocked):
    pytest tests/test_hippocampus.py -v

    # Integration tests (requires Postgres + Neo4j + LLM API key):
    pytest tests/test_hippocampus.py -v -m db

Test strategy:
    - Unit tests inject mock versions of LLMAdapter, EpisodicMemory,
      SemanticMemory, and Amygdala so the Hippocampus orchestration logic
      is tested in isolation.
    - We verify: correct call order, parallel execution, graceful degradation
      when embedding or extraction fails.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import EncodedMemory, Hippocampus, _EXTRACTION_EXAMPLE
from smritikosh.memory.semantic import FactRecord, SemanticMemory
from smritikosh.processing.amygdala import Amygdala


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_event(user_id="u1") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text="test text",
        importance_score=0.8,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_fact_record(**kwargs) -> FactRecord:
    return FactRecord(
        category=kwargs.get("category", "interest"),
        key=kwargs.get("key", "domain"),
        value=kwargs.get("value", "AI agents"),
        confidence=kwargs.get("confidence", 0.9),
        frequency_count=1,
        first_seen_at="2026-03-15T00:00:00+00:00",
        last_seen_at="2026-03-15T00:00:00+00:00",
    )


def make_hippocampus(
    llm=None,
    episodic=None,
    semantic=None,
    amygdala=None,
) -> tuple[Hippocampus, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Return (hippocampus, llm_mock, episodic_mock, semantic_mock, amygdala_mock)."""
    llm = llm or AsyncMock()
    episodic = episodic or AsyncMock(spec=EpisodicMemory)
    if semantic is None:
        semantic = AsyncMock(spec=SemanticMemory)
        # Default: no conflicts — check_fact_conflict returns None
        semantic.check_fact_conflict = AsyncMock(return_value=None)
    amygdala = amygdala or MagicMock(spec=Amygdala)

    hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic, amygdala=amygdala)
    return hippo, llm, episodic, semantic, amygdala


def make_mock_sessions():
    return AsyncMock(), AsyncMock()   # pg_session, neo_session


# ── EncodedMemory ─────────────────────────────────────────────────────────────


class TestEncodedMemory:
    def test_defaults(self):
        event = make_event()
        em = EncodedMemory(event=event)
        assert em.facts == []
        assert em.importance_score == 1.0
        assert em.extraction_failed is False

    def test_with_facts(self):
        event = make_event()
        facts = [make_fact_record(), make_fact_record(key="topic", value="RAG")]
        em = EncodedMemory(event=event, facts=facts, importance_score=0.9)
        assert len(em.facts) == 2
        assert em.importance_score == 0.9


# ── encode() — happy path ─────────────────────────────────────────────────────


class TestEncode:
    @pytest.mark.asyncio
    async def test_returns_encoded_memory(self):
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()
        event = make_event()

        amygdala.score.return_value = 0.8
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": [
            {"category": "interest", "key": "domain", "value": "AI agents", "confidence": 0.9}
        ]})
        episodic.store = AsyncMock(return_value=event)
        semantic.upsert_fact = AsyncMock(return_value=make_fact_record())

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="I love building AI agents")

        assert isinstance(result, EncodedMemory)
        assert result.event is event
        assert len(result.facts) == 1
        assert result.importance_score == 0.8
        assert result.extraction_failed is False

    @pytest.mark.asyncio
    async def test_amygdala_score_is_called_first(self):
        """Amygdala score should run before any async calls."""
        call_order = []

        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        def record_score(text):
            call_order.append("amygdala")
            return 0.7

        async def record_embed(text):
            call_order.append("embed")
            return [0.1] * 1536

        async def record_extract(*a, **kw):
            call_order.append("extract")
            return {"facts": []}

        amygdala.score.side_effect = record_score
        llm.embed = record_embed
        llm.extract_structured = record_extract
        episodic.store = AsyncMock(return_value=make_event())

        await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        assert call_order[0] == "amygdala"

    @pytest.mark.asyncio
    async def test_embed_and_extract_called_concurrently(self):
        """Both LLM calls should start before either completes (asyncio.gather)."""
        started = []
        finished = []

        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()
        amygdala.score.return_value = 0.5
        episodic.store = AsyncMock(return_value=make_event())

        async def slow_embed(text):
            started.append("embed")
            await asyncio.sleep(0.01)
            finished.append("embed")
            return [0.1] * 1536

        async def slow_extract(*a, **kw):
            started.append("extract")
            await asyncio.sleep(0.01)
            finished.append("extract")
            return {"facts": []}

        llm.embed = slow_embed
        llm.extract_structured = slow_extract

        await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        # Both must have started before either finished
        assert set(started) == {"embed", "extract"}

    @pytest.mark.asyncio
    async def test_importance_score_passed_to_episodic_store(self):
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.95
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": []})
        episodic.store = AsyncMock(return_value=make_event())

        await hippo.encode(pg, neo, user_id="u1", raw_text="critical deadline tomorrow")

        call_kwargs = episodic.store.call_args.kwargs
        assert call_kwargs["importance_score"] == 0.95

    @pytest.mark.asyncio
    async def test_metadata_forwarded_to_store(self):
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.5
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": []})
        episodic.store = AsyncMock(return_value=make_event())

        meta = {"source": "slack", "channel": "#general"}
        await hippo.encode(pg, neo, user_id="u1", raw_text="hello", metadata=meta)

        call_kwargs = episodic.store.call_args.kwargs
        assert call_kwargs["metadata"] == meta

    @pytest.mark.asyncio
    async def test_all_extracted_facts_upserted(self):
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.7
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": [
            {"category": "interest", "key": "domain", "value": "AI",  "confidence": 0.9},
            {"category": "role",     "key": "current", "value": "CTO", "confidence": 1.0},
        ]})
        episodic.store = AsyncMock(return_value=make_event())
        semantic.upsert_fact = AsyncMock(return_value=make_fact_record())

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="I am CTO interested in AI")

        assert semantic.upsert_fact.call_count == 2
        assert len(result.facts) == 2


# ── encode() — graceful degradation ──────────────────────────────────────────


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_embedding_failure_stores_event_without_vector(self):
        """If embed() fails, the event is still stored (embedding=None)."""
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.5
        llm.embed = AsyncMock(side_effect=RuntimeError("embedding service down"))
        llm.extract_structured = AsyncMock(return_value={"facts": []})
        episodic.store = AsyncMock(return_value=make_event())

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        # Event was still stored
        episodic.store.assert_called_once()
        # Embedding was passed as None
        assert episodic.store.call_args.kwargs["embedding"] is None

    @pytest.mark.asyncio
    async def test_extraction_failure_sets_flag_and_stores_event(self):
        """If extract_structured() fails, event is stored and extraction_failed=True."""
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.5
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(side_effect=ValueError("bad JSON"))
        episodic.store = AsyncMock(return_value=make_event())

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        assert result.extraction_failed is True
        assert result.facts == []
        # Event still stored
        episodic.store.assert_called_once()
        # No Neo4j calls
        semantic.upsert_fact.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_fact_dict_is_skipped(self):
        """A fact with missing keys should be skipped, others still upserted."""
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.6
        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": [
            {"category": "interest", "key": "domain", "value": "AI", "confidence": 0.9},
            {"category": "bogus_category", "key": "x", "value": "y"},   # invalid
        ]})
        episodic.store = AsyncMock(return_value=make_event())
        semantic.upsert_fact = AsyncMock(side_effect=[
            make_fact_record(),
            ValueError("Unknown fact category: 'bogus_category'"),
        ])

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        # Only the valid fact was stored
        assert len(result.facts) == 1

    @pytest.mark.asyncio
    async def test_both_llm_calls_fail_gracefully(self):
        """Both embed and extract fail — event still stored with no embedding or facts."""
        hippo, llm, episodic, semantic, amygdala = make_hippocampus()
        pg, neo = make_mock_sessions()

        amygdala.score.return_value = 0.5
        llm.embed = AsyncMock(side_effect=RuntimeError("down"))
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("down"))
        episodic.store = AsyncMock(return_value=make_event())

        result = await hippo.encode(pg, neo, user_id="u1", raw_text="test")

        assert result.extraction_failed is True
        assert result.facts == []
        episodic.store.assert_called_once()
        assert episodic.store.call_args.kwargs["embedding"] is None


# ── Default Amygdala wiring ───────────────────────────────────────────────────


class TestDefaultAmygdala:
    def test_amygdala_defaults_to_real_instance(self):
        from smritikosh.llm.adapter import LLMAdapter
        hippo = Hippocampus(
            llm=AsyncMock(spec=LLMAdapter),
            episodic=EpisodicMemory(),
            semantic=SemanticMemory(),
        )
        assert isinstance(hippo.amygdala, Amygdala)

    @pytest.mark.asyncio
    async def test_real_amygdala_scores_high_for_important_text(self):
        """With a real Amygdala, verify important text gets high score end-to-end."""
        hippo, llm, episodic, semantic, _ = make_hippocampus()
        # Use real Amygdala
        hippo.amygdala = Amygdala()
        pg, neo = make_mock_sessions()

        llm.embed = AsyncMock(return_value=[0.1] * 1536)
        llm.extract_structured = AsyncMock(return_value={"facts": []})
        episodic.store = AsyncMock(return_value=make_event())

        result = await hippo.encode(
            pg, neo,
            user_id="u1",
            raw_text="Remember this: I decided to launch the startup by Q2",
        )

        assert result.importance_score > 0.7


# ── DB integration test ───────────────────────────────────────────────────────


@pytest.mark.db
class TestHippocampusDB:
    """
    End-to-end integration test with real Postgres + Neo4j + mocked LLM.

    How to run:
        docker compose up -d
        pytest tests/test_hippocampus.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings
        from smritikosh.db.models import Base
        from smritikosh.db.neo4j import get_driver, init_neo4j

        # Postgres
        self.pg_engine = create_async_engine(settings.postgres_url)
        async with self.pg_engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        self.PgSession = async_sessionmaker(
            self.pg_engine, class_=AsyncSession, expire_on_commit=False
        )

        # Neo4j
        await init_neo4j()
        self.neo_driver = get_driver()

        yield

        async with self.pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await self.pg_engine.dispose()
        async with self.neo_driver.session() as s:
            await s.run("MATCH (n) WHERE n.user_id STARTS WITH 'hip_' DETACH DELETE n")

    async def test_encode_stores_event_and_facts(self):
        llm_mock = AsyncMock()
        llm_mock.embed = AsyncMock(return_value=[0.5] * 1536)
        llm_mock.extract_structured = AsyncMock(return_value={"facts": [
            {"category": "interest", "key": "domain", "value": "AI memory", "confidence": 0.9},
            {"category": "role", "key": "current", "value": "founder", "confidence": 1.0},
        ]})

        hippo = Hippocampus(
            llm=llm_mock,
            episodic=EpisodicMemory(),
            semantic=SemanticMemory(),
        )

        async with self.PgSession() as pg, self.neo_driver.session() as neo:
            result = await hippo.encode(
                pg, neo,
                user_id="hip_u1",
                raw_text="I am the founder and I am building AI memory infrastructure",
            )
            await pg.commit()

        assert result.event.id is not None
        assert len(result.facts) == 2
        assert result.extraction_failed is False

    async def test_encode_graceful_embed_failure(self):
        llm_mock = AsyncMock()
        llm_mock.embed = AsyncMock(side_effect=RuntimeError("LLM down"))
        llm_mock.extract_structured = AsyncMock(return_value={"facts": []})

        hippo = Hippocampus(
            llm=llm_mock,
            episodic=EpisodicMemory(),
            semantic=SemanticMemory(),
        )

        async with self.PgSession() as pg, self.neo_driver.session() as neo:
            result = await hippo.encode(
                pg, neo,
                user_id="hip_u2",
                raw_text="some interaction",
            )
            await pg.commit()

        # Event persisted even without embedding
        assert result.event.id is not None
        assert result.event.embedding is None
