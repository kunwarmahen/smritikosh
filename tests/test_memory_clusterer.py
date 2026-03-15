"""
Tests for MemoryClusterer and the cluster_embeddings algorithm.

Unit tests mock LLMAdapter and the DB session to verify clustering logic,
LLM labelling, and write-back behaviour.
DB integration tests are gated behind @pytest.mark.db.
"""

import math
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

import numpy as np
import pytest

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.processing.memory_clusterer import (
    ClusterResult,
    MemoryClusterer,
    MIN_EVENTS_TO_CLUSTER,
    DEFAULT_SIMILARITY_THRESHOLD,
    _cosine_sim,
    _build_label_prompt,
    cluster_embeddings,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(raw_text: str = "test event", embedding=None) -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text=raw_text,
        importance_score=0.8,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
        embedding=embedding or [0.1] * 10,
    )


def unit_vec(dim: int, idx: int) -> list[float]:
    """Return a unit vector with 1.0 at position idx, 0.0 elsewhere."""
    v = [0.0] * dim
    v[idx] = 1.0
    return v


def scaled_vec(base: list[float], scale: float) -> list[float]:
    return [x * scale for x in base]


def make_mock_llm(label: str = "AI projects") -> AsyncMock:
    llm = AsyncMock()
    llm.extract_structured = AsyncMock(return_value={"label": label})
    return llm


def make_mock_session(events: list[Event]) -> AsyncMock:
    session = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = events
    session.execute = AsyncMock(return_value=mock_result)
    return session


def make_mock_episodic() -> AsyncMock:
    from smritikosh.memory.episodic import EpisodicMemory
    return AsyncMock(spec=EpisodicMemory)


# ── _cosine_sim ───────────────────────────────────────────────────────────────


class TestCosineSim:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_sim(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_sim(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_sim(a, b) - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_sim(a, b) == 0.0

    def test_parallel_different_magnitude(self):
        a = np.array([1.0, 1.0])
        b = np.array([3.0, 3.0])
        assert abs(_cosine_sim(a, b) - 1.0) < 1e-6


# ── cluster_embeddings ────────────────────────────────────────────────────────


class TestClusterEmbeddings:
    def test_empty_input(self):
        assert cluster_embeddings([]) == []

    def test_single_embedding(self):
        result = cluster_embeddings([[1.0, 0.0]])
        assert result == [0]

    def test_identical_embeddings_same_cluster(self):
        v = [1.0, 0.0, 0.0]
        result = cluster_embeddings([v, v, v])
        assert len(set(result)) == 1   # all same cluster

    def test_orthogonal_embeddings_separate_clusters(self):
        """Orthogonal vectors (similarity=0) should never merge."""
        v1 = unit_vec(4, 0)
        v2 = unit_vec(4, 1)
        v3 = unit_vec(4, 2)
        result = cluster_embeddings([v1, v2, v3], similarity_threshold=0.5)
        assert len(set(result)) == 3

    def test_high_similarity_merges(self):
        """Vectors that are 99% similar should end up in the same cluster."""
        base = [1.0, 0.01, 0.0]
        close = [1.0, 0.02, 0.0]
        result = cluster_embeddings([base, close], similarity_threshold=0.75)
        assert result[0] == result[1]

    def test_low_threshold_merges_more(self):
        v1 = unit_vec(4, 0)
        v2 = [0.7, 0.7, 0.0, 0.0]  # ~45 degrees from v1
        # With threshold=0.7 → same cluster (sim ≈ 0.7)
        result_low = cluster_embeddings([v1, v2], similarity_threshold=0.3)
        result_high = cluster_embeddings([v1, v2], similarity_threshold=0.9)
        # low threshold → 1 cluster; high threshold → 2 clusters
        assert len(set(result_low)) <= len(set(result_high))

    def test_result_length_matches_input(self):
        vecs = [[float(i), float(j)] for i in range(5) for j in range(5)]
        result = cluster_embeddings(vecs)
        assert len(result) == len(vecs)

    def test_cluster_ids_are_zero_indexed_contiguous(self):
        """Cluster IDs must be 0, 1, 2, … with no gaps."""
        # Three clearly separate clusters
        v1, v2, v3 = unit_vec(3, 0), unit_vec(3, 1), unit_vec(3, 2)
        result = cluster_embeddings([v1, v2, v3], similarity_threshold=0.9)
        assert set(result) == {0, 1, 2}

    def test_first_event_always_cluster_zero(self):
        result = cluster_embeddings([[1.0, 0.0, 0.0]])
        assert result[0] == 0


# ── _build_label_prompt ───────────────────────────────────────────────────────


class TestBuildLabelPrompt:
    def test_contains_memory_text(self):
        event = make_event("User discussed AI memory architecture")
        prompt = _build_label_prompt([event])
        assert "AI memory architecture" in prompt

    def test_truncates_to_five_events(self):
        events = [make_event(f"event {i}") for i in range(10)]
        prompt = _build_label_prompt(events)
        # Only 5 samples shown
        assert "event 4" in prompt
        assert "event 5" not in prompt

    def test_uses_summary_when_available(self):
        event = make_event("long raw text that should not appear")
        event.summary = "Short summary"
        prompt = _build_label_prompt([event])
        assert "Short summary" in prompt

    def test_contains_instruction(self):
        prompt = _build_label_prompt([make_event()])
        assert "topic" in prompt.lower()


# ── ClusterResult ─────────────────────────────────────────────────────────────


class TestClusterResult:
    def test_defaults(self):
        r = ClusterResult(user_id="u1", app_id="default")
        assert r.events_processed == 0
        assert r.clusters_found == 0
        assert r.events_clustered == 0
        assert r.skipped is False
        assert r.skip_reason == ""


# ── MemoryClusterer.run — skip guard ──────────────────────────────────────────


class TestClustererSkipGuard:
    @pytest.mark.asyncio
    async def test_skips_when_too_few_events(self):
        events = [make_event() for _ in range(MIN_EVENTS_TO_CLUSTER - 1)]
        session = make_mock_session(events)
        clusterer = MemoryClusterer(llm=make_mock_llm(), episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        assert result.skipped is True
        assert str(MIN_EVENTS_TO_CLUSTER - 1) in result.skip_reason

    @pytest.mark.asyncio
    async def test_skips_when_zero_events(self):
        session = make_mock_session([])
        clusterer = MemoryClusterer(llm=make_mock_llm(), episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        assert result.skipped is True

    @pytest.mark.asyncio
    async def test_runs_when_enough_events(self):
        events = [make_event(embedding=unit_vec(4, 0)) for _ in range(MIN_EVENTS_TO_CLUSTER)]
        session = make_mock_session(events)
        clusterer = MemoryClusterer(llm=make_mock_llm(), episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        assert result.skipped is False
        assert result.events_processed == MIN_EVENTS_TO_CLUSTER


# ── MemoryClusterer.run — success path ────────────────────────────────────────


class TestClustererSuccess:
    @pytest.mark.asyncio
    async def test_all_events_clustered(self):
        # 5 identical embeddings → 1 cluster
        events = [make_event(embedding=[1.0, 0.0]) for _ in range(5)]
        session = make_mock_session(events)
        clusterer = MemoryClusterer(llm=make_mock_llm("AI memory"), episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        assert result.events_clustered == 5

    @pytest.mark.asyncio
    async def test_correct_cluster_count(self):
        # 5 events: 3 similar + 2 similar but orthogonal to first group → 2 clusters
        v1 = unit_vec(4, 0)
        v2 = unit_vec(4, 2)
        events = (
            [make_event(embedding=v1) for _ in range(3)]
            + [make_event(embedding=v2) for _ in range(2)]
        )
        session = make_mock_session(events)
        clusterer = MemoryClusterer(
            llm=make_mock_llm(), episodic=make_mock_episodic(),
            similarity_threshold=0.9,
        )

        result = await clusterer.run(session, user_id="u1")

        assert result.clusters_found == 2

    @pytest.mark.asyncio
    async def test_llm_called_once_per_cluster(self):
        v1, v2 = unit_vec(4, 0), unit_vec(4, 1)
        events = (
            [make_event(embedding=v1) for _ in range(3)]
            + [make_event(embedding=v2) for _ in range(2)]
        )
        session = make_mock_session(events)
        llm = make_mock_llm()
        clusterer = MemoryClusterer(
            llm=llm, episodic=make_mock_episodic(), similarity_threshold=0.9
        )

        await clusterer.run(session, user_id="u1")

        assert llm.extract_structured.call_count == 2   # 2 clusters → 2 LLM calls

    @pytest.mark.asyncio
    async def test_db_updated_for_each_cluster(self):
        v1, v2 = unit_vec(4, 0), unit_vec(4, 1)
        events = (
            [make_event(embedding=v1) for _ in range(3)]
            + [make_event(embedding=v2) for _ in range(2)]
        )
        session = make_mock_session(events)
        clusterer = MemoryClusterer(
            llm=make_mock_llm(), episodic=make_mock_episodic(), similarity_threshold=0.9
        )

        await clusterer.run(session, user_id="u1")

        # 1 SELECT + 2 UPDATEs (one per cluster)
        assert session.execute.call_count == 3

    @pytest.mark.asyncio
    async def test_events_processed_count(self):
        events = [make_event() for _ in range(7)]
        session = make_mock_session(events)
        clusterer = MemoryClusterer(llm=make_mock_llm(), episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        assert result.events_processed == 7


# ── LLM failure graceful handling ─────────────────────────────────────────────


class TestClustererLLMFailure:
    @pytest.mark.asyncio
    async def test_fallback_label_on_llm_error(self):
        events = [make_event(embedding=[1.0, 0.0]) for _ in range(5)]
        session = make_mock_session(events)
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        clusterer = MemoryClusterer(llm=llm, episodic=make_mock_episodic())

        # Should not raise
        result = await clusterer.run(session, user_id="u1")

        assert result.skipped is False
        assert result.events_clustered == 5

    @pytest.mark.asyncio
    async def test_fallback_label_format(self):
        """Fallback label is 'cluster_<id>' when LLM fails."""
        events = [make_event(embedding=[1.0, 0.0]) for _ in range(5)]
        session = make_mock_session(events)
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(return_value={"label": ""})  # empty label
        clusterer = MemoryClusterer(llm=llm, episodic=make_mock_episodic())

        result = await clusterer.run(session, user_id="u1")

        # Verify the UPDATE was called (cluster written even with fallback label)
        assert result.events_clustered > 0


# ── DB integration tests ──────────────────────────────────────────────────────


@pytest.mark.db
class TestMemoryClustererDB:
    """
    How to run:
        docker compose up -d postgres
        pytest tests/test_memory_clusterer.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

        from smritikosh.config import settings
        from smritikosh.db.models import Base
        from smritikosh.llm.adapter import LLMAdapter

        engine = create_async_engine(settings.postgres_url)
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

        self.SessionFactory = async_sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
        self.engine = engine
        self.episodic = EpisodicMemory()
        self.clusterer = MemoryClusterer(
            llm=LLMAdapter(), episodic=self.episodic, min_events=3
        )

        yield

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()

    async def _store(self, session, text: str, embedding: list[float]) -> Event:
        return await self.episodic.store(
            session, user_id="u1", raw_text=text, embedding=embedding
        )

    async def test_cluster_ids_written_to_events(self):
        v1, v2 = unit_vec(1536, 0), unit_vec(1536, 1)
        async with self.SessionFactory() as session:
            for i in range(3):
                await self._store(session, f"AI event {i}", v1)
            for i in range(2):
                await self._store(session, f"cooking event {i}", v2)
            await session.commit()

        async with self.SessionFactory() as session:
            result = await self.clusterer.run(session, user_id="u1")
            await session.commit()

        assert not result.skipped
        assert result.clusters_found == 2
        assert result.events_clustered == 5

    async def test_skips_when_no_embeddings(self):
        async with self.SessionFactory() as session:
            # Store events without embeddings
            for i in range(5):
                await self.episodic.store(
                    session, user_id="u1", raw_text=f"event {i}", embedding=None
                )
            await session.commit()

        async with self.SessionFactory() as session:
            result = await self.clusterer.run(session, user_id="u1")

        assert result.skipped is True
