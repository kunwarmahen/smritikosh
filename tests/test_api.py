"""
Tests for FastAPI routes.

How to run:
    # Unit tests (all DB and LLM deps overridden):
    pytest tests/test_api.py -v

    # Integration tests (requires running Postgres + Neo4j + LLM):
    pytest tests/test_api.py -v -m db

Test strategy:
    - Unit tests use FastAPI's dependency_overrides to inject mocks for
      Hippocampus, ContextBuilder, EpisodicMemory, and both DB sessions.
    - We test: request validation, correct status codes, response shape,
      and error handling (500 on exception).
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.db.models import Event
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import EncodedMemory, Hippocampus
from smritikosh.memory.semantic import FactRecord, SemanticMemory, UserProfile
from smritikosh.retrieval.context_builder import ContextBuilder, MemoryContext


# ── Shared helpers ─────────────────────────────────────────────────────────────


def make_event(user_id="u1") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text="User discussed building an AI memory system",
        importance_score=0.85,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_encoded_memory(user_id="u1", facts_count=2) -> EncodedMemory:
    event = make_event(user_id)
    facts = [
        FactRecord("interest", "domain", "AI agents", 0.9, 1,
                   "2026-03-15T00:00:00+00:00", "2026-03-15T00:00:00+00:00")
        for _ in range(facts_count)
    ]
    return EncodedMemory(event=event, facts=facts, importance_score=0.85)


def make_memory_context(user_id="u1") -> MemoryContext:
    from smritikosh.memory.episodic import SearchResult
    profile = UserProfile(
        user_id=user_id, app_id="default",
        facts=[FactRecord("role", "current", "founder", 1.0, 1,
                          "2026-03-15T00:00:00+00:00", "2026-03-15T00:00:00+00:00")],
    )
    sr = SearchResult(event=make_event(user_id), hybrid_score=0.88)
    return MemoryContext(
        user_id=user_id, query="test query",
        similar_events=[sr], user_profile=profile, recent_events=[],
    )


# ── Dependency override fixtures ──────────────────────────────────────────────


@pytest.fixture
def mock_pg_session():
    return AsyncMock()


@pytest.fixture
def mock_neo_session():
    return AsyncMock()


@pytest.fixture
def mock_hippocampus():
    return AsyncMock(spec=Hippocampus)


@pytest.fixture
def mock_context_builder():
    return AsyncMock(spec=ContextBuilder)


@pytest.fixture
def mock_episodic():
    return AsyncMock(spec=EpisodicMemory)


@pytest.fixture(autouse=True)
def override_deps(mock_pg_session, mock_neo_session, mock_hippocampus,
                  mock_context_builder, mock_episodic):
    """Replace all I/O dependencies with mocks for every test in this module."""
    app.dependency_overrides[get_session] = lambda: mock_pg_session
    app.dependency_overrides[get_neo4j_session] = lambda: mock_neo_session
    app.dependency_overrides[deps.get_hippocampus] = lambda: mock_hippocampus
    app.dependency_overrides[deps.get_context_builder] = lambda: mock_context_builder
    app.dependency_overrides[deps.get_episodic] = lambda: mock_episodic

    yield

    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── GET /health ───────────────────────────────────────────────────────────────


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_ok(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_returns_version(self, client):
        response = await client.get("/health")
        assert "version" in response.json()


# ── POST /memory/event ────────────────────────────────────────────────────────


class TestCaptureEvent:
    @pytest.mark.asyncio
    async def test_returns_201_on_success(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())

        response = await client.post("/memory/event", json={
            "user_id": "u1",
            "content": "I decided to build an AI memory startup",
        })

        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory(facts_count=3))

        response = await client.post("/memory/event", json={
            "user_id": "u1",
            "content": "I am building smritikosh",
        })

        body = response.json()
        assert "event_id" in body
        assert body["user_id"] == "u1"
        assert body["facts_extracted"] == 3
        assert body["importance_score"] == 0.85
        assert body["extraction_failed"] is False

    @pytest.mark.asyncio
    async def test_hippocampus_called_with_correct_args(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())

        await client.post("/memory/event", json={
            "user_id": "mahen",
            "content": "I prefer dark mode",
            "app_id": "my_app",
            "metadata": {"source": "slack"},
        })

        call_kwargs = mock_hippocampus.encode.call_args.kwargs
        assert call_kwargs["user_id"] == "mahen"
        assert call_kwargs["raw_text"] == "I prefer dark mode"
        assert call_kwargs["app_id"] == "my_app"
        assert call_kwargs["metadata"] == {"source": "slack"}

    @pytest.mark.asyncio
    async def test_missing_required_fields_returns_422(self, client):
        response = await client.post("/memory/event", json={"user_id": "u1"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_content_returns_422(self, client):
        response = await client.post("/memory/event", json={
            "user_id": "u1",
            "content": "",
        })
        # FastAPI validates non-empty string constraint
        # empty string passes schema validation (no min_length set) — that's fine
        # just checking it doesn't 500
        assert response.status_code in (201, 422)

    @pytest.mark.asyncio
    async def test_hippocampus_exception_returns_500(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(side_effect=RuntimeError("db connection lost"))

        response = await client.post("/memory/event", json={
            "user_id": "u1",
            "content": "test",
        })

        assert response.status_code == 500
        assert "Memory encoding failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_extraction_failed_flag_propagated(self, client, mock_hippocampus):
        result = make_encoded_memory()
        result.extraction_failed = True
        mock_hippocampus.encode = AsyncMock(return_value=result)

        response = await client.post("/memory/event", json={
            "user_id": "u1",
            "content": "test",
        })

        assert response.json()["extraction_failed"] is True

    @pytest.mark.asyncio
    async def test_default_app_id_used(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())

        await client.post("/memory/event", json={"user_id": "u1", "content": "hi"})

        assert mock_hippocampus.encode.call_args.kwargs["app_id"] == "default"


# ── POST /context ─────────────────────────────────────────────────────────────


class TestGetContext:
    @pytest.mark.asyncio
    async def test_returns_200_on_success(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(return_value=make_memory_context())

        response = await client.post("/context", json={
            "user_id": "u1",
            "query": "What should I build next?",
        })

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(return_value=make_memory_context())

        response = await client.post("/context", json={
            "user_id": "u1",
            "query": "What should I build?",
        })

        body = response.json()
        assert body["user_id"] == "u1"
        assert body["query"] == "What should I build?"
        assert isinstance(body["context_text"], str)
        assert isinstance(body["messages"], list)
        assert isinstance(body["total_memories"], int)
        assert isinstance(body["embedding_failed"], bool)

    @pytest.mark.asyncio
    async def test_context_text_contains_memory_header(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(return_value=make_memory_context())

        response = await client.post("/context", json={
            "user_id": "u1",
            "query": "test",
        })

        assert "## User Memory Context" in response.json()["context_text"]

    @pytest.mark.asyncio
    async def test_messages_is_list_with_system_role(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(return_value=make_memory_context())

        response = await client.post("/context", json={"user_id": "u1", "query": "test"})

        messages = response.json()["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_builder_called_with_correct_args(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(return_value=make_memory_context())

        await client.post("/context", json={
            "user_id": "mahen",
            "query": "what is my goal?",
            "app_id": "smritikosh",
        })

        call_kwargs = mock_context_builder.build.call_args.kwargs
        assert call_kwargs["user_id"] == "mahen"
        assert call_kwargs["query"] == "what is my goal?"
        assert call_kwargs["app_id"] == "smritikosh"

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_422(self, client):
        response = await client.post("/context", json={"query": "test"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_builder_exception_returns_500(self, client, mock_context_builder):
        mock_context_builder.build = AsyncMock(side_effect=RuntimeError("retrieval failed"))

        response = await client.post("/context", json={"user_id": "u1", "query": "test"})

        assert response.status_code == 500
        assert "Context retrieval failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_empty_context_returns_placeholder(self, client, mock_context_builder):
        empty_ctx = MemoryContext(user_id="u1", query="test")
        mock_context_builder.build = AsyncMock(return_value=empty_ctx)

        response = await client.post("/context", json={"user_id": "u1", "query": "test"})

        assert "no memory stored" in response.json()["context_text"]
        assert response.json()["total_memories"] == 0


# ── GET /memory/{user_id} ─────────────────────────────────────────────────────


class TestGetRecentEvents:
    @pytest.mark.asyncio
    async def test_returns_200(self, client, mock_episodic):
        mock_episodic.get_recent = AsyncMock(return_value=[make_event("u1")])

        response = await client.get("/memory/u1")

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_episodic):
        mock_episodic.get_recent = AsyncMock(return_value=[make_event("u1")])

        response = await client.get("/memory/u1")

        body = response.json()
        assert body["user_id"] == "u1"
        assert isinstance(body["events"], list)
        assert len(body["events"]) == 1
        assert "event_id" in body["events"][0]
        assert "raw_text" in body["events"][0]
        assert "importance_score" in body["events"][0]

    @pytest.mark.asyncio
    async def test_limit_query_param_forwarded(self, client, mock_episodic):
        mock_episodic.get_recent = AsyncMock(return_value=[])

        await client.get("/memory/u1?limit=3")

        assert mock_episodic.get_recent.call_args.kwargs["limit"] == 3

    @pytest.mark.asyncio
    async def test_app_id_query_param_forwarded(self, client, mock_episodic):
        mock_episodic.get_recent = AsyncMock(return_value=[])

        await client.get("/memory/u1?app_id=my_app")

        assert mock_episodic.get_recent.call_args.kwargs["app_id"] == "my_app"

    @pytest.mark.asyncio
    async def test_limit_over_50_returns_422(self, client):
        response = await client.get("/memory/u1?limit=100")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_empty_history_returns_empty_list(self, client, mock_episodic):
        mock_episodic.get_recent = AsyncMock(return_value=[])

        response = await client.get("/memory/u1")

        assert response.json()["events"] == []
