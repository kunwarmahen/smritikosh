"""
Tests for POST /ingest/session and POST /memory/fact endpoints.

All tests are offline — DB and LLM dependencies are mocked.

Run:
    pytest tests/test_session_ingest.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.db.models import Event, FactStatus, ProcessedSession, SourceType
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import EncodedMemory, Hippocampus
from smritikosh.memory.semantic import FactRecord, SemanticMemory, UserProfile


# ── Helpers ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc)


def make_fact(**overrides) -> FactRecord:
    defaults = dict(
        category="preference",
        key="editor",
        value="neovim",
        confidence=1.0,
        frequency_count=1,
        first_seen_at=_now().isoformat(),
        last_seen_at=_now().isoformat(),
        source_type=SourceType.UI_MANUAL,
        source_meta={},
        status=FactStatus.ACTIVE,
    )
    defaults.update(overrides)
    return FactRecord(**defaults)


def make_event(user_id: str = "u1") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text="Session content",
        importance_score=0.85,
        consolidated=False,
        event_metadata={},
        source_type=SourceType.PASSIVE_DISTILLATION,
        source_meta={},
        created_at=_now(),
    )


def make_encoded_memory(user_id: str = "u1") -> EncodedMemory:
    return EncodedMemory(
        event=make_event(user_id),
        facts=[make_fact()],
        importance_score=0.85,
    )


def _auth_override(user_id: str = "u1"):
    return {"sub": user_id, "user_id": user_id, "role": "admin", "app_ids": ["default"]}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pg():
    pg = AsyncMock()
    # execute() returns an object with scalar_one_or_none() → None (no existing session)
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    pg.execute = AsyncMock(return_value=execute_result)
    pg.flush = AsyncMock()
    pg.add = MagicMock()
    return pg


@pytest.fixture
def mock_neo():
    return AsyncMock()


@pytest.fixture
def mock_hippocampus():
    h = AsyncMock(spec=Hippocampus)
    h.encode.return_value = make_encoded_memory()
    return h


@pytest.fixture
def mock_semantic():
    s = AsyncMock(spec=SemanticMemory)
    s.get_user_profile.return_value = UserProfile(user_id="u1", app_id="default", facts=[])
    s.upsert_fact.return_value = make_fact()
    return s


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.extract_structured.return_value = {
        "facts": [
            {"category": "preference", "key": "editor", "value": "neovim", "confidence": 0.9}
        ]
    }
    return llm


@pytest.fixture(autouse=True)
def override_deps(mock_pg, mock_neo, mock_hippocampus, mock_semantic, mock_llm):
    from smritikosh.auth.deps import get_current_user
    app.dependency_overrides[get_session] = lambda: mock_pg
    app.dependency_overrides[get_neo4j_session] = lambda: mock_neo
    app.dependency_overrides[deps.get_hippocampus] = lambda: mock_hippocampus
    app.dependency_overrides[deps.get_semantic] = lambda: mock_semantic
    app.dependency_overrides[deps.get_llm] = lambda: mock_llm
    app.dependency_overrides[get_current_user] = lambda: _auth_override()
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── POST /ingest/session tests ────────────────────────────────────────────────


class TestSessionIngest:
    _ENDPOINT = "/ingest/session"

    def _payload(self, **overrides):
        base = {
            "user_id": "u1",
            "app_id": "default",
            "session_id": "sess-abc",
            "turns": [
                {"role": "user", "content": "I always prefer dark mode."},
                {"role": "assistant", "content": "Sure, noted."},
                {"role": "user", "content": "My goal is to ship by Q3."},
            ],
            "partial": False,
            "use_trigger_filter": True,
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_returns_201_on_success(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 201

    @pytest.mark.asyncio
    async def test_response_shape(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload())
        data = r.json()
        assert data["session_id"] == "sess-abc"
        assert data["user_id"] == "u1"
        assert "turns_processed" in data
        assert "facts_extracted" in data
        assert "already_processed" in data
        assert "extraction_skipped" in data
        assert "partial" in data

    @pytest.mark.asyncio
    async def test_idempotency_returns_cached_result(self, client, mock_pg):
        # Simulate a fully-processed session already in DB
        existing = ProcessedSession(
            id=uuid.uuid4(),
            user_id="u1",
            app_id="default",
            session_id="sess-abc",
            turns_count=3,
            facts_extracted=2,
            last_turn_index=3,
            is_partial=False,
        )
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = existing
        mock_pg.execute = AsyncMock(return_value=execute_result)

        r = await client.post(self._ENDPOINT, json=self._payload())
        data = r.json()
        assert r.status_code == 201
        assert data["already_processed"] is True

    @pytest.mark.asyncio
    async def test_trigger_filter_skips_extraction_when_no_triggers(self, client, mock_llm):
        payload = self._payload(
            turns=[
                {"role": "user", "content": "What is the weather today?"},
                {"role": "user", "content": "Tell me a joke."},
            ],
            use_trigger_filter=True,
        )
        r = await client.post(self._ENDPOINT, json=payload)
        data = r.json()
        assert r.status_code == 201
        assert data["extraction_skipped"] is True
        # LLM should NOT have been called
        mock_llm.extract_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_trigger_filter_disabled_calls_llm(self, client, mock_llm):
        payload = self._payload(
            turns=[
                {"role": "user", "content": "What is the weather today?"},
            ],
            use_trigger_filter=False,
        )
        r = await client.post(self._ENDPOINT, json=payload)
        assert r.status_code == 201
        mock_llm.extract_structured.assert_called_once()

    @pytest.mark.asyncio
    async def test_partial_session_flag_preserved(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload(partial=True))
        data = r.json()
        assert data["partial"] is True

    @pytest.mark.asyncio
    async def test_strips_assistant_turns_from_processing(self, client, mock_llm):
        """Assistant-only turns should result in 0 user turns and skipped extraction."""
        payload = self._payload(
            turns=[
                {"role": "assistant", "content": "I always use the best tools."},
                {"role": "assistant", "content": "My goal is to help you."},
            ],
        )
        r = await client.post(self._ENDPOINT, json=payload)
        data = r.json()
        # No user turns → no processing
        assert data["turns_processed"] == 0
        assert data["extraction_skipped"] is True

    @pytest.mark.asyncio
    async def test_empty_turns_list(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload(turns=[]))
        assert r.status_code == 201
        data = r.json()
        assert data["turns_processed"] == 0

    @pytest.mark.asyncio
    async def test_transcript_alias_endpoint(self, client):
        """POST /ingest/transcript is an alias for /ingest/session."""
        r = await client.post("/ingest/transcript", json=self._payload())
        assert r.status_code == 201

    @pytest.mark.asyncio
    async def test_sentinel_blocks_stripped_from_content(self, client, mock_llm):
        """Sentinel blocks should be stripped before extraction, not re-encoded."""
        payload = self._payload(
            turns=[
                {
                    "role": "user",
                    "content": (
                        "<!-- smritikosh:context-start -->existing fact<!-- smritikosh:context-end -->"
                        " I prefer vim."
                    ),
                }
            ],
            use_trigger_filter=False,
        )
        r = await client.post(self._ENDPOINT, json=payload)
        assert r.status_code == 201
        # The extraction prompt passed to LLM should not contain "existing fact"
        call_kwargs = mock_llm.extract_structured.call_args
        if call_kwargs:
            prompt = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
            assert "existing fact" not in prompt


# ── POST /memory/fact tests ───────────────────────────────────────────────────


class TestMemoryFact:
    _ENDPOINT = "/memory/fact"

    def _payload(self, **overrides):
        base = {
            "user_id": "u1",
            "app_id": "default",
            "category": "preference",
            "key": "editor",
            "value": "neovim",
        }
        base.update(overrides)
        return base

    @pytest.mark.asyncio
    async def test_returns_201_on_success(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 201

    @pytest.mark.asyncio
    async def test_response_shape(self, client):
        r = await client.post(self._ENDPOINT, json=self._payload())
        data = r.json()
        assert data["category"] == "preference"
        assert data["key"] == "editor"
        assert data["value"] == "neovim"
        assert "confidence" in data
        assert "source_type" in data
        assert "status" in data
        assert "frequency_count" in data

    @pytest.mark.asyncio
    async def test_ui_manual_source_type_default(self, client, mock_semantic):
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 201
        # Check that upsert_fact was called with ui_manual source_type
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["source_type"] == SourceType.UI_MANUAL

    @pytest.mark.asyncio
    async def test_custom_source_type(self, client, mock_semantic):
        r = await client.post(
            self._ENDPOINT,
            json=self._payload(source_type="tool_use"),
        )
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["source_type"] == "tool_use"

    @pytest.mark.asyncio
    async def test_confidence_override(self, client, mock_semantic):
        r = await client.post(
            self._ENDPOINT,
            json=self._payload(confidence=0.75),
        )
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["confidence"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_note_stored_in_source_meta(self, client, mock_semantic):
        r = await client.post(
            self._ENDPOINT,
            json=self._payload(note="user typed this during onboarding"),
        )
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["source_meta"].get("note") == "user typed this during onboarding"

    @pytest.mark.asyncio
    async def test_invalid_category_returns_422(self, client, mock_semantic):
        mock_semantic.upsert_fact.side_effect = ValueError("Unknown fact category: 'bogus'")
        r = await client.post(self._ENDPOINT, json=self._payload(category="bogus"))
        assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_semantic_error_returns_500(self, client, mock_semantic):
        mock_semantic.upsert_fact.side_effect = RuntimeError("Neo4j gone")
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 500

    @pytest.mark.asyncio
    async def test_full_confidence_for_ui_manual(self, client, mock_semantic):
        """ui_manual facts should default to confidence=1.0."""
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["confidence"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_active_status_for_high_confidence(self, client, mock_semantic):
        r = await client.post(self._ENDPOINT, json=self._payload())
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["status"] == FactStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_pending_status_for_low_confidence(self, client, mock_semantic):
        r = await client.post(
            self._ENDPOINT,
            json=self._payload(confidence=0.40),
        )
        assert r.status_code == 201
        call_kwargs = mock_semantic.upsert_fact.call_args.kwargs
        assert call_kwargs["status"] == FactStatus.PENDING
