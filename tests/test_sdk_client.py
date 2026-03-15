"""
Tests for SmritikoshClient SDK.

Uses respx to mock HTTP responses — no real server needed.
Live tests against a running server are gated behind @pytest.mark.live.
"""

import pytest
import respx
from httpx import Response

from smritikosh.sdk.client import SmritikoshClient, SmritikoshError
from smritikosh.sdk.types import (
    BeliefItem,
    EncodedEvent,
    FeedbackRecord,
    HealthStatus,
    IdentityDimensionItem,
    IdentityProfile,
    MemoryContext,
    RecentEvent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE_URL = "http://localhost:8080"

ENCODE_RESPONSE = {
    "event_id": "evt-001",
    "user_id": "alice",
    "importance_score": 0.75,
    "facts_extracted": 2,
    "extraction_failed": False,
}

CONTEXT_RESPONSE = {
    "user_id": "alice",
    "query": "What editor does Alice use?",
    "context_text": "## User Memory Context\nAlice uses Neovim.",
    "messages": [{"role": "system", "content": "Alice uses Neovim."}],
    "total_memories": 3,
    "embedding_failed": False,
}

RECENT_RESPONSE = {
    "user_id": "alice",
    "app_id": "default",
    "events": [
        {
            "event_id": "evt-001",
            "raw_text": "I prefer dark mode.",
            "importance_score": 0.6,
            "consolidated": False,
            "created_at": "2024-01-01T10:00:00+00:00",
        },
    ],
}

HEALTH_RESPONSE = {"status": "ok", "version": "0.1.0"}

FEEDBACK_RESPONSE = {
    "feedback_id": "fb-001",
    "event_id": "evt-001",
    "new_importance_score": 0.85,
}

IDENTITY_RESPONSE = {
    "user_id": "alice",
    "app_id": "default",
    "summary": "A founder building AI tools.",
    "dimensions": [
        {
            "category": "role",
            "dominant_value": "founder",
            "confidence": 0.9,
            "fact_count": 2,
        }
    ],
    "beliefs": [
        {
            "statement": "values speed over perfection",
            "category": "value",
            "confidence": 0.85,
            "evidence_count": 3,
        }
    ],
    "total_facts": 2,
    "computed_at": "2026-03-15T00:00:00+00:00",
    "is_empty": False,
}


@pytest.fixture
async def client():
    async with SmritikoshClient(base_url=BASE_URL) as c:
        yield c


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class TestLifecycle:
    async def test_context_manager_opens_and_closes(self):
        async with SmritikoshClient(base_url=BASE_URL) as client:
            assert client._client is not None
        assert client._client is None

    async def test_manual_open_close(self):
        client = SmritikoshClient(base_url=BASE_URL)
        await client.aopen()
        assert client._client is not None
        await client.aclose()
        assert client._client is None

    async def test_raises_if_not_open(self):
        client = SmritikoshClient(base_url=BASE_URL)
        with pytest.raises(RuntimeError, match="not open"):
            await client.health()

    async def test_double_close_is_safe(self):
        client = SmritikoshClient(base_url=BASE_URL)
        await client.aopen()
        await client.aclose()
        await client.aclose()  # should not raise


# ── encode() ─────────────────────────────────────────────────────────────────

class TestEncode:
    @respx.mock
    async def test_returns_encoded_event(self, client):
        respx.post(f"{BASE_URL}/memory/event").mock(
            return_value=Response(201, json=ENCODE_RESPONSE)
        )
        event = await client.encode(user_id="alice", content="I prefer dark mode.")
        assert isinstance(event, EncodedEvent)
        assert event.event_id == "evt-001"
        assert event.user_id == "alice"
        assert event.importance_score == 0.75
        assert event.facts_extracted == 2
        assert event.extraction_failed is False

    @respx.mock
    async def test_sends_correct_payload(self, client):
        route = respx.post(f"{BASE_URL}/memory/event").mock(
            return_value=Response(201, json=ENCODE_RESPONSE)
        )
        await client.encode(
            user_id="alice",
            content="I prefer Neovim.",
            app_id="myapp",
            metadata={"source": "slack"},
        )
        body = route.calls[0].request.content
        import json
        payload = json.loads(body)
        assert payload["user_id"] == "alice"
        assert payload["content"] == "I prefer Neovim."
        assert payload["app_id"] == "myapp"
        assert payload["metadata"] == {"source": "slack"}

    @respx.mock
    async def test_uses_client_app_id_by_default(self, client):
        # client was created without app_id → uses "default"
        route = respx.post(f"{BASE_URL}/memory/event").mock(
            return_value=Response(201, json=ENCODE_RESPONSE)
        )
        await client.encode(user_id="alice", content="Hello.")
        import json
        payload = json.loads(route.calls[0].request.content)
        assert payload["app_id"] == "default"

    @respx.mock
    async def test_uses_custom_client_app_id(self):
        async with SmritikoshClient(base_url=BASE_URL, app_id="myapp") as client:
            route = respx.post(f"{BASE_URL}/memory/event").mock(
                return_value=Response(201, json=ENCODE_RESPONSE)
            )
            await client.encode(user_id="alice", content="Hello.")
            import json
            payload = json.loads(route.calls[0].request.content)
            assert payload["app_id"] == "myapp"

    @respx.mock
    async def test_per_call_app_id_overrides_client_default(self):
        async with SmritikoshClient(base_url=BASE_URL, app_id="myapp") as client:
            route = respx.post(f"{BASE_URL}/memory/event").mock(
                return_value=Response(201, json=ENCODE_RESPONSE)
            )
            await client.encode(user_id="alice", content="Hello.", app_id="override")
            import json
            payload = json.loads(route.calls[0].request.content)
            assert payload["app_id"] == "override"

    @respx.mock
    async def test_raises_on_server_error(self, client):
        respx.post(f"{BASE_URL}/memory/event").mock(
            return_value=Response(500, json={"detail": "Internal error"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.encode(user_id="alice", content="Hello.")
        assert exc_info.value.status_code == 500
        assert "Internal error" in exc_info.value.detail


# ── build_context() ───────────────────────────────────────────────────────────

class TestBuildContext:
    @respx.mock
    async def test_returns_memory_context(self, client):
        respx.post(f"{BASE_URL}/context").mock(
            return_value=Response(200, json=CONTEXT_RESPONSE)
        )
        ctx = await client.build_context(user_id="alice", query="What editor?")
        assert isinstance(ctx, MemoryContext)
        assert ctx.user_id == "alice"
        assert ctx.query == "What editor does Alice use?"
        assert "Neovim" in ctx.context_text
        assert ctx.total_memories == 3
        assert ctx.embedding_failed is False

    @respx.mock
    async def test_messages_is_openai_format(self, client):
        respx.post(f"{BASE_URL}/context").mock(
            return_value=Response(200, json=CONTEXT_RESPONSE)
        )
        ctx = await client.build_context(user_id="alice", query="What editor?")
        assert isinstance(ctx.messages, list)
        assert ctx.messages[0]["role"] == "system"

    @respx.mock
    async def test_is_empty_when_zero_memories(self, client):
        response = {**CONTEXT_RESPONSE, "total_memories": 0}
        respx.post(f"{BASE_URL}/context").mock(
            return_value=Response(200, json=response)
        )
        ctx = await client.build_context(user_id="alice", query="Any?")
        assert ctx.is_empty() is True

    @respx.mock
    async def test_raises_on_404(self, client):
        respx.post(f"{BASE_URL}/context").mock(
            return_value=Response(404, json={"detail": "Not found"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.build_context(user_id="alice", query="?")
        assert exc_info.value.status_code == 404


# ── get_recent() ──────────────────────────────────────────────────────────────

class TestGetRecent:
    @respx.mock
    async def test_returns_list_of_recent_events(self, client):
        respx.get(f"{BASE_URL}/memory/alice").mock(
            return_value=Response(200, json=RECENT_RESPONSE)
        )
        events = await client.get_recent(user_id="alice")
        assert len(events) == 1
        assert isinstance(events[0], RecentEvent)
        assert events[0].event_id == "evt-001"
        assert events[0].raw_text == "I prefer dark mode."

    @respx.mock
    async def test_passes_limit_param(self, client):
        route = respx.get(f"{BASE_URL}/memory/alice").mock(
            return_value=Response(200, json=RECENT_RESPONSE)
        )
        await client.get_recent(user_id="alice", limit=5)
        assert "limit=5" in str(route.calls[0].request.url)

    @respx.mock
    async def test_passes_app_id_param(self, client):
        route = respx.get(f"{BASE_URL}/memory/alice").mock(
            return_value=Response(200, json=RECENT_RESPONSE)
        )
        await client.get_recent(user_id="alice", app_id="myapp")
        assert "app_id=myapp" in str(route.calls[0].request.url)

    @respx.mock
    async def test_returns_empty_list(self, client):
        empty = {"user_id": "alice", "app_id": "default", "events": []}
        respx.get(f"{BASE_URL}/memory/alice").mock(
            return_value=Response(200, json=empty)
        )
        events = await client.get_recent(user_id="alice")
        assert events == []


# ── health() ─────────────────────────────────────────────────────────────────

class TestHealth:
    @respx.mock
    async def test_returns_health_status(self, client):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=Response(200, json=HEALTH_RESPONSE)
        )
        status = await client.health()
        assert isinstance(status, HealthStatus)
        assert status.status == "ok"
        assert status.version == "0.1.0"

    @respx.mock
    async def test_raises_on_server_down(self, client):
        respx.get(f"{BASE_URL}/health").mock(
            return_value=Response(503, json={"detail": "Service unavailable"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.health()
        assert exc_info.value.status_code == 503


# ── SmritikoshError ───────────────────────────────────────────────────────────

class TestSmritikoshError:
    def test_str_includes_status_and_detail(self):
        err = SmritikoshError(status_code=422, detail="Validation error")
        assert "422" in str(err)
        assert "Validation error" in str(err)

    def test_attributes(self):
        err = SmritikoshError(status_code=500, detail="Boom")
        assert err.status_code == 500
        assert err.detail == "Boom"

    @respx.mock
    async def test_non_json_error_body_handled(self, client):
        respx.post(f"{BASE_URL}/memory/event").mock(
            return_value=Response(502, text="Bad Gateway")
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.encode(user_id="alice", content="Hi.")
        assert exc_info.value.status_code == 502


# ── Types ─────────────────────────────────────────────────────────────────────

class TestTypes:
    def test_encoded_event_fields(self):
        e = EncodedEvent(
            event_id="e1", user_id="u1",
            importance_score=0.8, facts_extracted=3, extraction_failed=False
        )
        assert e.event_id == "e1"

    def test_recent_event_fields(self):
        r = RecentEvent(
            event_id="e1", raw_text="hi",
            importance_score=0.5, consolidated=True, created_at="2024-01-01"
        )
        assert r.consolidated is True

    def test_memory_context_is_empty(self):
        ctx = MemoryContext(
            user_id="u1", query="q", context_text="",
            messages=[], total_memories=0, embedding_failed=False
        )
        assert ctx.is_empty() is True

    def test_memory_context_not_empty(self):
        ctx = MemoryContext(
            user_id="u1", query="q", context_text="text",
            messages=[], total_memories=5, embedding_failed=False
        )
        assert ctx.is_empty() is False

    def test_health_status_fields(self):
        h = HealthStatus(status="ok", version="0.1.0")
        assert h.status == "ok"

    def test_feedback_record_fields(self):
        f = FeedbackRecord(feedback_id="fb-1", event_id="evt-1", new_importance_score=0.7)
        assert f.feedback_id == "fb-1"
        assert f.new_importance_score == 0.7

    def test_identity_profile_fields(self):
        profile = IdentityProfile(
            user_id="u1", app_id="default", summary="",
            dimensions=[], beliefs=[], total_facts=0,
            computed_at="2026-03-15T00:00:00+00:00", is_empty=True,
        )
        assert profile.is_empty is True
        assert profile.beliefs == []


# ── submit_feedback() ─────────────────────────────────────────────────────────

class TestSubmitFeedback:
    @respx.mock
    async def test_returns_feedback_record(self, client):
        respx.post(f"{BASE_URL}/feedback").mock(
            return_value=Response(201, json=FEEDBACK_RESPONSE)
        )
        record = await client.submit_feedback(
            event_id="evt-001",
            user_id="alice",
            feedback_type="positive",
        )
        assert isinstance(record, FeedbackRecord)
        assert record.feedback_id == "fb-001"
        assert record.event_id == "evt-001"
        assert record.new_importance_score == 0.85

    @respx.mock
    async def test_sends_correct_payload(self, client):
        route = respx.post(f"{BASE_URL}/feedback").mock(
            return_value=Response(201, json=FEEDBACK_RESPONSE)
        )
        await client.submit_feedback(
            event_id="evt-001",
            user_id="alice",
            feedback_type="negative",
            app_id="myapp",
            comment="wrong memory",
        )
        import json
        payload = json.loads(route.calls[0].request.content)
        assert payload["event_id"] == "evt-001"
        assert payload["user_id"] == "alice"
        assert payload["feedback_type"] == "negative"
        assert payload["app_id"] == "myapp"
        assert payload["comment"] == "wrong memory"

    @respx.mock
    async def test_uses_client_app_id_by_default(self, client):
        route = respx.post(f"{BASE_URL}/feedback").mock(
            return_value=Response(201, json=FEEDBACK_RESPONSE)
        )
        await client.submit_feedback(
            event_id="evt-001", user_id="alice", feedback_type="neutral"
        )
        import json
        payload = json.loads(route.calls[0].request.content)
        assert payload["app_id"] == "default"

    @respx.mock
    async def test_raises_on_404(self, client):
        respx.post(f"{BASE_URL}/feedback").mock(
            return_value=Response(404, json={"detail": "Event not found"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.submit_feedback(
                event_id="evt-001", user_id="alice", feedback_type="positive"
            )
        assert exc_info.value.status_code == 404

    @respx.mock
    async def test_raises_on_422(self, client):
        respx.post(f"{BASE_URL}/feedback").mock(
            return_value=Response(422, json={"detail": "Invalid feedback_type"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.submit_feedback(
                event_id="evt-001", user_id="alice", feedback_type="great"
            )
        assert exc_info.value.status_code == 422


# ── get_identity() ────────────────────────────────────────────────────────────

class TestGetIdentity:
    @respx.mock
    async def test_returns_identity_profile(self, client):
        respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(200, json=IDENTITY_RESPONSE)
        )
        profile = await client.get_identity(user_id="alice")
        assert isinstance(profile, IdentityProfile)
        assert profile.user_id == "alice"
        assert profile.summary == "A founder building AI tools."
        assert profile.total_facts == 2
        assert profile.is_empty is False

    @respx.mock
    async def test_dimensions_parsed(self, client):
        respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(200, json=IDENTITY_RESPONSE)
        )
        profile = await client.get_identity(user_id="alice")
        assert len(profile.dimensions) == 1
        assert isinstance(profile.dimensions[0], IdentityDimensionItem)
        assert profile.dimensions[0].category == "role"
        assert profile.dimensions[0].dominant_value == "founder"
        assert profile.dimensions[0].fact_count == 2

    @respx.mock
    async def test_beliefs_parsed(self, client):
        respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(200, json=IDENTITY_RESPONSE)
        )
        profile = await client.get_identity(user_id="alice")
        assert len(profile.beliefs) == 1
        assert isinstance(profile.beliefs[0], BeliefItem)
        assert profile.beliefs[0].statement == "values speed over perfection"
        assert profile.beliefs[0].confidence == 0.85
        assert profile.beliefs[0].evidence_count == 3

    @respx.mock
    async def test_passes_app_id_param(self, client):
        route = respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(200, json=IDENTITY_RESPONSE)
        )
        await client.get_identity(user_id="alice", app_id="myapp")
        assert "app_id=myapp" in str(route.calls[0].request.url)

    @respx.mock
    async def test_empty_beliefs_list(self, client):
        response = {**IDENTITY_RESPONSE, "beliefs": [], "is_empty": True}
        respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(200, json=response)
        )
        profile = await client.get_identity(user_id="alice")
        assert profile.beliefs == []

    @respx.mock
    async def test_raises_on_500(self, client):
        respx.get(f"{BASE_URL}/identity/alice").mock(
            return_value=Response(500, json={"detail": "neo4j down"})
        )
        with pytest.raises(SmritikoshError) as exc_info:
            await client.get_identity(user_id="alice")
        assert exc_info.value.status_code == 500
