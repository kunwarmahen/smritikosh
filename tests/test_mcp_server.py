"""
Tests for the Smritikosh MCP server (smritikosh.mcp.server).

Uses respx to mock the Smritikosh HTTP API — no real server or MCP
transport needed. Tools are exercised as plain async functions with the
lifespan-managed state driven manually.
"""

import json

import pytest
import respx
from httpx import Response

from smritikosh.mcp import server as mcp_server
from smritikosh.mcp.server import (
    config_from_env,
    get_context,
    mcp,
    recall,
    store_memory,
)
from smritikosh.sdk.client import SmritikoshError

BASE_URL = "http://localhost:8080"

ENCODE_RESPONSE = {
    "event_id": "evt-001",
    "user_id": "alice",
    "importance_score": 0.75,
    "facts_extracted": 2,
    "extraction_failed": False,
}

SEARCH_RESPONSE = {
    "user_id": "alice",
    "query": "editor",
    "results": [
        {
            "event_id": "evt-001",
            "raw_text": "I prefer dark mode and use Neovim.",
            "importance_score": 0.6,
            "hybrid_score": 0.82,
            "similarity_score": 0.9,
            "recency_score": 0.5,
            "consolidated": False,
            "created_at": "2026-01-01T10:00:00+00:00",
        },
    ],
    "total": 1,
    "embedding_failed": False,
}

CONTEXT_RESPONSE = {
    "user_id": "alice",
    "query": "What editor does Alice use?",
    "context_text": "## User Memory Context\nAlice uses Neovim.",
    "messages": [{"role": "system", "content": "Alice uses Neovim."}],
    "total_memories": 3,
    "embedding_failed": False,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mcp_env(monkeypatch):
    monkeypatch.setenv("SMRITIKOSH_API_KEY", "sk-smriti-test-key")
    monkeypatch.setenv("SMRITIKOSH_USER_ID", "alice")
    monkeypatch.setenv("SMRITIKOSH_BASE_URL", BASE_URL)
    monkeypatch.setenv("SMRITIKOSH_APP_ID", "default")


@pytest.fixture
async def mcp_state(mcp_env):
    """Run the FastMCP lifespan so the module-level state is populated."""
    async with mcp_server._lifespan(mcp) as state:
        yield state


# ── Configuration ─────────────────────────────────────────────────────────────


def test_config_from_env(mcp_env):
    config = config_from_env()
    assert config.api_key == "sk-smriti-test-key"
    assert config.user_id == "alice"
    assert config.base_url == BASE_URL
    assert config.app_id == "default"


def test_config_strips_trailing_slash(mcp_env, monkeypatch):
    monkeypatch.setenv("SMRITIKOSH_BASE_URL", BASE_URL + "/")
    assert config_from_env().base_url == BASE_URL


def test_config_requires_api_key(mcp_env, monkeypatch):
    monkeypatch.delenv("SMRITIKOSH_API_KEY")
    with pytest.raises(RuntimeError, match="SMRITIKOSH_API_KEY"):
        config_from_env()


def test_config_requires_user_id(mcp_env, monkeypatch):
    monkeypatch.setenv("SMRITIKOSH_USER_ID", "  ")
    with pytest.raises(RuntimeError, match="SMRITIKOSH_USER_ID"):
        config_from_env()


def test_config_defaults(mcp_env, monkeypatch):
    monkeypatch.delenv("SMRITIKOSH_BASE_URL")
    monkeypatch.delenv("SMRITIKOSH_APP_ID")
    config = config_from_env()
    assert config.base_url == "http://localhost:8080"
    assert config.app_id == "default"


# ── Lifespan ──────────────────────────────────────────────────────────────────


async def test_lifespan_sets_and_clears_state(mcp_env):
    assert mcp_server._state is None
    async with mcp_server._lifespan(mcp) as state:
        assert mcp_server._state is state
        assert state.config.user_id == "alice"
    assert mcp_server._state is None


async def test_tools_fail_without_lifespan():
    with pytest.raises(RuntimeError, match="not initialized"):
        await store_memory(content="anything")


# ── Tool registration ─────────────────────────────────────────────────────────


async def test_three_tools_registered():
    tools = {t.name for t in await mcp.list_tools()}
    assert tools == {"store_memory", "recall", "get_context"}


# ── store_memory ──────────────────────────────────────────────────────────────


@respx.mock
async def test_store_memory(mcp_state):
    route = respx.post(f"{BASE_URL}/memory/event").mock(
        return_value=Response(200, json=ENCODE_RESPONSE)
    )

    result = await store_memory(content="I prefer dark mode.", metadata={"source": "test"})

    assert result["event_id"] == "evt-001"
    assert result["facts_extracted"] == 2

    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer sk-smriti-test-key"
    payload = json.loads(request.content)
    assert payload["user_id"] == "alice"
    assert payload["app_id"] == "default"
    assert payload["metadata"] == {"source": "test"}


@respx.mock
async def test_store_memory_user_override(mcp_state):
    respx.post(f"{BASE_URL}/memory/event").mock(
        return_value=Response(200, json={**ENCODE_RESPONSE, "user_id": "bob"})
    )

    result = await store_memory(content="Bob likes tabs.", user_id="bob")

    assert result["user_id"] == "bob"


@respx.mock
async def test_store_memory_propagates_api_error(mcp_state):
    respx.post(f"{BASE_URL}/memory/event").mock(
        return_value=Response(403, json={"detail": "This API key does not have write access."})
    )

    with pytest.raises(SmritikoshError, match="403"):
        await store_memory(content="anything")


# ── recall ────────────────────────────────────────────────────────────────────


@respx.mock
async def test_recall(mcp_state):
    route = respx.post(f"{BASE_URL}/memory/search").mock(
        return_value=Response(200, json=SEARCH_RESPONSE)
    )

    result = await recall(query="editor", limit=5)

    assert result["total"] == 1
    assert result["results"][0]["raw_text"] == "I prefer dark mode and use Neovim."
    assert result["results"][0]["hybrid_score"] == 0.82

    payload = json.loads(route.calls.last.request.content)
    assert payload == {"user_id": "alice", "query": "editor", "app_ids": ["default"], "limit": 5}


# ── get_context ───────────────────────────────────────────────────────────────


@respx.mock
async def test_get_context(mcp_state):
    route = respx.post(f"{BASE_URL}/context").mock(
        return_value=Response(200, json=CONTEXT_RESPONSE)
    )

    result = await get_context(query="What editor does Alice use?")

    assert result["context_text"].startswith("## User Memory Context")
    assert result["total_memories"] == 3

    payload = json.loads(route.calls.last.request.content)
    assert payload["user_id"] == "alice"
    assert payload["query"] == "What editor does Alice use?"
