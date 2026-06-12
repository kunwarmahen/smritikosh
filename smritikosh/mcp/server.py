"""
Smritikosh MCP server.

A thin Model Context Protocol wrapper over the Python SDK
(:class:`smritikosh.sdk.client.SmritikoshClient`). Exposes three tools:

    store_memory  — save a fact/preference/event to episodic memory
    recall        — hybrid search over a user's memories
    get_context   — build a ready-to-inject memory context block for a query

Configuration (environment variables):

    SMRITIKOSH_API_KEY        required — API key (``sk-smriti-…``), sent as a
                              Bearer token. Determines the accessible user and
                              app namespaces; ``store_memory`` needs the
                              ``write`` scope.
    SMRITIKOSH_USER_ID        required — default user whose memory the tools
                              act on. Non-admin keys may only act on their own
                              username; admin keys can pass ``user_id`` per call.
    SMRITIKOSH_BASE_URL       default ``http://localhost:8080``
    SMRITIKOSH_APP_ID         default ``default``
    SMRITIKOSH_MCP_TRANSPORT  default ``stdio`` (also: ``sse``, ``streamable-http``)

Run directly::

    smritikosh-mcp                      # console script (pyproject entry point)
    python -m smritikosh.mcp.server     # equivalent

Claude Code registration::

    claude mcp add smritikosh \
      -e SMRITIKOSH_API_KEY=sk-smriti-... \
      -e SMRITIKOSH_USER_ID=alice \
      -- smritikosh-mcp
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

from smritikosh.sdk.client import SmritikoshClient

_DEFAULT_BASE_URL = "http://localhost:8080"


@dataclass
class MCPConfig:
    """Resolved server configuration (from environment variables)."""

    base_url: str
    api_key: str
    app_id: str
    user_id: str


@dataclass
class MCPState:
    """Lifespan state: one open SDK client for the server's lifetime."""

    client: SmritikoshClient
    config: MCPConfig


# Set by the lifespan; module-level so tool functions are plain callables
# that tests can drive without an MCP transport.
_state: MCPState | None = None


def config_from_env() -> MCPConfig:
    """Build an :class:`MCPConfig` from environment variables.

    Raises ``RuntimeError`` with a setup hint when a required variable is
    missing, so misconfiguration fails at startup rather than on first tool call.
    """
    api_key = os.environ.get("SMRITIKOSH_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "SMRITIKOSH_API_KEY is not set. Generate one in the Smritikosh "
            "dashboard (or POST /keys) and export it before starting the MCP server."
        )
    user_id = os.environ.get("SMRITIKOSH_USER_ID", "").strip()
    if not user_id:
        raise RuntimeError(
            "SMRITIKOSH_USER_ID is not set. Set it to the username the API key "
            "belongs to (non-admin keys may only access their own user's memory)."
        )
    return MCPConfig(
        base_url=os.environ.get("SMRITIKOSH_BASE_URL", _DEFAULT_BASE_URL).rstrip("/"),
        api_key=api_key,
        app_id=os.environ.get("SMRITIKOSH_APP_ID", "default"),
        user_id=user_id,
    )


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[MCPState]:
    global _state
    config = config_from_env()
    async with SmritikoshClient(
        base_url=config.base_url,
        app_id=config.app_id,
        headers={"Authorization": f"Bearer {config.api_key}"},
    ) as client:
        _state = MCPState(client=client, config=config)
        try:
            yield _state
        finally:
            _state = None


def _require_state() -> MCPState:
    if _state is None:
        raise RuntimeError("Smritikosh MCP server is not initialized (lifespan not running).")
    return _state


mcp = FastMCP(
    "smritikosh",
    instructions=(
        "Smritikosh is a persistent memory layer. Use get_context at the start of "
        "a task to load what is already known about the user, recall to search for "
        "specific memories, and store_memory whenever the user shares durable "
        "information worth remembering across sessions."
    ),
    lifespan=_lifespan,
)


@mcp.tool()
async def store_memory(
    content: str,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Save a memory to Smritikosh's long-term store.

    Call this when the user shares durable information worth remembering across
    sessions: preferences, facts about themselves or their work, decisions,
    corrections, or notable events. Do not store transient task chatter.

    Args:
        content:  The information to remember, phrased as a standalone statement
                  (e.g. "Alice prefers dark mode and uses Neovim").
        metadata: Optional extra context, e.g. {"source": "claude-code"}.
        user_id:  Override the default user (admin API keys only).

    Returns the stored event id, an importance score (0-1), and how many
    semantic facts were extracted from the content.
    """
    state = _require_state()
    event = await state.client.encode(
        user_id=user_id or state.config.user_id,
        content=content,
        metadata=metadata,
    )
    return asdict(event)


@mcp.tool()
async def recall(
    query: str,
    limit: int = 10,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Search the user's stored memories.

    Call this when you need to check what is already known about a specific
    topic — past decisions, stated preferences, prior conversations — before
    answering or acting. Uses hybrid ranking (semantic similarity + recency +
    importance).

    Args:
        query:   Natural-language description of what to look for.
        limit:   Maximum memories to return (1-50, default 10).
        user_id: Override the default user (admin API keys only).

    Returns scored memory events, most relevant first.
    """
    state = _require_state()
    result = await state.client.search(
        user_id=user_id or state.config.user_id,
        query=query,
        limit=limit,
    )
    return asdict(result)


@mcp.tool()
async def get_context(
    query: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Build a ready-to-use memory context block for the current task.

    Call this at the start of a task or conversation to load everything
    relevant that Smritikosh knows about the user: it combines the most
    relevant past events with the user's semantic profile (facts, identity)
    into a single context block.

    Args:
        query:   The current question, task, or topic.
        user_id: Override the default user (admin API keys only).

    Returns ``context_text`` (a plain-text block to ground your response in)
    plus the total number of memories it draws on.
    """
    state = _require_state()
    context = await state.client.build_context(
        user_id=user_id or state.config.user_id,
        query=query,
    )
    return asdict(context)


def main() -> None:
    """Entry point for the ``smritikosh-mcp`` console script."""
    transport = os.environ.get("SMRITIKOSH_MCP_TRANSPORT", "stdio")
    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
