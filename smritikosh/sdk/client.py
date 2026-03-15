"""
SmritikoshClient — async Python SDK for the Smritikosh memory API.

Usage (async context manager):

    async with SmritikoshClient(base_url="http://localhost:8080") as client:
        # Store a memory
        event = await client.encode(
            user_id="alice",
            content="I prefer dark mode and use Neovim.",
        )

        # Build context before an LLM call
        ctx = await client.build_context(
            user_id="alice",
            query="What editor does Alice use?",
        )
        response = await llm.complete(ctx.messages + user_messages)

        # Browse recent events
        events = await client.get_recent(user_id="alice", limit=5)

Usage (manual lifecycle):

    client = SmritikoshClient(base_url="http://localhost:8080")
    await client.aopen()
    try:
        ...
    finally:
        await client.aclose()
"""

from __future__ import annotations

import httpx

from smritikosh.sdk.types import EncodedEvent, HealthStatus, MemoryContext, RecentEvent

_DEFAULT_TIMEOUT = 30.0   # seconds


class SmritikoshError(Exception):
    """Raised when the API returns a non-2xx status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class SmritikoshClient:
    """
    Async HTTP client for the Smritikosh memory API.

    Args:
        base_url:   Base URL of the running Smritikosh server
                    (e.g. ``"http://localhost:8080"``).
        app_id:     Default application namespace. Can be overridden
                    per-call. Use different app_ids to isolate memory
                    for multiple applications sharing one server.
        timeout:    Per-request timeout in seconds (default: 30).
        headers:    Extra headers to send with every request
                    (e.g. ``{"Authorization": "Bearer <token>"}``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        app_id: str = "default",
        timeout: float = _DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._app_id = app_id
        self._timeout = timeout
        self._extra_headers = headers or {}
        self._client: httpx.AsyncClient | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def aopen(self) -> None:
        """Open the underlying HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers={"Content-Type": "application/json", **self._extra_headers},
        )

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "SmritikoshClient":
        await self.aopen()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    # ── API methods ───────────────────────────────────────────────────────────

    async def encode(
        self,
        *,
        user_id: str,
        content: str,
        app_id: str | None = None,
        metadata: dict | None = None,
    ) -> EncodedEvent:
        """
        Store a user interaction in episodic memory.

        The server runs the full Hippocampus pipeline:
        importance scoring → embedding → fact extraction → storage.

        Args:
            user_id:  Unique identifier for the user.
            content:  Raw interaction text (conversation turn, note, etc.).
            app_id:   Application namespace. Defaults to the client-level app_id.
            metadata: Optional extra context (``{"source": "slack", "channel": "#general"}``).

        Returns:
            :class:`EncodedEvent` with the stored event ID and extraction summary.
        """
        payload = {
            "user_id": user_id,
            "content": content,
            "app_id": app_id or self._app_id,
            "metadata": metadata or {},
        }
        data = await self._post("/memory/event", payload)
        return EncodedEvent(
            event_id=data["event_id"],
            user_id=data["user_id"],
            importance_score=data["importance_score"],
            facts_extracted=data["facts_extracted"],
            extraction_failed=data["extraction_failed"],
        )

    async def build_context(
        self,
        *,
        user_id: str,
        query: str,
        app_id: str | None = None,
    ) -> MemoryContext:
        """
        Retrieve a memory context block for a user query.

        Uses hybrid search (vector similarity + recency + importance) to
        find the most relevant past events and combines them with the user's
        semantic profile (facts from Neo4j).

        The returned :class:`MemoryContext` can be injected directly into an
        LLM call via ``context.messages`` (OpenAI-style) or ``context.context_text``
        (plain string for custom prompt building).

        Args:
            user_id: User to retrieve memory for.
            query:   The current question or topic (used for similarity search).
            app_id:  Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`MemoryContext` — ready-to-use context block.
        """
        payload = {
            "user_id": user_id,
            "query": query,
            "app_id": app_id or self._app_id,
        }
        data = await self._post("/context", payload)
        return MemoryContext(
            user_id=data["user_id"],
            query=data["query"],
            context_text=data["context_text"],
            messages=data["messages"],
            total_memories=data["total_memories"],
            embedding_failed=data["embedding_failed"],
        )

    async def get_recent(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
        limit: int = 10,
    ) -> list[RecentEvent]:
        """
        Fetch the most recent memory events for a user.

        Args:
            user_id: User whose events to retrieve.
            app_id:  Application namespace. Defaults to the client-level app_id.
            limit:   Maximum events to return (1–50).

        Returns:
            List of :class:`RecentEvent`, newest first.
        """
        params = {
            "app_id": app_id or self._app_id,
            "limit": limit,
        }
        data = await self._get(f"/memory/{user_id}", params=params)
        return [
            RecentEvent(
                event_id=item["event_id"],
                raw_text=item["raw_text"],
                importance_score=item["importance_score"],
                consolidated=item["consolidated"],
                created_at=item["created_at"],
            )
            for item in data["events"]
        ]

    async def health(self) -> HealthStatus:
        """
        Check server health.

        Returns:
            :class:`HealthStatus` with ``status="ok"`` when the server is running.

        Raises:
            :class:`SmritikoshError` if the server is unreachable or unhealthy.
        """
        data = await self._get("/health")
        return HealthStatus(status=data["status"], version=data["version"])

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _ensure_open(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "SmritikoshClient is not open. "
                "Use 'async with SmritikoshClient(...) as client:' "
                "or call 'await client.aopen()' first."
            )
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        client = self._ensure_open()
        response = await client.post(path, json=payload)
        return self._raise_or_json(response)

    async def _get(self, path: str, params: dict | None = None) -> dict:
        client = self._ensure_open()
        response = await client.get(path, params=params)
        return self._raise_or_json(response)

    @staticmethod
    def _raise_or_json(response: httpx.Response) -> dict:
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise SmritikoshError(status_code=response.status_code, detail=detail)
        return response.json()
