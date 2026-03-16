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

from datetime import datetime
from typing import Optional

import httpx

from smritikosh.sdk.types import (
    AdminJobResponse,
    AdminJobResult,
    BeliefItem,
    DeleteEventResult,
    DeleteProcedureResult,
    DeleteUserMemoryResult,
    DeleteUserProceduresResult,
    EncodedEvent,
    FeedbackRecord,
    HealthStatus,
    IdentityDimensionItem,
    IdentityProfile,
    IngestResult,
    MemoryContext,
    ProcedureCreated,
    ProcedureRecord,
    RecentEvent,
    ReconsolidationResult,
    SearchResult,
    SearchResultItem,
)

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
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
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
        payload: dict = {
            "user_id": user_id,
            "query": query,
            "app_id": app_id or self._app_id,
        }
        if from_date is not None:
            payload["from_date"] = from_date.isoformat()
        if to_date is not None:
            payload["to_date"] = to_date.isoformat()
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

    async def submit_feedback(
        self,
        *,
        event_id: str,
        user_id: str,
        feedback_type: str,
        app_id: str | None = None,
        comment: str | None = None,
    ) -> FeedbackRecord:
        """
        Submit feedback on a recalled memory event.

        Feedback immediately adjusts the event's importance_score, influencing
        how prominently it surfaces in future hybrid_search results.

        Args:
            event_id:      UUID of the recalled event being rated.
            user_id:       User submitting the feedback.
            feedback_type: ``"positive"``, ``"negative"``, or ``"neutral"``.
            app_id:        Application namespace. Defaults to the client-level app_id.
            comment:       Optional free-text note.

        Returns:
            :class:`FeedbackRecord` with the new importance_score.
        """
        payload = {
            "event_id": event_id,
            "user_id": user_id,
            "feedback_type": feedback_type,
            "app_id": app_id or self._app_id,
            "comment": comment,
        }
        data = await self._post("/feedback", payload)
        return FeedbackRecord(
            feedback_id=data["feedback_id"],
            event_id=data["event_id"],
            new_importance_score=data["new_importance_score"],
        )

    async def get_identity(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
    ) -> IdentityProfile:
        """
        Fetch the synthesized identity model for a user.

        Aggregates all semantic facts from Neo4j, groups them by category,
        and generates a narrative summary via LLM.

        Args:
            user_id: User whose identity to retrieve.
            app_id:  Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`IdentityProfile` with per-category dimensions and a narrative summary.
        """
        params = {"app_id": app_id or self._app_id}
        data = await self._get(f"/identity/{user_id}", params=params)
        return IdentityProfile(
            user_id=data["user_id"],
            app_id=data["app_id"],
            summary=data["summary"],
            dimensions=[
                IdentityDimensionItem(
                    category=d["category"],
                    dominant_value=d["dominant_value"],
                    confidence=d["confidence"],
                    fact_count=d["fact_count"],
                )
                for d in data["dimensions"]
            ],
            beliefs=[
                BeliefItem(
                    statement=b["statement"],
                    category=b["category"],
                    confidence=b["confidence"],
                    evidence_count=b["evidence_count"],
                )
                for b in data["beliefs"]
            ],
            total_facts=data["total_facts"],
            computed_at=data["computed_at"],
            is_empty=data["is_empty"],
        )

    async def store_procedure(
        self,
        *,
        user_id: str,
        trigger: str,
        instruction: str,
        app_id: str | None = None,
        category: str = "topic_response",
        priority: int = 5,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> ProcedureCreated:
        """
        Store a behavioral rule that fires when the trigger phrase is detected.

        Args:
            user_id:     User this rule applies to.
            trigger:     Topic/keyword phrase that activates the rule (e.g. ``"LLM deployment"``).
            instruction: What the AI should do when triggered.
            category:    ``"topic_response"`` | ``"communication"`` | ``"preference"`` | ``"domain_workflow"``.
            priority:    1 (low) – 10 (high). Higher priority rules appear first in context.

        Returns:
            :class:`ProcedureCreated` with the stored procedure ID.
        """
        payload = {
            "user_id": user_id,
            "trigger": trigger,
            "instruction": instruction,
            "app_id": app_id or self._app_id,
            "category": category,
            "priority": priority,
            "confidence": confidence,
            "source": source,
        }
        data = await self._post("/procedures", payload)
        return ProcedureCreated(
            procedure_id=data["procedure_id"],
            user_id=data["user_id"],
            trigger=data["trigger"],
            instruction=data["instruction"],
            category=data["category"],
            priority=data["priority"],
            is_active=data["is_active"],
            hit_count=data["hit_count"],
            confidence=data["confidence"],
            source=data["source"],
            created_at=data["created_at"],
        )

    async def list_procedures(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
        active_only: bool = True,
        category: str | None = None,
    ) -> list[ProcedureRecord]:
        """
        List all behavioral rules for a user.

        Args:
            user_id:    User whose rules to fetch.
            active_only: If True (default), only return active rules.
            category:   Filter to a specific category if provided.

        Returns:
            List of :class:`ProcedureRecord`, ordered by priority descending.
        """
        params: dict = {"app_id": app_id or self._app_id, "active_only": active_only}
        if category is not None:
            params["category"] = category
        data = await self._get(f"/procedures/{user_id}", params=params)
        return [
            ProcedureRecord(
                procedure_id=item["procedure_id"],
                trigger=item["trigger"],
                instruction=item["instruction"],
                category=item["category"],
                priority=item["priority"],
                is_active=item["is_active"],
                hit_count=item["hit_count"],
            )
            for item in data["procedures"]
        ]

    async def delete_procedure(
        self,
        *,
        procedure_id: str,
    ) -> DeleteProcedureResult:
        """Delete a specific behavioral rule by ID."""
        data = await self._delete(f"/procedures/{procedure_id}")
        return DeleteProcedureResult(
            deleted=data["deleted"], procedure_id=data["procedure_id"]
        )

    async def delete_user_procedures(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
    ) -> DeleteUserProceduresResult:
        """Delete all behavioral rules for a user within an app namespace."""
        params = {"app_id": app_id or self._app_id}
        data = await self._delete(f"/procedures/user/{user_id}", params=params)
        return DeleteUserProceduresResult(
            procedures_deleted=data["procedures_deleted"],
            user_id=data["user_id"],
            app_id=data["app_id"],
        )

    async def delete_event(
        self,
        *,
        event_id: str,
    ) -> DeleteEventResult:
        """
        Delete a specific memory event by ID.

        Args:
            event_id: UUID of the event to delete.

        Returns:
            :class:`DeleteEventResult` with ``deleted=True`` if the event existed.
        """
        data = await self._delete(f"/memory/event/{event_id}")
        return DeleteEventResult(deleted=data["deleted"], event_id=data["event_id"])

    async def delete_user_memory(
        self,
        *,
        user_id: str,
        app_id: str | None = None,
    ) -> DeleteUserMemoryResult:
        """
        Delete all memory events for a user within an app namespace.

        Args:
            user_id: User whose events should be deleted.
            app_id:  Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`DeleteUserMemoryResult` with the number of events removed.
        """
        params = {"app_id": app_id or self._app_id}
        data = await self._delete(f"/memory/user/{user_id}", params=params)
        return DeleteUserMemoryResult(
            events_deleted=data["events_deleted"],
            user_id=data["user_id"],
            app_id=data["app_id"],
        )

    async def reconsolidate(
        self,
        *,
        event_id: str,
        query: str,
        user_id: str,
    ) -> ReconsolidationResult:
        """
        Manually reconsolidate a specific memory event.

        Refines the event's summary by incorporating the recall context (``query``).
        The same gate conditions apply as automatic reconsolidation:
        the event must have been recalled at least twice, have sufficient importance,
        and not have been reconsolidated within the cooldown window.

        Args:
            event_id: UUID of the event to reconsolidate.
            query:    The context in which this memory was recalled.
            user_id:  Owner of the event.

        Returns:
            :class:`ReconsolidationResult` with ``updated=True`` if the summary changed.
        """
        payload = {"event_id": event_id, "query": query, "user_id": user_id}
        data = await self._post("/admin/reconsolidate", payload)
        return ReconsolidationResult(
            event_id=data["event_id"],
            user_id=data["user_id"],
            updated=data["updated"],
            skipped=data["skipped"],
            skip_reason=data.get("skip_reason", ""),
            old_summary=data.get("old_summary", ""),
            new_summary=data.get("new_summary", ""),
        )

    async def admin_consolidate(
        self,
        *,
        user_id: str | None = None,
        app_id: str | None = None,
    ) -> AdminJobResponse:
        """Trigger memory consolidation. Pass ``user_id`` to target one user."""
        payload: dict = {"app_id": app_id or self._app_id}
        if user_id:
            payload["user_id"] = user_id
        data = await self._post("/admin/consolidate", payload)
        return _parse_admin_response(data)

    async def admin_prune(
        self,
        *,
        user_id: str | None = None,
        app_id: str | None = None,
    ) -> AdminJobResponse:
        """Trigger synaptic pruning. Pass ``user_id`` to target one user."""
        payload: dict = {"app_id": app_id or self._app_id}
        if user_id:
            payload["user_id"] = user_id
        data = await self._post("/admin/prune", payload)
        return _parse_admin_response(data)

    async def admin_cluster(
        self,
        *,
        user_id: str | None = None,
        app_id: str | None = None,
    ) -> AdminJobResponse:
        """Trigger memory clustering. Pass ``user_id`` to target one user."""
        payload: dict = {"app_id": app_id or self._app_id}
        if user_id:
            payload["user_id"] = user_id
        data = await self._post("/admin/cluster", payload)
        return _parse_admin_response(data)

    async def admin_mine_beliefs(
        self,
        *,
        user_id: str | None = None,
        app_id: str | None = None,
    ) -> AdminJobResponse:
        """Trigger belief mining. Pass ``user_id`` to target one user."""
        payload: dict = {"app_id": app_id or self._app_id}
        if user_id:
            payload["user_id"] = user_id
        data = await self._post("/admin/mine-beliefs", payload)
        return _parse_admin_response(data)

    async def search(
        self,
        *,
        user_id: str,
        query: str,
        app_id: str | None = None,
        limit: int = 10,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> SearchResult:
        """
        Hybrid search over a user's episodic memory.

        Returns raw scored events (similarity, recency, hybrid scores) rather
        than the pre-formatted context block returned by ``build_context()``.
        Useful for building custom memory UIs or filtering/ranking logic.

        Args:
            user_id:   User to search.
            query:     Natural-language search query.
            app_id:    Application namespace. Defaults to the client-level app_id.
            limit:     Maximum events to return (1–50).
            from_date: Only include events on or after this datetime.
            to_date:   Only include events on or before this datetime.

        Returns:
            :class:`SearchResult` with a list of scored :class:`SearchResultItem`.
        """
        payload: dict = {
            "user_id": user_id,
            "query": query,
            "app_id": app_id or self._app_id,
            "limit": limit,
        }
        if from_date is not None:
            payload["from_date"] = from_date.isoformat()
        if to_date is not None:
            payload["to_date"] = to_date.isoformat()
        data = await self._post("/memory/search", payload)
        return SearchResult(
            user_id=data["user_id"],
            query=data["query"],
            results=[
                SearchResultItem(
                    event_id=r["event_id"],
                    raw_text=r["raw_text"],
                    importance_score=r["importance_score"],
                    hybrid_score=r["hybrid_score"],
                    similarity_score=r["similarity_score"],
                    recency_score=r["recency_score"],
                    consolidated=r["consolidated"],
                    created_at=r["created_at"],
                )
                for r in data["results"]
            ],
            total=data["total"],
            embedding_failed=data["embedding_failed"],
        )

    async def ingest_push(
        self,
        *,
        user_id: str,
        content: str,
        source: str = "api",
        source_id: str = "",
        app_id: str | None = None,
        metadata: dict | None = None,
    ) -> IngestResult:
        """
        Push a single event from an external source.

        Equivalent to the server's ``POST /ingest/push`` endpoint — useful
        for programmatic ingestion from webhooks or backend services.

        Args:
            user_id:   Owner of the memory.
            content:   Raw text to store.
            source:    Label for the source (e.g. ``"github"``, ``"jira"``).
            source_id: Optional unique identifier within the source.
            app_id:    Application namespace. Defaults to the client-level app_id.
            metadata:  Optional extra key/value context.

        Returns:
            :class:`IngestResult` with ingestion counts and stored event IDs.
        """
        payload = {
            "user_id": user_id,
            "content": content,
            "source": source,
            "source_id": source_id,
            "app_id": app_id or self._app_id,
            "metadata": metadata or {},
        }
        data = await self._post("/ingest/push", payload)
        return _parse_ingest_result(data)

    async def ingest_file(
        self,
        *,
        user_id: str,
        file_content: bytes,
        filename: str,
        app_id: str | None = None,
    ) -> IngestResult:
        """
        Ingest a file as episodic memories.

        Supported formats: ``.txt``, ``.md`` (paragraph chunks), ``.csv``
        (one event per row), ``.json`` (array of strings or objects).

        Args:
            user_id:      Owner of the memories.
            file_content: Raw file bytes.
            filename:     Original filename including extension (used for format detection).
            app_id:       Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`IngestResult` with ingestion counts and stored event IDs.
        """
        client = self._ensure_open()
        response = await client.post(
            "/ingest/file",
            data={"user_id": user_id, "app_id": app_id or self._app_id},
            files={"file": (filename, file_content)},
        )
        return _parse_ingest_result(self._raise_or_json(response))

    async def ingest_email(
        self,
        *,
        user_id: str,
        host: str,
        port: int = 993,
        username: str,
        password: str,
        mailbox: str = "INBOX",
        limit: int = 20,
        unseen_only: bool = True,
        app_id: str | None = None,
    ) -> IngestResult:
        """
        Fetch and ingest unread emails from an IMAP mailbox.

        Credentials are used per-request and never stored on the server.

        Args:
            user_id:     Owner of the memories.
            host:        IMAP server hostname (e.g. ``"imap.gmail.com"``).
            port:        IMAP SSL port (default: 993).
            username:    IMAP account username / email address.
            password:    IMAP account password or app password.
            mailbox:     Mailbox to fetch from (default: ``"INBOX"``).
            limit:       Maximum emails to fetch.
            unseen_only: If True (default), fetch only UNSEEN messages.
            app_id:      Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`IngestResult` with ingestion counts and stored event IDs.
        """
        payload = {
            "user_id": user_id,
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "mailbox": mailbox,
            "limit": limit,
            "unseen_only": unseen_only,
            "app_id": app_id or self._app_id,
        }
        data = await self._post("/ingest/email/sync", payload)
        return _parse_ingest_result(data)

    async def ingest_calendar(
        self,
        *,
        user_id: str,
        file_content: bytes,
        filename: str = "calendar.ics",
        app_id: str | None = None,
    ) -> IngestResult:
        """
        Ingest calendar events from an iCalendar (``.ics``) file.

        Each VEVENT becomes one memory event containing the summary,
        description, location, and time range.

        Args:
            user_id:      Owner of the memories.
            file_content: Raw ``.ics`` file bytes.
            filename:     Filename to use when uploading (default: ``"calendar.ics"``).
            app_id:       Application namespace. Defaults to the client-level app_id.

        Returns:
            :class:`IngestResult` with ingestion counts and stored event IDs.
        """
        client = self._ensure_open()
        response = await client.post(
            "/ingest/calendar",
            data={"user_id": user_id, "app_id": app_id or self._app_id},
            files={"file": (filename, file_content)},
        )
        return _parse_ingest_result(self._raise_or_json(response))

    async def health(self) -> HealthStatus:
        """
        Check server health, including database connectivity.

        Returns:
            :class:`HealthStatus` with ``status="ok"`` when both PostgreSQL and
            Neo4j are reachable.  ``status="degraded"`` when the server is up
            but a database is unreachable.

        Raises:
            :class:`SmritikoshError` if the server is unreachable.
        """
        data = await self._get("/health")
        return HealthStatus(
            status=data["status"],
            version=data["version"],
            postgres=data.get("postgres", "unknown"),
            neo4j=data.get("neo4j", "unknown"),
        )

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

    async def _delete(self, path: str, params: dict | None = None) -> dict:
        client = self._ensure_open()
        response = await client.delete(path, params=params)
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


# ── Module helpers ─────────────────────────────────────────────────────────────


def _parse_ingest_result(data: dict) -> IngestResult:
    return IngestResult(
        source=data["source"],
        events_ingested=data["events_ingested"],
        events_failed=data["events_failed"],
        event_ids=data["event_ids"],
    )


def _parse_admin_response(data: dict) -> AdminJobResponse:
    return AdminJobResponse(
        job=data["job"],
        users_processed=data["users_processed"],
        results=[
            AdminJobResult(
                user_id=r["user_id"],
                app_id=r["app_id"],
                skipped=r["skipped"],
                detail=r.get("detail", ""),
            )
            for r in data["results"]
        ],
    )
