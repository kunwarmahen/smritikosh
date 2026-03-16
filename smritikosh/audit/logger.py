"""
AuditLogger — fire-and-forget provenance tracking for every pipeline step.

Every meaningful transformation in the Smritikosh pipeline emits an AuditEvent:
  memory.encoded          Raw text → episodic event (importance, embedding, facts)
  memory.facts_extracted  Structured facts upserted to Neo4j
  memory.consolidated     Batch of events → summary + distilled facts
  memory.reconsolidated   Event summary refined after recall
  memory.pruned           Low-value event deleted by SynapticPruner
  memory.clustered        Event assigned to a topic cluster
  belief.mined            Higher-order belief inferred from event patterns
  feedback.submitted      User feedback + importance score change
  context.built           Memory context assembled for an LLM call
  search.performed        Hybrid search executed

Design principles:
  - Fire-and-forget: emit() schedules writes via asyncio.create_task() so
    audit writes never block the main pipeline.
  - No-op fallback: if MongoDB is not configured (audit_logger is None),
    all emit() calls are silently dropped — zero overhead, zero breakage.
  - Single collection, discriminated by event_type — makes timeline queries
    across all event types easy with a single index scan.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Event type constants ───────────────────────────────────────────────────────

class EventType:
    MEMORY_ENCODED         = "memory.encoded"
    MEMORY_FACTS_EXTRACTED = "memory.facts_extracted"
    MEMORY_CONSOLIDATED    = "memory.consolidated"
    MEMORY_RECONSOLIDATED  = "memory.reconsolidated"
    MEMORY_PRUNED          = "memory.pruned"
    MEMORY_CLUSTERED       = "memory.clustered"
    BELIEF_MINED           = "belief.mined"
    FEEDBACK_SUBMITTED     = "feedback.submitted"
    CONTEXT_BUILT          = "context.built"
    SEARCH_PERFORMED       = "search.performed"


# ── Audit record ──────────────────────────────────────────────────────────────

@dataclass
class AuditEvent:
    """
    One immutable audit record.

    event_id links back to the episodic Event UUID where applicable.
    session_id groups related records within one pipeline run (e.g. all
    records produced by a single hippocampus.encode() call).
    payload holds event-type-specific data — deliberately untyped so each
    event type can carry exactly the fields it needs.
    """
    event_type: str
    user_id: str
    app_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str | None = None          # episodic event UUID (if applicable)
    session_id: str | None = None        # groups related events in one run
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_document(self) -> dict[str, Any]:
        """Serialise to a MongoDB-ready dict."""
        return {
            "_id": self.id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "app_id": self.app_id,
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


# ── AuditLogger ───────────────────────────────────────────────────────────────

class AuditLogger:
    """
    Thin async interface for writing audit events to MongoDB.

    Inject this into pipeline components (Hippocampus, Consolidator, etc.).
    If not injected (None), emit calls are no-ops — the system works without
    MongoDB configured.

    Usage:
        audit = AuditLogger(collection)

        # In a pipeline step:
        await audit.emit(AuditEvent(
            event_type=EventType.MEMORY_ENCODED,
            user_id="alice",
            app_id="myapp",
            event_id=str(event.id),
            session_id=session_id,
            payload={
                "raw_text_preview": raw_text[:200],
                "importance_score": 0.72,
                "embedding_success": True,
                "facts_extracted": 2,
            },
        ))
    """

    def __init__(self, collection: Any) -> None:
        """
        Args:
            collection: A motor AsyncIOMotorCollection instance.
        """
        self._col = collection

    async def emit(self, event: AuditEvent) -> None:
        """
        Write one audit event. Schedules the write as a background task so
        callers are never blocked by MongoDB latency.
        """
        asyncio.create_task(self._write(event))

    async def emit_sync(self, event: AuditEvent) -> None:
        """
        Write one audit event and await completion.
        Use this only in tests or when you need guaranteed delivery.
        """
        await self._write(event)

    async def _write(self, event: AuditEvent) -> None:
        try:
            await self._col.insert_one(event.to_document())
        except Exception as exc:
            # Never let audit failures propagate — log and continue
            logger.warning(
                "Audit write failed (non-fatal): %s", exc,
                extra={"event_type": event.event_type, "user_id": event.user_id},
            )

    # ── Query helpers (used by the audit API routes) ───────────────────────

    async def get_timeline(
        self,
        user_id: str,
        app_id: str = "default",
        event_type: str | None = None,
        event_id: str | None = None,
        from_ts: datetime | None = None,
        to_ts: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Return audit records for a user, newest first."""
        filt: dict[str, Any] = {"user_id": user_id, "app_id": app_id}
        if event_type:
            filt["event_type"] = event_type
        if event_id:
            filt["event_id"] = event_id
        ts_clause: dict[str, Any] = {}
        if from_ts:
            ts_clause["$gte"] = from_ts
        if to_ts:
            ts_clause["$lte"] = to_ts
        if ts_clause:
            filt["timestamp"] = ts_clause

        cursor = (
            self._col.find(filt, {"_id": 0})
            .sort("timestamp", -1)
            .skip(offset)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def get_event_lineage(self, event_id: str) -> list[dict]:
        """Return all audit records linked to one episodic event_id."""
        cursor = (
            self._col.find({"event_id": event_id}, {"_id": 0})
            .sort("timestamp", 1)
        )
        return await cursor.to_list(length=200)

    async def get_stats(
        self, user_id: str, app_id: str = "default"
    ) -> dict[str, int]:
        """Return count per event_type for a user."""
        pipeline = [
            {"$match": {"user_id": user_id, "app_id": app_id}},
            {"$group": {"_id": "$event_type", "count": {"$sum": 1}}},
        ]
        results: dict[str, int] = {}
        async for doc in self._col.aggregate(pipeline):
            results[doc["_id"]] = doc["count"]
        return results
