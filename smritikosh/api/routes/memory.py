"""
Memory routes — capture and retrieve episodic events.

POST /memory/event      Encode a raw interaction into memory (Hippocampus.encode)
POST /memory/search     Hybrid search — returns raw scored events
GET  /memory/{user_id}  Return recent events for a user
"""

import json
import logging
import uuid
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import or_, select
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.ratelimit import limiter
from smritikosh.auth.deps import assert_app_access, assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_audit_logger, get_hippocampus, get_episodic, get_llm
from smritikosh.config import settings
from smritikosh.api.schemas import (
    DeleteEventResponse,
    DeleteUserMemoryResponse,
    EventRequest,
    EventResponse,
    ExportEventItem,
    MemoryEventDetail,
    MemoryLinkItem,
    MemoryLinksResponse,
    RecentEventItem,
    RecentEventsResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.db.models import Event, MemoryLink
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/event", response_model=EventResponse, status_code=201)
@limiter.limit(lambda: settings.rate_limit_encode or "10000/minute")
async def capture_event(
    request: Request,
    body: EventRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> EventResponse:
    """
    Encode a user interaction into persistent memory.

    The Hippocampus pipeline runs:
      1. Amygdala scores importance.
      2. Embedding + fact extraction run concurrently.
      3. Event is stored to PostgreSQL (EpisodicMemory).
      4. Extracted facts are upserted to Neo4j (SemanticMemory).

    Returns immediately with the stored event ID and extraction summary.
    If fact extraction fails the event is still stored (extraction_failed=True).
    """
    assert_self_or_admin(current_user, body.user_id)
    assert_app_access(current_user, body.app_id)
    try:
        result = await hippocampus.encode(
            pg,
            neo,
            user_id=body.user_id,
            raw_text=body.content,
            app_id=body.app_id,
            metadata=body.metadata,
        )
    except Exception as exc:
        logger.exception("Hippocampus encode failed", extra={"user_id": body.user_id})
        raise HTTPException(status_code=500, detail=f"Memory encoding failed: {exc}") from exc

    return EventResponse(
        event_id=str(result.event.id),
        user_id=body.user_id,
        importance_score=result.importance_score,
        facts_extracted=len(result.facts),
        extraction_failed=result.extraction_failed,
    )


@router.delete("/event/{event_id}", response_model=DeleteEventResponse)
async def delete_event(
    event_id: str,
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> DeleteEventResponse:
    """
    Delete a specific memory event by ID.

    Returns ``deleted=true`` if the event existed and was removed,
    ``deleted=false`` if no event with that ID was found.
    """
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid event_id UUID format.")

    # Load first to check ownership
    result = await pg.execute(select(Event).where(Event.id == eid))
    event = result.scalar_one_or_none()
    if event:
        assert_self_or_admin(current_user, event.user_id)

    deleted = await episodic.delete(pg, eid)
    return DeleteEventResponse(deleted=deleted, event_id=event_id)


@router.delete("/user/{user_id}", response_model=DeleteUserMemoryResponse)
async def delete_user_memory(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> DeleteUserMemoryResponse:
    """
    Delete all memory events for a user within an app namespace.

    Use with care — this removes all episodic events for the user.
    Semantic facts in Neo4j are not affected by this endpoint.
    """
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)
    count = await episodic.delete_all_for_user(pg, user_id, app_id)
    logger.info(
        "Deleted all user memory",
        extra={"user_id": user_id, "app_id": app_id, "events_deleted": count},
    )
    return DeleteUserMemoryResponse(events_deleted=count, user_id=user_id, app_id=app_id)


@router.post("/search", response_model=SearchResponse)
@limiter.limit(lambda: settings.rate_limit_search or "10000/minute")
async def search_memory(
    request: Request,
    body: SearchRequest,
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)],
    llm: Annotated[LLMAdapter, Depends(get_llm)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> SearchResponse:
    """
    Hybrid search over a user's episodic memory.

    Embeds the query, then scores all events using the same weighted formula
    as ``/context`` (cosine similarity + recency decay + importance).
    Returns raw scored events so callers can build their own presentation layer.

    Unlike ``/context``, this endpoint does not inject semantic facts from Neo4j
    or procedural rules — it returns event rows with their score breakdown.
    """
    assert_self_or_admin(current_user, body.user_id)
    resolved_app_ids = body.app_ids or current_user.get("app_ids")
    embedding_failed = False
    query_embedding: list[float] | None = None

    try:
        query_embedding = await llm.embed(body.query)
    except Exception as exc:
        logger.warning("Embedding failed for search query: %s", exc)
        embedding_failed = True

    if query_embedding is None:
        return SearchResponse(
            user_id=body.user_id,
            query=body.query,
            results=[],
            total=0,
            embedding_failed=True,
        )

    results = await episodic.hybrid_search(
        pg,
        body.user_id,
        query_embedding,
        app_ids=resolved_app_ids,
        top_k=body.limit,
        from_date=body.from_date,
        to_date=body.to_date,
    )

    items = [
        SearchResultItem(
            event_id=str(r.event.id),
            raw_text=r.event.raw_text,
            importance_score=r.event.importance_score,
            hybrid_score=round(r.hybrid_score, 4),
            similarity_score=round(r.similarity_score, 4),
            recency_score=round(r.recency_score, 4),
            consolidated=r.event.consolidated,
            created_at=r.event.created_at.isoformat() if r.event.created_at else "",
        )
        for r in results
    ]

    audit = get_audit_logger()
    if audit:
        from smritikosh.audit.logger import AuditEvent, EventType
        await audit.emit(AuditEvent(
            event_type=EventType.SEARCH_PERFORMED,
            user_id=body.user_id,
            app_id=(resolved_app_ids[0] if resolved_app_ids else "default"),
            payload={
                "query_preview": body.query[:200],
                "results_count": len(items),
                "embedding_failed": embedding_failed,
                "limit": body.limit,
            },
        ))

    return SearchResponse(
        user_id=body.user_id,
        query=body.query,
        results=items,
        total=len(items),
        embedding_failed=embedding_failed,
    )


@router.get("/event/{event_id}", response_model=MemoryEventDetail)
async def get_event(
    event_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> MemoryEventDetail:
    """Return a single memory event by ID."""
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid event_id UUID format.")

    result = await pg.execute(select(Event).where(Event.id == eid))  # noqa: E501
    event = result.scalar_one_or_none()
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    assert_self_or_admin(current_user, event.user_id)

    return MemoryEventDetail(
        event_id=str(event.id),
        user_id=event.user_id,
        app_id=event.app_id,
        raw_text=event.raw_text,
        summary=event.summary,
        importance_score=event.importance_score,
        recall_count=event.recall_count or 0,
        reconsolidation_count=event.reconsolidation_count or 0,
        consolidated=event.consolidated,
        cluster_id=event.cluster_id,
        cluster_label=event.cluster_label,
        created_at=event.created_at.isoformat() if event.created_at else "",
        updated_at=event.updated_at.isoformat() if event.updated_at else "",
        last_reconsolidated_at=(
            event.last_reconsolidated_at.isoformat()
            if event.last_reconsolidated_at else None
        ),
    )


@router.get("/event/{event_id}/links", response_model=MemoryLinksResponse)
async def get_event_links(
    event_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> MemoryLinksResponse:
    """
    Return all narrative links touching this event (both directions).

    Each link includes a short preview of the connected event's text
    so callers can render the graph without a second round-trip.
    """
    try:
        eid = uuid.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid event_id UUID format.")

    # Always verify ownership before returning any data
    ev_result = await pg.execute(select(Event).where(Event.id == eid))
    ev = ev_result.scalar_one_or_none()
    if ev is None:
        raise HTTPException(status_code=404, detail="Event not found.")
    assert_self_or_admin(current_user, ev.user_id)

    # Load all links where this event is either source or target
    links_result = await pg.execute(
        select(MemoryLink).where(
            or_(MemoryLink.from_event_id == eid, MemoryLink.to_event_id == eid)
        ).limit(100)
    )
    links = links_result.scalars().all()

    if not links:
        return MemoryLinksResponse(event_id=event_id, links=[])

    # Collect the IDs of the *other* side of each link
    related_ids = {
        link.to_event_id if link.from_event_id == eid else link.from_event_id
        for link in links
    }
    events_result = await pg.execute(
        select(Event).where(Event.id.in_(related_ids))
    )
    event_map: dict[uuid.UUID, Event] = {
        e.id: e for e in events_result.scalars().all()
    }

    def preview(e: Event | None) -> str:
        if e is None:
            return ""
        return e.raw_text[:100] + ("…" if len(e.raw_text) > 100 else "")

    items = []
    for link in links:
        from_event = event_map.get(link.from_event_id) if link.from_event_id != eid else None
        to_event   = event_map.get(link.to_event_id)   if link.to_event_id   != eid else None

        # Reconstruct full preview pair: anchor event has an empty preview slot
        if link.from_event_id == eid:
            from_preview = ""          # this is the anchor itself
            to_preview   = preview(event_map.get(link.to_event_id))
        else:
            from_preview = preview(event_map.get(link.from_event_id))
            to_preview   = ""          # this is the anchor itself

        items.append(
            MemoryLinkItem(
                link_id=str(link.id),
                from_event_id=str(link.from_event_id),
                from_event_preview=from_preview,
                to_event_id=str(link.to_event_id),
                to_event_preview=to_preview,
                relation_type=link.relation_type,
                created_at=link.created_at.isoformat() if link.created_at else "",
            )
        )

    return MemoryLinksResponse(event_id=event_id, links=items)


@router.get("/export", tags=["memory"])
async def export_memory(
    user_id: Annotated[str, Query(description="User whose memories to export")],
    app_ids: Annotated[list[str] | None, Query(description="App namespaces to include. Defaults to all.")] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> StreamingResponse:
    """
    Export all memory events for a user as NDJSON (newline-delimited JSON).

    Each line is a JSON object with: event_id, raw_text, summary,
    importance_score, consolidated, recall_count, cluster_label, created_at.

    The response streams incrementally — safe for large memory sets.
    Content-Type: application/x-ndjson
    """
    assert_self_or_admin(current_user, user_id)
    resolved_app_ids = app_ids or current_user.get("app_ids")

    async def _ndjson_stream() -> AsyncIterator[str]:
        stmt = select(Event).where(Event.user_id == user_id)
        if resolved_app_ids:
            stmt = stmt.where(Event.app_id.in_(resolved_app_ids))
        stmt = stmt.order_by(Event.created_at.asc())

        result = await pg.execute(stmt)
        for event in result.scalars().all():
            item = ExportEventItem(
                event_id=str(event.id),
                raw_text=event.raw_text or "",
                summary=event.summary,
                importance_score=event.importance_score,
                consolidated=bool(event.consolidated),
                recall_count=event.recall_count or 0,
                cluster_label=event.cluster_label,
                created_at=event.created_at.isoformat() if event.created_at else "",
            )
            yield item.model_dump_json() + "\n"

    return StreamingResponse(
        _ndjson_stream(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="memories_{user_id}.ndjson"'},
    )


@router.get("/{user_id}", response_model=RecentEventsResponse, tags=["memory"])
async def get_recent_events(
    user_id: str,
    app_ids: Annotated[list[str] | None, Query(description="App namespaces to filter by")] = None,
    limit: Annotated[int, Query(ge=1, le=500, description="Number of events to return")] = 10,
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> RecentEventsResponse:
    """Return the most recent memory events for a user, newest first."""
    assert_self_or_admin(current_user, user_id)
    resolved_app_ids = app_ids or current_user.get("app_ids")
    events = await episodic.get_recent(pg, user_id, resolved_app_ids, limit=limit)

    return RecentEventsResponse(
        user_id=user_id,
        app_ids=resolved_app_ids or [],
        events=[
            RecentEventItem(
                event_id=str(e.id),
                raw_text=e.raw_text,
                importance_score=e.importance_score,
                consolidated=e.consolidated,
                created_at=e.created_at.isoformat() if e.created_at else "",
                cluster_id=e.cluster_id,
                cluster_label=e.cluster_label,
            )
            for e in events
        ],
    )
