"""
Memory routes — capture and retrieve episodic events.

POST /memory/event   Encode a raw interaction into memory (Hippocampus.encode)
GET  /memory/{user_id}  Return recent events for a user
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_hippocampus, get_episodic
from smritikosh.api.schemas import (
    DeleteEventResponse,
    DeleteUserMemoryResponse,
    EventRequest,
    EventResponse,
    RecentEventItem,
    RecentEventsResponse,
)
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/memory", tags=["memory"])


@router.post("/event", response_model=EventResponse, status_code=201)
async def capture_event(
    request: EventRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
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
    try:
        result = await hippocampus.encode(
            pg,
            neo,
            user_id=request.user_id,
            raw_text=request.content,
            app_id=request.app_id,
            metadata=request.metadata,
        )
    except Exception as exc:
        logger.exception("Hippocampus encode failed", extra={"user_id": request.user_id})
        raise HTTPException(status_code=500, detail=f"Memory encoding failed: {exc}") from exc

    return EventResponse(
        event_id=str(result.event.id),
        user_id=request.user_id,
        importance_score=result.importance_score,
        facts_extracted=len(result.facts),
        extraction_failed=result.extraction_failed,
    )


@router.delete("/event/{event_id}", response_model=DeleteEventResponse)
async def delete_event(
    event_id: str,
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)],
    pg: Annotated[AsyncSession, Depends(get_session)],
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

    deleted = await episodic.delete(pg, eid)
    return DeleteEventResponse(deleted=deleted, event_id=event_id)


@router.delete("/user/{user_id}", response_model=DeleteUserMemoryResponse)
async def delete_user_memory(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
) -> DeleteUserMemoryResponse:
    """
    Delete all memory events for a user within an app namespace.

    Use with care — this removes all episodic events for the user.
    Semantic facts in Neo4j are not affected by this endpoint.
    """
    count = await episodic.delete_all_for_user(pg, user_id, app_id)
    logger.info(
        "Deleted all user memory",
        extra={"user_id": user_id, "app_id": app_id, "events_deleted": count},
    )
    return DeleteUserMemoryResponse(events_deleted=count, user_id=user_id, app_id=app_id)


@router.get("/{user_id}", response_model=RecentEventsResponse, tags=["memory"])
async def get_recent_events(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    limit: Annotated[int, Query(ge=1, le=50, description="Number of events to return")] = 10,
    episodic: Annotated[EpisodicMemory, Depends(get_episodic)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
) -> RecentEventsResponse:
    """Return the most recent memory events for a user, newest first."""
    events = await episodic.get_recent(pg, user_id, app_id=app_id, limit=limit)

    return RecentEventsResponse(
        user_id=user_id,
        app_id=app_id,
        events=[
            RecentEventItem(
                event_id=str(e.id),
                raw_text=e.raw_text,
                importance_score=e.importance_score,
                consolidated=e.consolidated,
                created_at=e.created_at.isoformat() if e.created_at else "",
            )
            for e in events
        ],
    )
