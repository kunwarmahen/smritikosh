"""
Audit trail routes — query the provenance history of memories.

All routes require MongoDB to be configured (MONGODB_URL in .env).
Returns 503 if the audit trail is not available.

Endpoints:
    GET /audit/{user_id}                  Full timeline for a user
    GET /audit/event/{event_id}/lineage   All records linked to one episodic event
    GET /audit/event/{event_id}           Same as lineage (alias)
    GET /audit/stats/{user_id}            Count per event_type for a user
"""

import logging
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_audit_logger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/audit", tags=["audit"])


def _require_audit(audit=Depends(get_audit_logger)):
    if audit is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Audit trail is not configured. "
                "Set MONGODB_URL in your environment to enable it."
            ),
        )
    return audit


@router.get("/{user_id}", summary="User audit timeline")
async def get_user_timeline(
    user_id: str,
    audit=Depends(_require_audit),
    app_id: Annotated[str, Query()] = "default",
    event_type: Annotated[Optional[str], Query(description="Filter to one event type")] = None,
    event_id: Annotated[Optional[str], Query(description="Filter to one episodic event")] = None,
    from_ts: Annotated[Optional[datetime], Query(description="Start of time range (ISO 8601)")] = None,
    to_ts: Annotated[Optional[datetime], Query(description="End of time range (ISO 8601)")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Return the audit timeline for a user — all pipeline events that touched
    their data, newest first. Filterable by event_type, episodic event_id,
    and date range.
    """
    assert_self_or_admin(current_user, user_id)
    records = await audit.get_timeline(
        user_id=user_id,
        app_id=app_id,
        event_type=event_type,
        event_id=event_id,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=limit,
        offset=offset,
    )
    return {
        "user_id": user_id,
        "app_id": app_id,
        "total": len(records),
        "offset": offset,
        "records": records,
    }


@router.get("/event/{event_id}/lineage", summary="Memory lineage")
async def get_event_lineage(
    event_id: str,
    audit=Depends(_require_audit),
) -> dict:
    """
    Return the complete provenance chain for one episodic memory event —
    from raw encoding through fact extraction, consolidation, reconsolidation,
    and any feedback signals.  Records are returned oldest first so you can
    read the history of a memory chronologically.
    """
    records = await audit.get_event_lineage(event_id)
    return {
        "event_id": event_id,
        "total": len(records),
        "lineage": records,
    }


@router.get("/stats/{user_id}", summary="Audit stats per event type")
async def get_audit_stats(
    user_id: str,
    app_id: Annotated[str, Query()] = "default",
    audit=Depends(_require_audit),
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Return total audit record count broken down by event_type for a user.
    Useful for dashboards and monitoring.
    """
    assert_self_or_admin(current_user, user_id)
    counts = await audit.get_stats(user_id=user_id, app_id=app_id)
    return {
        "user_id": user_id,
        "app_id": app_id,
        "counts": counts,
        "total": sum(counts.values()),
    }
