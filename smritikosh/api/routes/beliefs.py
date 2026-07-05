"""
Belief routes — inspect and retract inferred beliefs (item E2).

GET    /beliefs/{user_id}                        List beliefs (active by default).
GET    /beliefs/{user_id}/{belief_id}/evidence   Belief + the events it was inferred from.
DELETE /beliefs/{user_id}/{belief_id}            Retract a belief (status=rejected).

Retraction keeps the row: the belief miner reads rejected statements and will
never re-derive them (prompt exclusion + a WHERE guard on its upsert), so a
wrong belief stays gone instead of resurfacing with growing confidence.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_audit_logger
from smritikosh.api.schemas import (
    BeliefEvidenceEvent,
    BeliefEvidenceResponse,
    BeliefListResponse,
    BeliefRecord,
    BeliefRetractResponse,
)
from smritikosh.auth.deps import assert_app_access, assert_self_or_admin, get_current_user
from smritikosh.db.models import BeliefStatus, Event, UserBelief
from smritikosh.db.postgres import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/beliefs", tags=["beliefs"])


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _to_record(b: UserBelief) -> BeliefRecord:
    return BeliefRecord(
        belief_id=str(b.id),
        user_id=b.user_id,
        app_id=b.app_id,
        statement=b.statement,
        category=b.category,
        confidence=b.confidence,
        evidence_count=b.evidence_count,
        evidence_event_ids=[str(e) for e in (b.evidence_event_ids or [])],
        status=b.status,
        retracted_at=_iso(b.retracted_at),
        first_inferred_at=_iso(b.first_inferred_at) or "",
        last_updated_at=_iso(b.last_updated_at) or "",
    )


async def _get_belief_or_404(
    pg: AsyncSession, user_id: str, app_id: str, belief_id: str
) -> UserBelief:
    try:
        bid = uuid.UUID(belief_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="belief_id must be a UUID.")
    result = await pg.execute(
        select(UserBelief).where(
            UserBelief.id == bid,
            UserBelief.user_id == user_id,
            UserBelief.app_id == app_id,
        )
    )
    belief = result.scalar_one_or_none()
    if belief is None:
        raise HTTPException(status_code=404, detail="Belief not found.")
    return belief


@router.get("/{user_id}", response_model=BeliefListResponse)
async def list_beliefs(
    user_id: str,
    app_id: Annotated[str, Query()] = "default",
    include_rejected: Annotated[bool, Query()] = False,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> BeliefListResponse:
    """List a user's inferred beliefs with their evidence event IDs."""
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)

    q = (
        select(UserBelief)
        .where(
            UserBelief.user_id == user_id,
            UserBelief.app_id == app_id,
            UserBelief.confidence >= min_confidence,
        )
        .order_by(UserBelief.confidence.desc())
    )
    if not include_rejected:
        q = q.where(UserBelief.status != BeliefStatus.REJECTED)

    result = await pg.execute(q)
    beliefs = result.scalars().all()
    return BeliefListResponse(
        user_id=user_id,
        app_id=app_id,
        beliefs=[_to_record(b) for b in beliefs],
    )


@router.get("/{user_id}/{belief_id}/evidence", response_model=BeliefEvidenceResponse)
async def get_belief_evidence(
    user_id: str,
    belief_id: str,
    app_id: Annotated[str, Query()] = "default",
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> BeliefEvidenceResponse:
    """
    Return the belief and the episodic events it was inferred from
    ("based on these events" — the evidence trail behind the inference).
    """
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)
    belief = await _get_belief_or_404(pg, user_id, app_id, belief_id)

    evidence_ids: list[uuid.UUID] = []
    for raw in belief.evidence_event_ids or []:
        try:
            evidence_ids.append(uuid.UUID(str(raw)))
        except ValueError:
            continue

    events: list[Event] = []
    if evidence_ids:
        result = await pg.execute(
            select(Event)
            .where(Event.id.in_(evidence_ids), Event.user_id == user_id)
            .order_by(Event.created_at.desc())
        )
        events = list(result.scalars().all())

    found_ids = {e.id for e in events}
    missing = [str(i) for i in evidence_ids if i not in found_ids]

    return BeliefEvidenceResponse(
        belief=_to_record(belief),
        evidence_events=[
            BeliefEvidenceEvent(
                event_id=str(e.id),
                text=e.summary or e.raw_text,
                importance_score=e.importance_score,
                created_at=_iso(e.created_at),
            )
            for e in events
        ],
        missing_event_ids=missing,
    )


@router.delete("/{user_id}/{belief_id}", response_model=BeliefRetractResponse)
async def retract_belief(
    user_id: str,
    belief_id: str,
    app_id: Annotated[str, Query()] = "default",
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
    audit=Depends(get_audit_logger),
) -> BeliefRetractResponse:
    """
    Retract a belief: mark it rejected (the row is kept so re-mining can
    never resurrect the statement). Idempotent — retracting an already
    rejected belief returns its existing state.
    """
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)
    belief = await _get_belief_or_404(pg, user_id, app_id, belief_id)

    if belief.status != BeliefStatus.REJECTED:
        belief.status = BeliefStatus.REJECTED
        belief.retracted_at = datetime.now(timezone.utc)
        await pg.flush()

        logger.info(
            "Belief retracted",
            extra={"user_id": user_id, "belief_id": belief_id, "by": current_user.get("sub")},
        )
        if audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await audit.emit(AuditEvent(
                event_type=EventType.BELIEF_RETRACTED,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "belief_id": belief_id,
                    "statement": belief.statement,
                    "category": belief.category,
                    "confidence": belief.confidence,
                    "retracted_by": current_user.get("sub"),
                },
            ))

    return BeliefRetractResponse(
        belief_id=str(belief.id),
        status=belief.status,
        retracted_at=_iso(belief.retracted_at) or "",
    )
