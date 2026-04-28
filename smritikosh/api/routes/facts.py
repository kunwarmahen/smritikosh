"""
Fact management routes — QC layer for semantic facts.

GET  /facts/{user_id}                         List facts (filterable by status)
PATCH /facts/{user_id}/{category}/{key}/status Approve or reject a pending fact
GET  /facts/contradictions/{user_id}           List unresolved contradictions
PATCH /facts/contradictions/{contradiction_id} Resolve a contradiction
"""

import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import AsyncSession as NeoSession
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_semantic
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.db.models import FactContradiction, FactStatus
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/facts", tags=["facts"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class FactItem(BaseModel):
    category: str
    key: str
    value: str
    confidence: float
    frequency_count: int
    status: str
    source_type: str
    first_seen_at: str
    last_seen_at: str


class FactListResponse(BaseModel):
    user_id: str
    app_id: str
    facts: list[FactItem]
    total: int


class FactStatusPatch(BaseModel):
    status: str = Field(..., description="'active' or 'rejected'")


class ContradictionItem(BaseModel):
    id: str
    category: str
    key: str
    existing_value: str
    existing_confidence: float
    candidate_value: str
    candidate_source: str
    candidate_confidence: float
    created_at: str


class ContradictionListResponse(BaseModel):
    user_id: str
    app_id: str
    contradictions: list[ContradictionItem]
    total: int


class ContradictionResolution(BaseModel):
    keep: str = Field(..., description="'existing', 'candidate', or 'merge'")
    merged_value: str | None = Field(None, description="Required when keep='merge': the canonical merged fact value")


class ContradictionResolved(BaseModel):
    id: str
    resolution: str
    resolved_at: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/{user_id}", response_model=FactListResponse)
async def list_facts(
    user_id: str,
    semantic: Annotated[SemanticMemory, Depends(get_semantic)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
    app_id: str = Query("default"),
    status: Optional[str] = Query(None, description="Filter by status: active, pending, rejected"),
    category: Optional[str] = Query(None),
) -> FactListResponse:
    """
    List semantic facts for a user.

    Pass status=pending to get the review queue.
    Omit status to get all facts regardless of lifecycle state.
    """
    assert_self_or_admin(current_user, user_id)

    # active_only=True only when status filter is explicitly 'active' or unset-but-context-is-default
    # For review purposes (status=pending, status=rejected, or status=None) we disable the gate.
    active_only = status == FactStatus.ACTIVE if status else False

    facts = await semantic.get_facts(
        neo,
        user_id,
        app_id,
        category=category,
        active_only=active_only,
    )

    # Client-side status filter if explicit status was requested
    if status and status != FactStatus.ACTIVE:
        facts = [f for f in facts if f.status == status]

    items = [
        FactItem(
            category=f.category,
            key=f.key,
            value=f.value,
            confidence=f.confidence,
            frequency_count=f.frequency_count,
            status=f.status,
            source_type=f.source_type,
            first_seen_at=f.first_seen_at,
            last_seen_at=f.last_seen_at,
        )
        for f in facts
    ]
    return FactListResponse(user_id=user_id, app_id=app_id, facts=items, total=len(items))


@router.patch("/{user_id}/{category}/{key}/status", response_model=FactItem)
async def update_fact_status(
    user_id: str,
    category: str,
    key: str,
    body: FactStatusPatch,
    semantic: Annotated[SemanticMemory, Depends(get_semantic)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
    app_id: str = Query("default"),
) -> FactItem:
    """
    Approve (→ active) or reject (→ rejected) a pending fact.

    Used by the review dashboard to action items from the QC queue.
    """
    assert_self_or_admin(current_user, user_id)

    if body.status not in (FactStatus.ACTIVE, FactStatus.REJECTED):
        raise HTTPException(status_code=422, detail="status must be 'active' or 'rejected'")

    updated = await semantic.set_fact_status(
        neo,
        user_id=user_id,
        app_id=app_id,
        category=category,
        key=key,
        status=body.status,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Fact not found")

    return FactItem(
        category=updated.category,
        key=updated.key,
        value=updated.value,
        confidence=updated.confidence,
        frequency_count=updated.frequency_count,
        status=updated.status,
        source_type=updated.source_type,
        first_seen_at=updated.first_seen_at,
        last_seen_at=updated.last_seen_at,
    )


@router.get("/contradictions/{user_id}", response_model=ContradictionListResponse)
async def list_contradictions(
    user_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
    app_id: str = Query("default"),
) -> ContradictionListResponse:
    """List unresolved fact contradictions for a user."""
    assert_self_or_admin(current_user, user_id)

    result = await pg.execute(
        select(FactContradiction)
        .where(
            FactContradiction.user_id == user_id,
            FactContradiction.app_id == app_id,
            FactContradiction.resolved == False,  # noqa: E712
        )
        .order_by(FactContradiction.created_at.desc())
    )
    rows = result.scalars().all()

    items = [
        ContradictionItem(
            id=str(row.id),
            category=row.category,
            key=row.key,
            existing_value=row.existing_value,
            existing_confidence=row.existing_confidence,
            candidate_value=row.candidate_value,
            candidate_source=row.candidate_source,
            candidate_confidence=row.candidate_confidence,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]
    return ContradictionListResponse(
        user_id=user_id, app_id=app_id, contradictions=items, total=len(items)
    )


@router.patch("/contradictions/{contradiction_id}", response_model=ContradictionResolved)
async def resolve_contradiction(
    contradiction_id: str,
    body: ContradictionResolution,
    semantic: Annotated[SemanticMemory, Depends(get_semantic)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> ContradictionResolved:
    """
    Resolve a contradiction by keeping the existing value, taking the candidate,
    or writing a custom merged value.

    keep=existing  → dismiss the candidate; keep the existing fact unchanged.
    keep=candidate → overwrite the existing fact with the candidate value.
    keep=merge     → write merged_value as the canonical fact (must be provided).
    """
    if body.keep not in ("existing", "candidate", "merge"):
        raise HTTPException(status_code=422, detail="keep must be 'existing', 'candidate', or 'merge'")
    if body.keep == "merge" and not (body.merged_value or "").strip():
        raise HTTPException(status_code=422, detail="merged_value is required when keep='merge'")

    result = await pg.execute(
        select(FactContradiction).where(FactContradiction.id == contradiction_id)
    )
    contradiction = result.scalar_one_or_none()
    if contradiction is None:
        raise HTTPException(status_code=404, detail="Contradiction not found")

    assert_self_or_admin(current_user, contradiction.user_id)

    if body.keep in ("candidate", "merge"):
        from smritikosh.db.models import SOURCE_CONFIDENCE_DEFAULTS
        new_value = body.merged_value if body.keep == "merge" else contradiction.candidate_value
        new_confidence = (
            max(contradiction.candidate_confidence, contradiction.existing_confidence)
            if body.keep == "merge"
            else contradiction.candidate_confidence
        )
        await semantic.upsert_fact(
            neo,
            user_id=contradiction.user_id,
            app_id=contradiction.app_id,
            category=contradiction.category,
            key=contradiction.key,
            value=new_value,
            confidence=new_confidence,
            source_type=contradiction.candidate_source,
            source_meta={"resolved_contradiction": str(contradiction.id), "resolution": body.keep},
            status=FactStatus.ACTIVE,
        )

    now = datetime.now(timezone.utc)
    resolution_label = f"keep_{body.keep}" if body.keep != "merge" else "merged"
    await pg.execute(
        update(FactContradiction)
        .where(FactContradiction.id == contradiction_id)
        .values(
            resolved=True,
            resolution=resolution_label,
            resolved_at=now,
        )
    )
    await pg.flush()

    return ContradictionResolved(
        id=contradiction_id,
        resolution=resolution_label,
        resolved_at=now.isoformat(),
    )
