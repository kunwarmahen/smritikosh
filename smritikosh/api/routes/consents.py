"""
Consent routes — cross-app memory sharing grants (item S4).

POST   /consents              Grant (or reactivate) app B's read access to
                              facts learned in app A, per fact category.
DELETE /consents              Revoke a grant (row kept for audit history).
GET    /consents/{user_id}    List a user's grants (?include_revoked=true).

Authorization: the caller must be the user themself or an admin, and — for
grant/revoke — must have access to the *source* app (the one whose facts are
being shared out). Enforcement of the grants happens at read time in the
context builder via ConsentService.consented_facts().
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_consent_service
from smritikosh.auth.deps import assert_app_access, assert_self_or_admin, get_current_user
from smritikosh.db.postgres import get_session
from smritikosh.memory.consent import ConsentError, ConsentService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/consents", tags=["consents"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class ConsentGrantRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    source_app_id: str = Field(..., min_length=1, max_length=255)
    target_app_id: str = Field(..., min_length=1, max_length=255)
    categories: list[str] = Field(
        default_factory=list,
        description="FactCategory values the grant covers; empty = all categories.",
    )


class ConsentRevokeRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=255)
    source_app_id: str = Field(..., min_length=1, max_length=255)
    target_app_id: str = Field(..., min_length=1, max_length=255)


class ConsentItem(BaseModel):
    consent_id: str
    user_id: str
    source_app_id: str
    target_app_id: str
    categories: list[str]
    active: bool
    granted_at: str
    revoked_at: Optional[str] = None
    created_by: str


class ConsentListResponse(BaseModel):
    user_id: str
    consents: list[ConsentItem]


def _to_item(consent) -> ConsentItem:
    return ConsentItem(
        consent_id=str(consent.id),
        user_id=consent.user_id,
        source_app_id=consent.source_app_id,
        target_app_id=consent.target_app_id,
        categories=list(consent.categories or []),
        active=consent.is_active,
        granted_at=consent.granted_at.isoformat(),
        revoked_at=consent.revoked_at.isoformat() if consent.revoked_at else None,
        created_by=consent.created_by,
    )


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("", response_model=ConsentItem, status_code=201)
async def grant_consent(
    body: ConsentGrantRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[ConsentService, Depends(get_consent_service)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> ConsentItem:
    """Grant (or reactivate) cross-app read access to the user's facts."""
    assert_self_or_admin(current_user, body.user_id)
    assert_app_access(current_user, body.source_app_id)
    try:
        consent = await service.grant(
            pg,
            user_id=body.user_id,
            source_app_id=body.source_app_id,
            target_app_id=body.target_app_id,
            categories=body.categories,
            created_by=current_user["sub"],
        )
    except ConsentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await pg.commit()
    logger.info(
        "Consent granted",
        extra={
            "user_id": body.user_id,
            "source_app": body.source_app_id,
            "target_app": body.target_app_id,
            "by": current_user["sub"],
        },
    )
    return _to_item(consent)


@router.delete("", response_model=dict)
async def revoke_consent(
    body: ConsentRevokeRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[ConsentService, Depends(get_consent_service)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """Revoke a grant. Takes effect on the next cross-app read."""
    assert_self_or_admin(current_user, body.user_id)
    assert_app_access(current_user, body.source_app_id)
    revoked = await service.revoke(
        pg,
        user_id=body.user_id,
        source_app_id=body.source_app_id,
        target_app_id=body.target_app_id,
        revoked_by=current_user["sub"],
    )
    if not revoked:
        raise HTTPException(status_code=404, detail="No active consent found.")
    await pg.commit()
    return {
        "revoked": True,
        "user_id": body.user_id,
        "source_app_id": body.source_app_id,
        "target_app_id": body.target_app_id,
    }


@router.get("/{user_id}", response_model=ConsentListResponse)
async def list_consents(
    user_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    service: Annotated[ConsentService, Depends(get_consent_service)],
    current_user: Annotated[dict, Depends(get_current_user)],
    include_revoked: Annotated[bool, Query()] = False,
) -> ConsentListResponse:
    """List a user's cross-app grants (active only, unless include_revoked)."""
    assert_self_or_admin(current_user, user_id)
    consents = await service.list_for_user(pg, user_id, include_revoked=include_revoked)
    return ConsentListResponse(
        user_id=user_id, consents=[_to_item(c) for c in consents]
    )
