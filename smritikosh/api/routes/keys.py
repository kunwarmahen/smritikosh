"""
API key routes — create, list, and revoke personal API keys.

POST   /keys          Generate a new API key (returns full key once).
GET    /keys          List active keys for the authenticated user.
DELETE /keys/{key_id} Revoke a key.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.auth.deps import assert_app_access, get_current_user
from smritikosh.auth.utils import generate_api_key
from smritikosh.db.models import ApiKey
from smritikosh.db.postgres import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/keys", tags=["api-keys"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    name: str
    app_ids: list[str] = Field(default_factory=lambda: ["default"])


class CreateKeyResponse(BaseModel):
    id: str
    name: str
    key: str          # full key — shown ONCE
    key_prefix: str
    app_ids: list[str]
    created_at: str


class KeyItem(BaseModel):
    id: str
    name: str
    key_prefix: str
    app_ids: list[str]
    last_used_at: str | None
    created_at: str


class KeyListResponse(BaseModel):
    keys: list[KeyItem]


class RevokeKeyResponse(BaseModel):
    revoked: bool
    id: str


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("", response_model=CreateKeyResponse, status_code=201)
async def create_key(
    request: CreateKeyRequest,
    current_user: Annotated[dict, Depends(get_current_user)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> CreateKeyResponse:
    """
    Generate a new API key scoped to the authenticated user.

    The full key is returned exactly once — store it immediately.
    Subsequent calls to GET /keys only show the short prefix.
    """
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Key name cannot be empty.")

    # Validate all requested app_ids are within the caller's allowed app_ids
    caller_app_ids = current_user.get("app_ids", [])
    if current_user.get("role") != "admin":
        for aid in request.app_ids:
            if aid not in caller_app_ids:
                raise HTTPException(
                    status_code=403,
                    detail=f"Access to app '{aid}' denied.",
                )

    full_key, key_hash, key_prefix = generate_api_key()

    api_key = ApiKey(
        user_id=current_user["sub"],
        app_ids=request.app_ids,
        name=name,
        key_prefix=key_prefix,
        key_hash=key_hash,
    )
    pg.add(api_key)
    await pg.flush()   # populate api_key.id and created_at

    logger.info("API key created", extra={"user_id": current_user["sub"], "key_id": str(api_key.id)})

    return CreateKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        key=full_key,
        key_prefix=key_prefix,
        app_ids=api_key.app_ids,
        created_at=api_key.created_at.isoformat(),
    )


@router.get("", response_model=KeyListResponse)
async def list_keys(
    current_user: Annotated[dict, Depends(get_current_user)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> KeyListResponse:
    """List all active (non-revoked) API keys for the authenticated user."""
    result = await pg.execute(
        select(ApiKey)
        .where(ApiKey.user_id == current_user["sub"])
        .where(ApiKey.revoked_at.is_(None))
        .order_by(ApiKey.created_at.desc())
    )
    keys = result.scalars().all()

    return KeyListResponse(
        keys=[
            KeyItem(
                id=str(k.id),
                name=k.name,
                key_prefix=k.key_prefix,
                app_ids=k.app_ids,
                last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
                created_at=k.created_at.isoformat(),
            )
            for k in keys
        ]
    )


@router.delete("/{key_id}", response_model=RevokeKeyResponse)
async def revoke_key(
    key_id: str,
    current_user: Annotated[dict, Depends(get_current_user)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> RevokeKeyResponse:
    """Revoke an API key. Only the key's owner (or an admin) can revoke it."""
    try:
        kid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid key_id UUID format.")

    result = await pg.execute(
        select(ApiKey)
        .where(ApiKey.id == kid)
        .where(ApiKey.revoked_at.is_(None))
    )
    api_key = result.scalar_one_or_none()

    if api_key is None:
        raise HTTPException(status_code=404, detail="Key not found or already revoked.")

    # Only owner or admin may revoke
    if current_user["role"] != "admin" and api_key.user_id != current_user["sub"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

    api_key.revoked_at = datetime.now(timezone.utc)
    logger.info("API key revoked", extra={"user_id": api_key.user_id, "key_id": key_id})

    return RevokeKeyResponse(revoked=True, id=key_id)
