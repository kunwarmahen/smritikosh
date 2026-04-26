"""
OAuth2 connector routes — Google Gmail and Google Calendar.

Endpoints:
    GET /connectors/google/authorize     Start OAuth2 flow (redirect to Google)
    GET /connectors/google/callback      OAuth2 callback (store encrypted tokens)
    GET /connectors/{user_id}            List connected connectors for a user
    DELETE /connectors/{user_id}/{provider}  Disconnect a connector
    POST /connectors/gmail/sync          Fetch Gmail messages
    POST /connectors/gcal/sync           Fetch Google Calendar events
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from neo4j import AsyncSession as NeoSession
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_hippocampus
from smritikosh.api.routes.ingest import _run_ingestion
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.config import settings
from smritikosh.connectors.gcal import GcalConnector
from smritikosh.connectors.gmail import GmailConnector
from smritikosh.connectors.oauth import (
    build_authorization_url,
    build_state_jwt,
    decrypt_tokens,
    encrypt_tokens,
    exchange_code,
    refresh_access_token,
    verify_state_jwt,
)
from smritikosh.db.models import ConnectorProvider, ConnectorStatus, UserConnector
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/connectors", tags=["connectors"])

# ── Response schemas ──────────────────────────────────────────────────────────


class GoogleAuthorizeResponse(BaseModel):
    authorize_url: str = Field(..., description="Google OAuth authorization URL")


class ConnectorInfo(BaseModel):
    provider: str
    status: str
    scopes: list[str]
    connected_at: datetime
    updated_at: datetime


class ConnectorListResponse(BaseModel):
    connectors: list[ConnectorInfo]


class GmailSyncRequest(BaseModel):
    user_id: str = Field(..., description="User to ingest emails for")
    app_id: str = Field("default", description="Application namespace")
    limit: int = Field(20, ge=1, le=100, description="Max emails to fetch")
    query: str = Field("is:unread", description="Gmail search query")


class GcalSyncRequest(BaseModel):
    user_id: str = Field(..., description="User to ingest events for")
    app_id: str = Field("default", description="Application namespace")
    days_back: int = Field(7, ge=0, le=365, description="How many days back to fetch")
    max_results: int = Field(50, ge=1, le=2500, description="Max events to fetch")


# ── GET /connectors/google/authorize ──────────────────────────────────────────


@router.get("/google/authorize", response_model=GoogleAuthorizeResponse)
async def google_authorize(
    user_id: Annotated[str, Query(description="User ID to connect")],
    app_id: Annotated[str, Query(description="App ID")] = "default",
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> GoogleAuthorizeResponse:
    """
    Start Google OAuth2 authorization flow.

    Returns a URL the user should visit. After authorizing, Google redirects back
    to /connectors/google/callback with a code parameter.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=501,
            detail="Google OAuth not configured (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET required)",
        )

    assert_self_or_admin(current_user, user_id)

    # Build signed state JWT containing user_id + app_id
    state = build_state_jwt(user_id, app_id)
    authorize_url = build_authorization_url(state)

    return GoogleAuthorizeResponse(authorize_url=authorize_url)


# ── GET /connectors/google/callback ───────────────────────────────────────────


@router.get("/google/callback")
async def google_callback(
    code: Annotated[str, Query(description="OAuth authorization code from Google")],
    state: Annotated[str, Query(description="State token with user_id + app_id")],
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
) -> dict[str, Any]:
    """
    OAuth2 callback — exchange authorization code for tokens.

    Stores encrypted tokens in the user_connectors table.
    Returns success message.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    # Verify state and extract user_id + app_id
    try:
        user_id, app_id = verify_state_jwt(state)
    except Exception as e:
        logger.warning(f"Invalid OAuth state: {e}")
        raise HTTPException(status_code=400, detail="Invalid state token")

    # Exchange code for tokens
    try:
        tokens = await exchange_code(code)
    except Exception as e:
        logger.error(f"Token exchange failed: {e}")
        raise HTTPException(status_code=400, detail="Token exchange failed")

    # Store encrypted tokens in database
    try:
        encrypted_tokens = encrypt_tokens(tokens)
        token_expires_at = None
        if "expires_in" in tokens:
            token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=tokens["expires_in"])

        # Upsert user_connector record
        stmt = select(UserConnector).where(
            (UserConnector.user_id == user_id)
            & (UserConnector.app_id == app_id)
            & (UserConnector.provider == ConnectorProvider.GMAIL)  # Store as GMAIL for both
        )
        existing = (await pg.execute(stmt)).scalars().first()

        if existing:
            existing.encrypted_tokens = encrypted_tokens
            existing.token_expires_at = token_expires_at
            existing.status = ConnectorStatus.ACTIVE
            existing.scopes = tokens.get("scope", "").split(" ")
            existing.updated_at = datetime.now(timezone.utc)
        else:
            connector = UserConnector(
                user_id=user_id,
                app_id=app_id,
                provider=ConnectorProvider.GMAIL,  # Store as GMAIL (covers both Gmail + Calendar)
                status=ConnectorStatus.ACTIVE,
                encrypted_tokens=encrypted_tokens,
                token_expires_at=token_expires_at,
                scopes=tokens.get("scope", "").split(" "),
            )
            pg.add(connector)

        await pg.commit()
        logger.info(f"Stored Google OAuth tokens for user={user_id} app={app_id}")

        return {
            "ok": True,
            "message": "Google account connected successfully. You can now sync Gmail and Calendar.",
        }

    except Exception as e:
        logger.exception(f"Failed to store connector tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to store credentials")


# ── GET /connectors/{user_id} ─────────────────────────────────────────────────


@router.get("/{user_id}", response_model=ConnectorListResponse)
async def list_connectors(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> ConnectorListResponse:
    """List all connected connectors for a user."""
    assert_self_or_admin(current_user, user_id)

    stmt = select(UserConnector).where(
        (UserConnector.user_id == user_id) & (UserConnector.app_id == app_id)
    )
    result = await pg.execute(stmt)
    connectors = result.scalars().all()

    return ConnectorListResponse(
        connectors=[
            ConnectorInfo(
                provider=c.provider,
                status=c.status,
                scopes=c.scopes,
                connected_at=c.connected_at,
                updated_at=c.updated_at,
            )
            for c in connectors
        ]
    )


# ── DELETE /connectors/{user_id}/{provider} ──────────────────────────────────


@router.delete("/{user_id}/{provider}")
async def disconnect_connector(
    user_id: str,
    provider: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict[str, Any]:
    """Disconnect and delete a connector (revoke access)."""
    assert_self_or_admin(current_user, user_id)

    stmt = delete(UserConnector).where(
        (UserConnector.user_id == user_id)
        & (UserConnector.app_id == app_id)
        & (UserConnector.provider == provider)
    )
    result = await pg.execute(stmt)
    await pg.commit()

    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Connector not found")

    logger.info(f"Disconnected {provider} for user={user_id}")
    return {"ok": True, "message": f"{provider} connector disconnected"}


# ── POST /connectors/gmail/sync ───────────────────────────────────────────────


@router.post("/gmail/sync", status_code=201)
async def gmail_sync(
    request: GmailSyncRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    neo: Annotated[NeoSession, Depends(get_neo4j_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict[str, Any]:
    """Fetch Gmail messages and ingest them."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    assert_self_or_admin(current_user, request.user_id)

    # Load connector credentials
    stmt = select(UserConnector).where(
        (UserConnector.user_id == request.user_id)
        & (UserConnector.app_id == request.app_id)
        & (UserConnector.provider == ConnectorProvider.GMAIL)
    )
    connector = (await pg.execute(stmt)).scalars().first()

    if not connector:
        raise HTTPException(status_code=404, detail="Gmail connector not configured")

    if connector.status == ConnectorStatus.REVOKED:
        raise HTTPException(status_code=403, detail="Connector has been revoked")

    # Decrypt tokens and check expiry
    try:
        tokens = decrypt_tokens(connector.encrypted_tokens)
    except Exception as e:
        logger.error(f"Failed to decrypt tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to decrypt credentials")

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    # Refresh if needed
    if connector.token_expires_at and datetime.now(timezone.utc) + timedelta(minutes=5) > connector.token_expires_at:
        if not refresh_token:
            raise HTTPException(status_code=403, detail="Token expired, reconnect required")

        try:
            new_tokens = await refresh_access_token(refresh_token)
            # Update tokens with new values
            tokens.update(new_tokens)
            access_token = new_tokens["access_token"]

            # Update database
            connector.encrypted_tokens = encrypt_tokens(tokens)
            if "expires_in" in new_tokens:
                connector.token_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=new_tokens["expires_in"]
                )
            connector.updated_at = datetime.now(timezone.utc)
            await pg.commit()
            logger.info(f"Refreshed Gmail token for user={request.user_id}")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            connector.status = ConnectorStatus.ERROR
            await pg.commit()
            raise HTTPException(status_code=403, detail="Token refresh failed; please reconnect")

    # Fetch emails
    gmail = GmailConnector()
    events = await gmail.extract_events(access_token, limit=request.limit, query=request.query)

    if not events:
        return {"source": "gmail", "events_ingested": 0, "events_failed": 0, "event_ids": []}

    # Ingest through Hippocampus
    result = await _run_ingestion(events, request.user_id, request.app_id, hippocampus, pg, neo, source="gmail")
    return result.model_dump()


# ── POST /connectors/gcal/sync ────────────────────────────────────────────────


@router.post("/gcal/sync", status_code=201)
async def gcal_sync(
    request: GcalSyncRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    neo: Annotated[NeoSession, Depends(get_neo4j_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> dict[str, Any]:
    """Fetch Google Calendar events and ingest them."""
    if not settings.google_client_id:
        raise HTTPException(status_code=501, detail="Google OAuth not configured")

    assert_self_or_admin(current_user, request.user_id)

    # Load connector credentials
    stmt = select(UserConnector).where(
        (UserConnector.user_id == request.user_id)
        & (UserConnector.app_id == request.app_id)
        & (UserConnector.provider == ConnectorProvider.GMAIL)
    )
    connector = (await pg.execute(stmt)).scalars().first()

    if not connector:
        raise HTTPException(status_code=404, detail="Google Calendar connector not configured")

    if connector.status == ConnectorStatus.REVOKED:
        raise HTTPException(status_code=403, detail="Connector has been revoked")

    # Decrypt tokens and check expiry
    try:
        tokens = decrypt_tokens(connector.encrypted_tokens)
    except Exception as e:
        logger.error(f"Failed to decrypt tokens: {e}")
        raise HTTPException(status_code=500, detail="Failed to decrypt credentials")

    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    # Refresh if needed
    if connector.token_expires_at and datetime.now(timezone.utc) + timedelta(minutes=5) > connector.token_expires_at:
        if not refresh_token:
            raise HTTPException(status_code=403, detail="Token expired, reconnect required")

        try:
            new_tokens = await refresh_access_token(refresh_token)
            tokens.update(new_tokens)
            access_token = new_tokens["access_token"]

            connector.encrypted_tokens = encrypt_tokens(tokens)
            if "expires_in" in new_tokens:
                connector.token_expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=new_tokens["expires_in"]
                )
            connector.updated_at = datetime.now(timezone.utc)
            await pg.commit()
            logger.info(f"Refreshed Calendar token for user={request.user_id}")
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            connector.status = ConnectorStatus.ERROR
            await pg.commit()
            raise HTTPException(status_code=403, detail="Token refresh failed; please reconnect")

    # Fetch calendar events
    now = datetime.now(timezone.utc)
    time_min = now - timedelta(days=request.days_back)
    time_max = now + timedelta(days=1)

    gcal = GcalConnector()
    events = await gcal.extract_events(
        access_token, time_min=time_min, time_max=time_max, max_results=request.max_results
    )

    if not events:
        return {"source": "gcal", "events_ingested": 0, "events_failed": 0, "event_ids": []}

    # Ingest through Hippocampus
    result = await _run_ingestion(events, request.user_id, request.app_id, hippocampus, pg, neo, source="gcal")
    return result.model_dump()
