"""
Ingest routes — external source connectors.

Each route normalises a different source format into ConnectorEvents, then
runs the full Hippocampus pipeline (importance scoring, embedding, fact
extraction) for every event.

Endpoints:
    POST /ingest/push           Generic JSON push (webhook / server-side events)
    POST /ingest/file           File upload (text, markdown, CSV, JSON)
    POST /ingest/slack/events   Slack Events API (signature-verified)
    POST /ingest/email/sync     IMAP fetch — caller supplies credentials per-request
    POST /ingest/calendar       iCalendar (.ics) file upload

All endpoints return an ``IngestResponse`` summarising how many events were
ingested, how many failed, and the resulting event IDs.

Configuration:
    SLACK_SIGNING_SECRET environment variable is required for Slack signature
    verification.  If absent the endpoint returns 501.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from neo4j import AsyncSession as NeoSession
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_hippocampus
from smritikosh.connectors.base import ConnectorEvent
from smritikosh.connectors.calendar import CalendarConnector
from smritikosh.connectors.email import EmailConnector, IMAPConfig
from smritikosh.connectors.file import FileConnector
from smritikosh.connectors.slack import SlackConnector
from smritikosh.connectors.webhook import WebhookConnector
from smritikosh.config import settings
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.hippocampus import Hippocampus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

# ── response schema ────────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    source: str
    events_ingested: int
    events_failed: int
    event_ids: list[str]


# ── shared ingestion helper ────────────────────────────────────────────────────

async def _run_ingestion(
    events: list[ConnectorEvent],
    user_id: str,
    app_id: str,
    hippocampus: Hippocampus,
    pg: AsyncSession,
    neo: NeoSession,
    source: str,
) -> IngestResponse:
    """Encode every ConnectorEvent through the Hippocampus pipeline."""
    event_ids: list[str] = []
    failed = 0

    for ev in events:
        try:
            result = await hippocampus.encode(
                pg,
                neo,
                user_id=user_id,
                raw_text=ev.content,
                app_id=app_id,
                metadata=ev.to_metadata(),
            )
            event_ids.append(str(result.event.id))
        except Exception:
            logger.exception(
                "Hippocampus encode failed for ingest event source=%s source_id=%s",
                ev.source,
                ev.source_id,
            )
            failed += 1

    return IngestResponse(
        source=source,
        events_ingested=len(event_ids),
        events_failed=failed,
        event_ids=event_ids,
    )


# ── POST /ingest/push ─────────────────────────────────────────────────────────


class PushIngestRequest(BaseModel):
    user_id: str = Field(..., description="User to encode the event for")
    content: str = Field(..., description="Text content to encode into memory")
    app_id: str = Field("default", description="Application namespace")
    source: str = Field("webhook", description="Source label stored in event metadata")
    source_id: str = Field("", description="Original event ID from the source system")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extra source context")


@router.post("/push", response_model=IngestResponse, status_code=201)
async def push_ingest(
    request: PushIngestRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> IngestResponse:
    """
    Encode a single event pushed from an external system.

    Use this endpoint for server-side integrations that can push JSON
    directly (e.g. custom webhooks, n8n, Zapier, Make.com workflows).
    """
    assert_self_or_admin(current_user, request.user_id)
    connector = WebhookConnector()
    payload: dict[str, Any] = {
        "content":   request.content,
        "id":        request.source_id,
        **request.metadata,
    }
    events = await connector.extract_events(
        payload,
        source_label=request.source,
    )
    if not events:
        raise HTTPException(status_code=422, detail="No content could be extracted from the payload.")

    return await _run_ingestion(
        events, request.user_id, request.app_id, hippocampus, pg, neo, source=request.source
    )


# ── POST /ingest/file ─────────────────────────────────────────────────────────


@router.post("/file", response_model=IngestResponse, status_code=201)
async def file_ingest(
    user_id: Annotated[str, Form(description="User to encode events for")],
    file: Annotated[UploadFile, File(description="File to ingest (.txt .md .csv .json)")],
    app_id: Annotated[str, Form()] = "default",
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    neo: Annotated[NeoSession, Depends(get_neo4j_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> IngestResponse:
    """
    Upload a file and encode its contents as memory events.

    Supported formats:
    - ``.txt`` / ``.md``  Paragraph-split plain text
    - ``.csv``            One memory event per row
    - ``.json``           Array of strings or objects
    """
    assert_self_or_admin(current_user, user_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    connector = FileConnector()
    events = await connector.extract_events(content, filename=file.filename or "upload.txt")

    if not events:
        raise HTTPException(status_code=422, detail="No extractable content found in the file.")

    return await _run_ingestion(
        events, user_id, app_id, hippocampus, pg, neo, source="file"
    )


# ── POST /ingest/slack/events ─────────────────────────────────────────────────


@router.post("/slack/events")
async def slack_events(
    request: Request,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    x_slack_request_timestamp: Annotated[str | None, Header()] = None,
    x_slack_signature: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """
    Receive Slack Events API callbacks.

    Handles:
    - ``url_verification`` — returns challenge to complete Slack handshake.
    - ``event_callback``   — encodes message/app_mention events into memory.

    Requires ``SLACK_SIGNING_SECRET`` to be set in environment / settings.
    Each Slack workspace connects to one ``user_id`` via a query parameter:
        POST /ingest/slack/events?user_id=mahen&app_id=myapp
    """
    signing_secret = getattr(settings, "slack_signing_secret", None)
    if not signing_secret:
        raise HTTPException(status_code=501, detail="SLACK_SIGNING_SECRET not configured.")

    raw_body = await request.body()

    # Signature verification
    if not x_slack_request_timestamp or not x_slack_signature:
        raise HTTPException(status_code=400, detail="Missing Slack signature headers.")

    if not SlackConnector.verify_signature(
        signing_secret, raw_body, x_slack_request_timestamp, x_slack_signature
    ):
        raise HTTPException(status_code=403, detail="Slack signature verification failed.")

    import json
    payload: dict[str, Any] = json.loads(raw_body)

    # URL verification handshake
    if payload.get("type") == "url_verification":
        return {"challenge": payload.get("challenge", "")}

    # Resolve user_id from query params (required for event_callback)
    user_id: str = request.query_params.get("user_id", "")
    app_id: str  = request.query_params.get("app_id", "default")
    if not user_id:
        raise HTTPException(
            status_code=422,
            detail="user_id query parameter is required for event_callback ingestion.",
        )

    connector = SlackConnector()
    events = await connector.extract_events(payload)

    if not events:
        # Not an error — could be a bot message or unsupported event type
        return {"ok": True, "events_ingested": 0}

    result = await _run_ingestion(events, user_id, app_id, hippocampus, pg, neo, source="slack")
    return {"ok": True, **result.model_dump()}


# ── POST /ingest/email/sync ───────────────────────────────────────────────────


class EmailSyncRequest(BaseModel):
    user_id: str   = Field(..., description="User to encode emails for")
    app_id: str    = Field("default", description="Application namespace")
    host: str      = Field(..., description="IMAP server hostname, e.g. imap.gmail.com")
    port: int      = Field(993, description="IMAP SSL port (default 993)")
    username: str  = Field(..., description="IMAP login username / email address")
    password: str  = Field(..., description="IMAP password or app-specific password")
    mailbox: str   = Field("INBOX", description="Mailbox to sync (default INBOX)")
    limit: int     = Field(20, ge=1, le=100, description="Max emails to fetch")
    unseen_only: bool = Field(True, description="Only fetch unread emails")


@router.post("/email/sync", response_model=IngestResponse, status_code=201)
async def email_sync(
    request: EmailSyncRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> IngestResponse:
    """
    Fetch emails from an IMAP mailbox and encode them as memory events.

    Credentials are used once per request and never stored.
    Connects over SSL (port 993 by default).

    Common configurations:
    - Gmail:   host=imap.gmail.com, use an app-specific password
    - Outlook: host=outlook.office365.com
    - iCloud:  host=imap.mail.me.com
    """
    assert_self_or_admin(current_user, request.user_id)
    config = IMAPConfig(
        host=request.host,
        username=request.username,
        password=request.password,
        port=request.port,
        mailbox=request.mailbox,
        limit=request.limit,
        unseen_only=request.unseen_only,
    )

    connector = EmailConnector()
    events = await connector.extract_events(config)

    if not events:
        return IngestResponse(
            source="email", events_ingested=0, events_failed=0, event_ids=[]
        )

    return await _run_ingestion(events, request.user_id, request.app_id, hippocampus, pg, neo, source="email")


# ── POST /ingest/calendar ─────────────────────────────────────────────────────


@router.post("/calendar", response_model=IngestResponse, status_code=201)
async def calendar_ingest(
    user_id: Annotated[str, Form(description="User to encode events for")],
    file: Annotated[UploadFile, File(description="iCalendar .ics file")],
    app_id: Annotated[str, Form()] = "default",
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
    neo: Annotated[NeoSession, Depends(get_neo4j_session)] = None,
    current_user: Annotated[dict, Depends(get_current_user)] = None,
) -> IngestResponse:
    """
    Upload an iCalendar (.ics) file and encode each VEVENT as a memory event.

    Useful for ingesting meeting history, schedule context, or past events
    that the AI should be aware of when building context for the user.
    """
    assert_self_or_admin(current_user, user_id)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    connector = CalendarConnector()
    events = await connector.extract_events(content, filename=file.filename or "calendar.ics")

    if not events:
        raise HTTPException(
            status_code=422, detail="No VEVENT components with parseable content found."
        )

    return await _run_ingestion(events, user_id, app_id, hippocampus, pg, neo, source="calendar")
