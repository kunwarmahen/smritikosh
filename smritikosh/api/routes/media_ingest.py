"""
Media ingestion routes — POST /ingest/media (voice notes, documents).

Handles file uploads with background processing:
1. POST /ingest/media — upload file → returns immediately with media_id
2. GET /ingest/media/{media_id}/status — poll for results + pending facts
3. POST /ingest/media/{media_id}/confirm — user confirms selected facts
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_semantic
from smritikosh.api.schemas import (
    MediaFactConfirmRequest,
    MediaIngestResponse,
    MediaStatusResponse,
)
from smritikosh.api.quotas import enforce_event_quota, enforce_token_quota
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.db.models import MediaIngest, MediaIngestStatus
from neo4j import AsyncSession as NeoSession

from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.tasks import enqueue
from smritikosh.tasks.jobs import _process_media_record

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_VOICE_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
_DOC_EXT = {".txt", ".md", ".csv", ".pdf"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_IMAGE_CONTENT_TYPES = {"receipt", "screenshot", "whiteboard"}
_MEETING_CONTENT_TYPES = {"meeting_recording"}
_MAX_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_DOC_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_MEETING_BYTES = 500 * 1024 * 1024


@router.post("/media", response_model=MediaIngestResponse, status_code=202)
async def ingest_media(
    user_id: str = Form(...),
    app_id: str = Form("default"),
    content_type: str = Form(...),
    context_note: str = Form(""),
    idempotency_key: Optional[str] = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> MediaIngestResponse:
    """
    Upload a media file (voice note or document) for memory extraction.

    Returns immediately with media_id and status='processing'.
    Client polls GET /ingest/media/{media_id}/status for results.
    """
    # Auth
    assert_self_or_admin(current_user, user_id)
    await enforce_event_quota(pg, user_id, app_id)
    await enforce_token_quota(pg, user_id, app_id)

    # Validate content_type
    _valid_types = {"voice_note", "document"} | _IMAGE_CONTENT_TYPES | _MEETING_CONTENT_TYPES
    if content_type not in _valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content_type: {content_type}. Must be one of: {', '.join(sorted(_valid_types))}",
        )

    # Validate file extension and size
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    ext = f".{ext}" if ext else ""

    if content_type in ("voice_note", "meeting_recording"):
        if ext not in _VOICE_EXT:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid audio format: {ext}. Supported: {', '.join(sorted(_VOICE_EXT))}",
            )
    elif content_type in _IMAGE_CONTENT_TYPES:
        if ext not in _IMAGE_EXT:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid image format: {ext}. Supported: {', '.join(sorted(_IMAGE_EXT))}",
            )
    elif content_type == "document":
        if ext not in _DOC_EXT:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid document format: {ext}. Supported: {', '.join(sorted(_DOC_EXT))}",
            )

    # Check idempotency
    if idempotency_key:
        existing = await pg.execute(
            __import__("sqlalchemy").select(MediaIngest).where(
                MediaIngest.user_id == user_id,
                MediaIngest.app_id == app_id,
                MediaIngest.idempotency_key == idempotency_key,
            )
        )
        existing_record = existing.scalar_one_or_none()
        if existing_record:
            return MediaIngestResponse(
                media_id=str(existing_record.id),
                user_id=user_id,
                app_id=app_id,
                content_type=content_type,
                status=existing_record.status,
                facts_extracted=existing_record.facts_extracted,
                facts_pending_review=existing_record.facts_pending_review,
                message="(cached — idempotency key)",
            )

    # Read file bytes
    media_id = uuid.uuid4()
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / 1024 / 1024

    # Validate file size
    if content_type == "voice_note" and len(file_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large: {file_size_mb:.1f} MB (max 25 MB)",
        )
    elif content_type == "meeting_recording" and len(file_bytes) > _MAX_MEETING_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Meeting recording too large: {file_size_mb:.1f} MB (max 500 MB)",
        )
    elif content_type in _IMAGE_CONTENT_TYPES and len(file_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large: {file_size_mb:.1f} MB (max 20 MB)",
        )
    elif content_type == "document" and len(file_bytes) > _MAX_DOC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Document too large: {file_size_mb:.1f} MB (max 10 MB)",
        )

    # Persist the upload (status=processing) with its raw bytes, so the
    # processing task is durable — it survives an API/worker restart.
    ingest = MediaIngest(
        id=media_id,
        user_id=user_id,
        app_id=app_id,
        content_type=content_type,
        idempotency_key=idempotency_key,
        status=MediaIngestStatus.PROCESSING,
        raw_file=file_bytes,
        filename=file.filename or "unknown",
        context_note=context_note,
    )
    pg.add(ingest)
    # Commit BEFORE enqueueing — the worker must be able to read the record.
    await pg.commit()

    # Enqueue onto the durable task queue; fall back to an in-process
    # background task when no queue (REDIS_URL) is configured.
    job = await enqueue("process_media", str(media_id))
    if job is None:
        background_tasks.add_task(_process_media_record, str(media_id))

    return MediaIngestResponse(
        media_id=str(media_id),
        user_id=user_id,
        app_id=app_id,
        content_type=content_type,
        status="processing",
        message="File accepted. Processing started.",
    )


@router.get("/media/{media_id}/status", response_model=MediaStatusResponse)
async def get_media_status(
    media_id: str,
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> MediaStatusResponse:
    """
    Poll the status of a media ingestion job.
    Returns status, facts extracted, and pending facts awaiting user confirmation.
    """
    # Fetch MediaIngest record
    ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
    if not ingest:
        raise HTTPException(status_code=404, detail="Media ingest not found")

    # Auth: user can only see their own media
    if ingest.user_id != current_user.get("user_id") and current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")

    return MediaStatusResponse(
        media_id=media_id,
        status=ingest.status,
        facts_extracted=ingest.facts_extracted,
        facts_pending_review=ingest.facts_pending_review,
        pending_facts=ingest.pending_facts,
        message=ingest.error_message or "",
    )


@router.post("/media/{media_id}/confirm", response_model=MediaIngestResponse)
async def confirm_media_facts(
    media_id: str,
    body: MediaFactConfirmRequest,
    pg: AsyncSession = Depends(get_session),
    neo: NeoSession = Depends(get_neo4j_session),
    semantic: SemanticMemory = Depends(get_semantic),
    current_user: dict = Depends(get_current_user),
) -> MediaIngestResponse:
    """
    User confirms selected facts from pending_facts.
    Saves confirmed facts to SemanticMemory.
    """
    # Auth
    assert_self_or_admin(current_user, body.user_id)

    # Fetch MediaIngest record
    ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
    if not ingest:
        raise HTTPException(status_code=404, detail="Media ingest not found")

    if ingest.status != MediaIngestStatus.COMPLETE:
        raise HTTPException(
            status_code=422,
            detail=f"Cannot confirm facts: ingest status is {ingest.status}",
        )

    if not ingest.pending_facts:
        raise HTTPException(
            status_code=422, detail="No pending facts to confirm"
        )

    # Extract confirmed facts
    confirmed_facts = []
    for idx in body.confirmed_indices:
        if 0 <= idx < len(ingest.pending_facts):
            confirmed_facts.append(ingest.pending_facts[idx])

    # Save confirmed facts to SemanticMemory (Neo4j)
    source_type = ingest.source_type or "media_document"
    for fact in confirmed_facts:
        try:
            await semantic.upsert_fact(
                neo,
                user_id=ingest.user_id,
                app_id=ingest.app_id,
                category=fact.get("category", "general"),
                key=fact.get("key", "unknown"),
                value=fact.get("value", fact.get("content", "")),
                source_type=source_type,
                source_meta={"confirmed_from_media": str(ingest.id)},
                status="active",
            )
        except Exception:
            logger.warning("Failed to save confirmed fact: %s", fact)

    ingest.pending_facts = []
    ingest.facts_pending_review = 0
    await pg.flush()
    await pg.commit()

    return MediaIngestResponse(
        media_id=media_id,
        user_id=ingest.user_id,
        app_id=ingest.app_id,
        content_type=ingest.content_type,
        status=ingest.status,
        facts_extracted=ingest.facts_extracted,
        facts_pending_review=ingest.facts_pending_review,
        message=f"Confirmed {len(confirmed_facts)} facts",
    )
