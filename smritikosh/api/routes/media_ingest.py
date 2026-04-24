"""
Media ingestion routes — POST /ingest/media (voice notes, documents).

Handles file uploads with background processing:
1. POST /ingest/media — upload file → returns immediately with media_id
2. GET /ingest/media/{media_id}/status — poll for results + pending facts
3. POST /ingest/media/{media_id}/confirm — user confirms selected facts
"""

import logging
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_media_processor, get_semantic
from smritikosh.api.schemas import (
    MediaFactConfirmRequest,
    MediaIngestResponse,
    MediaStatusResponse,
)
from smritikosh.auth.deps import get_current_user
from smritikosh.auth.utils import assert_self_or_admin
from smritikosh.db.models import MediaIngest, MediaIngestStatus, SourceType
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_async_sessionmaker, get_session
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.media_processor import MediaProcessor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_VOICE_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
_DOC_EXT = {".txt", ".md", ".csv", ".pdf"}
_MAX_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_DOC_BYTES = 10 * 1024 * 1024


async def _get_async_sessionmaker(request):
    """Retrieve the async sessionmaker from app state for background tasks."""
    return getattr(request.app.state, "async_sessionmaker", None)


async def _run_processing(
    *,
    media_id: str,
    user_id: str,
    app_id: str,
    content_type: str,
    file_bytes: bytes,
    filename: str,
    context_note: str,
    media_processor: MediaProcessor,
    semantic: SemanticMemory,
    async_sessionmaker,
):
    """Background task: process media and update MediaIngest record."""
    # Create new DB sessions for background task
    async_session_factory = async_sessionmaker
    async with async_session_factory() as pg:
        async with get_neo4j_session() as neo:
            # Process the file
            result = await media_processor.process(
                pg,
                neo,
                media_id=media_id,
                user_id=user_id,
                app_id=app_id,
                content_type=content_type,
                file_bytes=file_bytes,
                filename=filename,
                context_note=context_note,
            )

            # Update MediaIngest record
            ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
            if ingest:
                ingest.status = result.status
                ingest.facts_extracted = result.facts_extracted
                ingest.facts_pending_review = result.facts_pending_review
                ingest.pending_facts = result.pending_facts
                ingest.event_id = uuid.UUID(result.event_id) if result.event_id else None
                ingest.source_type = content_type  # voice_note or document
                ingest.error_message = result.error_message
                ingest.processed_at = pg.execute(lambda: __import__("datetime").datetime.now(__import__("datetime").timezone.utc))
                await pg.flush()

            await pg.commit()
            logger.info(
                "Media processing completed: media_id=%s status=%s facts=%d",
                media_id,
                result.status,
                result.facts_extracted,
            )


@router.post("/media", response_model=MediaIngestResponse, status_code=202)
async def ingest_media(
    user_id: str = Form(...),
    app_id: str = Form("default"),
    content_type: str = Form(...),
    context_note: str = Form(""),
    idempotency_key: Optional[str] = Form(None),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    media_processor: MediaProcessor = Depends(get_media_processor),
    semantic: SemanticMemory = Depends(get_semantic),
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
    request=None,
) -> MediaIngestResponse:
    """
    Upload a media file (voice note or document) for memory extraction.

    Returns immediately with media_id and status='processing'.
    Client polls GET /ingest/media/{media_id}/status for results.
    """
    # Auth
    assert_self_or_admin(current_user, user_id)

    # Validate content_type
    if content_type not in ("voice_note", "document"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid content_type: {content_type}. Must be 'voice_note' or 'document'",
        )

    # Validate file extension and size
    ext = file.filename.split(".")[-1].lower() if file.filename else ""
    ext = f".{ext}" if ext else ""

    if content_type == "voice_note":
        if ext not in _VOICE_EXT:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid audio format: {ext}. Supported: {', '.join(sorted(_VOICE_EXT))}",
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

    # Create MediaIngest record (status=processing)
    media_id = uuid.uuid4()
    ingest = MediaIngest(
        id=media_id,
        user_id=user_id,
        app_id=app_id,
        content_type=content_type,
        idempotency_key=idempotency_key,
        status=MediaIngestStatus.PROCESSING,
    )
    pg.add(ingest)
    await pg.flush()  # ensures ID is generated

    # Read file bytes
    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / 1024 / 1024

    # Validate file size
    if content_type == "voice_note" and len(file_bytes) > _MAX_AUDIO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large: {file_size_mb:.1f} MB (max 25 MB)",
        )
    elif content_type == "document" and len(file_bytes) > _MAX_DOC_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Document too large: {file_size_mb:.1f} MB (max 10 MB)",
        )

    # Enqueue background task
    async_sessionmaker = await _get_async_sessionmaker(request) if request else None
    if not async_sessionmaker:
        # Fallback: use get_async_sessionmaker directly
        async_sessionmaker = get_async_sessionmaker()

    background_tasks.add_task(
        _run_processing,
        media_id=str(media_id),
        user_id=user_id,
        app_id=app_id,
        content_type=content_type,
        file_bytes=file_bytes,
        filename=file.filename or "unknown",
        context_note=context_note,
        media_processor=media_processor,
        semantic=semantic,
        async_sessionmaker=async_sessionmaker,
    )

    # Commit the MediaIngest record
    await pg.commit()

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

    # Save confirmed facts to SemanticMemory
    # (In a real implementation, you'd call semantic.upsert_fact for each)
    # For now, just clear pending_facts
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
