"""Durable task definitions + ARQ worker settings (item A3).

The actual work lives in plain async helpers (`_process_media_record`,
`_re_embed_stale_events`) so both the ARQ task wrappers *and* the in-process
fallback (used when REDIS_URL is unset) call exactly the same code.

Run the worker:
    arq smritikosh.tasks.jobs.WorkerSettings
"""

import logging
import uuid
from datetime import datetime, timezone

from smritikosh.tasks.queue import queue_enabled, redis_settings

logger = logging.getLogger(__name__)


# ── Media processing ────────────────────────────────────────────────────────────


async def _process_media_record(media_id: str) -> str:
    """Load a MediaIngest's stored bytes, process them, update the record.

    Shared by the ARQ task and the in-process fallback. Idempotent enough to be
    retried: it re-reads state from the DB and writes a terminal status.
    Returns the final status string.
    """
    from smritikosh.api.deps import get_media_processor
    from smritikosh.db.models import MediaIngest, MediaIngestStatus
    from smritikosh.db.neo4j import neo4j_session
    from smritikosh.db.postgres import db_session

    async with db_session() as pg:
        ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
        if ingest is None:
            logger.warning("process_media: media_id=%s not found", media_id)
            return "not_found"
        raw = ingest.raw_file
        user_id = ingest.user_id
        app_id = ingest.app_id
        content_type = ingest.content_type
        filename = ingest.filename or "unknown"
        context_note = ingest.context_note or ""

    if raw is None:
        # Already processed (bytes cleared) or never stored — nothing to do.
        logger.info("process_media: media_id=%s has no stored bytes — skipping", media_id)
        return "skipped"

    media_processor = get_media_processor()
    async with db_session() as pg, neo4j_session() as neo:
        try:
            result = await media_processor.process(
                pg,
                neo,
                media_id=media_id,
                user_id=user_id,
                app_id=app_id,
                content_type=content_type,
                file_bytes=raw,
                filename=filename,
                context_note=context_note,
            )
        except Exception as exc:
            logger.exception("Media processing failed: media_id=%s", media_id)
            ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
            if ingest:
                ingest.status = MediaIngestStatus.FAILED
                ingest.error_message = str(exc)[:500]
                ingest.raw_file = None      # free the bytes even on failure
                ingest.processed_at = datetime.now(timezone.utc)
                await pg.flush()
            await pg.commit()
            return "failed"

        ingest = await pg.get(MediaIngest, uuid.UUID(media_id))
        if ingest:
            ingest.status = result.status
            ingest.facts_extracted = result.facts_extracted
            ingest.facts_pending_review = result.facts_pending_review
            ingest.pending_facts = result.pending_facts
            ingest.event_id = uuid.UUID(result.event_id) if result.event_id else None
            ingest.source_type = content_type
            ingest.error_message = result.error_message
            ingest.processed_at = datetime.now(timezone.utc)
            ingest.raw_file = None          # free the stored bytes once done
            await pg.flush()
        await pg.commit()

    logger.info(
        "Media processing complete: media_id=%s status=%s facts=%d",
        media_id, result.status, result.facts_extracted,
    )
    return result.status


# ── Bulk re-embed ───────────────────────────────────────────────────────────────


async def _re_embed_stale_events() -> dict:
    """Re-embed every event whose embedding is missing or has the wrong dimension.

    Recomputes the stale set itself (rather than receiving a snapshot), so the
    task carries no large payload and is safe to retry.
    """
    from sqlalchemy import text, update

    from smritikosh.api.deps import get_llm
    from smritikosh.config import settings
    from smritikosh.db.models import Event
    from smritikosh.db.postgres import db_session

    configured_dim = settings.embedding_dimensions
    llm = get_llm()

    async with db_session() as pg:
        rows = await pg.execute(
            text(
                "SELECT id, raw_text FROM events "
                "WHERE embedding IS NULL OR vector_dims(embedding) != :dim "
                "ORDER BY created_at ASC"
            ),
            {"dim": configured_dim},
        )
        stale = [(str(r.id), r.raw_text) for r in rows.fetchall()]

    success = 0
    errors = 0
    async with db_session() as pg:
        for event_id, raw_text in stale:
            try:
                embedding = await llm.embed(raw_text)
                await pg.execute(
                    update(Event)
                    .where(Event.id == uuid.UUID(event_id))
                    .values(embedding=embedding, updated_at=datetime.now(timezone.utc))
                )
                success += 1
            except Exception as exc:
                logger.warning("Re-embed failed for event %s: %s", event_id, exc)
                errors += 1

    logger.info("Re-embed complete: success=%d errors=%d total=%d", success, errors, len(stale))

    # Audit the completion (no-op when MongoDB is not configured).
    try:
        from smritikosh.api.deps import get_audit_logger
        from smritikosh.audit.logger import AuditEvent, EventType

        audit = get_audit_logger()
        if audit:
            await audit.emit_sync(AuditEvent(
                event_type=EventType.EMBEDDING_REEMBED_COMPLETE,
                user_id="__system__",
                app_id="__system__",
                payload={"success": success, "errors": errors, "total": len(stale)},
            ))
    except Exception as exc:  # pragma: no cover - audit must never break the task
        logger.debug("Re-embed audit emit failed: %s", exc)

    return {"success": success, "errors": errors, "total": len(stale)}


# ── Reconsolidation after recall ────────────────────────────────────────────────


async def _reconsolidate_recalled(
    event_ids: list[str], query: str, user_id: str, app_id: str
) -> dict:
    """Reconsolidate recalled events by ID (A3-followup).

    Shared by the ARQ task and the in-process fallback. Each /context call
    that surfaces memories enqueues one of these; running it here keeps the
    per-recall LLM call off the API process, where it can otherwise saturate
    a slow provider and stall subsequent /context requests.
    """
    from smritikosh.api.deps import get_reconsolidation_engine

    engine = get_reconsolidation_engine()
    batch = await engine.reconsolidate_after_recall_by_ids(
        event_ids, query, user_id, app_id
    )
    return {
        "evaluated": batch.events_evaluated,
        "updated": batch.events_updated,
        "skipped": batch.events_skipped,
    }


# ── ARQ task wrappers ───────────────────────────────────────────────────────────


async def process_media(ctx, media_id: str) -> str:
    """ARQ task: process one uploaded media file."""
    return await _process_media_record(media_id)


async def re_embed_events(ctx) -> dict:
    """ARQ task: re-embed all stale event embeddings."""
    return await _re_embed_stale_events()


async def reconsolidate_recalled(
    ctx, event_ids: list[str], query: str, user_id: str, app_id: str = "default"
) -> dict:
    """ARQ task: reconsolidate events recalled by a /context call."""
    return await _reconsolidate_recalled(event_ids, query, user_id, app_id)


# ── ARQ worker settings ─────────────────────────────────────────────────────────


async def _on_startup(ctx) -> None:
    logger.info("Smritikosh taskworker started — durable queue ready.")


async def _on_shutdown(ctx) -> None:
    from smritikosh.db.neo4j import close_neo4j
    from smritikosh.db.postgres import close_db

    await close_db()
    await close_neo4j()
    logger.info("Smritikosh taskworker stopped.")


class WorkerSettings:
    """ARQ worker entrypoint — run with: arq smritikosh.tasks.jobs.WorkerSettings"""

    functions = [process_media, re_embed_events, reconsolidate_recalled]
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    max_tries = 3                 # retry transient failures
    job_timeout = 3600            # media transcription of a long recording can be slow


# Bind the Redis connection only when configured. Importing this module from the
# API in a Redis-less dev setup must not fail — the taskworker is the only thing
# that needs REDIS_URL, and it is never run without it.
if queue_enabled():
    WorkerSettings.redis_settings = redis_settings()
