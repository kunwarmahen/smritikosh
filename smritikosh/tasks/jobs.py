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


# ── Bulk re-embed (item H1: chunked + resumable) ───────────────────────────────
#
# Changing EMBEDDING_MODEL used to trigger one monolithic re-embed pass — no
# progress, no resumability; a deploy mid-run silently restarted millions of
# embed calls. Now each run is an `embedding_migrations` row: chunks of
# RE_EMBED_BATCH_SIZE events are processed one queue task at a time, progress
# and a keyset cursor are committed after every chunk, and the task re-enqueues
# itself until done. Progress: GET /admin/re-embed/status.

_STALE_PREDICATE = "(embedding IS NULL OR vector_dims(embedding) != :dim)"


async def _create_or_resume_migration() -> tuple[str, int, bool]:
    """Return (migration_id, stale_total, resumed).

    An existing RUNNING migration is resumed (its cursor is kept) rather than
    starting a parallel run; otherwise a fresh row is created with the current
    stale count and embedding model as the target.
    """
    from sqlalchemy import select, text

    from smritikosh.config import settings
    from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus
    from smritikosh.db.postgres import db_session

    async with db_session() as pg:
        result = await pg.execute(
            select(EmbeddingMigration)
            .where(EmbeddingMigration.status == EmbeddingMigrationStatus.RUNNING)
            .order_by(EmbeddingMigration.started_at.desc())
        )
        existing = result.scalars().first()
        if existing is not None:
            return str(existing.id), existing.total, True

        row = await pg.execute(
            text(f"SELECT count(*) FROM events WHERE {_STALE_PREDICATE}"),
            {"dim": settings.embedding_dimensions},
        )
        total = int(row.scalar() or 0)

        migration = EmbeddingMigration(
            target_model=settings.embedding_model,
            target_dim=settings.embedding_dimensions,
            total=total,
        )
        pg.add(migration)
        await pg.flush()
        migration_id = str(migration.id)
        await pg.commit()
        return migration_id, total, False


async def _run_embedding_migration_chunk(migration_id: str) -> str:
    """Process one chunk of a migration. Returns 'continue' | 'complete' | 'stopped'.

    Per-row embed failures are counted and skipped — the cursor advances past
    them so one permanently-failing event can never wedge the migration.
    Chunk-level failures (DB/provider outage) propagate so the queue retries;
    the committed cursor makes the retry resume where it stopped.
    """
    from sqlalchemy import text, update

    from smritikosh.api.deps import get_llm
    from smritikosh.config import settings
    from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus, Event
    from smritikosh.db.postgres import db_session

    llm = get_llm()
    batch_size = max(1, settings.re_embed_batch_size)

    async with db_session() as pg:
        migration = await pg.get(EmbeddingMigration, uuid.UUID(migration_id))
        if migration is None:
            logger.warning("Embedding migration %s not found — stopping", migration_id)
            return "stopped"
        if migration.status != EmbeddingMigrationStatus.RUNNING:
            logger.info(
                "Embedding migration %s is %s — stopping", migration_id, migration.status
            )
            return "stopped"

        # Keyset pagination: strictly after the last processed (created_at, id).
        cursor_clause = ""
        params: dict = {"dim": migration.target_dim, "batch": batch_size}
        if migration.cursor_created_at is not None and migration.cursor_id is not None:
            cursor_clause = "AND (created_at, id) > (:cursor_created, :cursor_id)"
            params["cursor_created"] = migration.cursor_created_at
            params["cursor_id"] = migration.cursor_id

        rows = (await pg.execute(
            text(
                f"SELECT id, raw_text, created_at FROM events "
                f"WHERE {_STALE_PREDICATE} {cursor_clause} "
                f"ORDER BY created_at, id LIMIT :batch"
            ),
            params,
        )).fetchall()

        if not rows:
            migration.status = EmbeddingMigrationStatus.COMPLETE
            migration.finished_at = datetime.now(timezone.utc)
            await pg.flush()
            await pg.commit()
            await _emit_reembed_complete_audit(migration_id, migration.processed, migration.errors, migration.total)
            logger.info(
                "Embedding migration %s complete: processed=%d errors=%d",
                migration_id, migration.processed, migration.errors,
            )
            return "complete"

        chunk_errors = 0
        for row in rows:
            try:
                embedding = await llm.embed(row.raw_text)
                await pg.execute(
                    update(Event)
                    .where(Event.id == row.id)
                    .values(embedding=embedding, updated_at=datetime.now(timezone.utc))
                )
            except Exception as exc:
                logger.warning("Re-embed failed for event %s: %s", row.id, exc)
                chunk_errors += 1

        last = rows[-1]
        migration.processed += len(rows)
        migration.errors += chunk_errors
        migration.cursor_created_at = last.created_at
        migration.cursor_id = last.id
        await pg.flush()
        await pg.commit()

    logger.info(
        "Embedding migration %s chunk done: +%d rows (%d errors)",
        migration_id, len(rows), chunk_errors,
    )
    return "continue"


async def _emit_reembed_complete_audit(
    migration_id: str, processed: int, errors: int, total: int
) -> None:
    """Audit the completion (no-op when MongoDB is not configured)."""
    try:
        from smritikosh.api.deps import get_audit_logger
        from smritikosh.audit.logger import AuditEvent, EventType

        audit = get_audit_logger()
        if audit:
            await audit.emit_sync(AuditEvent(
                event_type=EventType.EMBEDDING_REEMBED_COMPLETE,
                user_id="__system__",
                app_id="__system__",
                payload={
                    "migration_id": migration_id,
                    "processed": processed,
                    "errors": errors,
                    "total": total,
                },
            ))
    except Exception as exc:  # pragma: no cover - audit must never break the task
        logger.debug("Re-embed audit emit failed: %s", exc)


async def _run_embedding_migration_inline(migration_id: str) -> dict:
    """In-process fallback (no Redis): loop chunks until a terminal state."""
    chunks = 0
    while True:
        outcome = await _run_embedding_migration_chunk(migration_id)
        if outcome != "continue":
            return {"migration_id": migration_id, "outcome": outcome, "chunks": chunks}
        chunks += 1


# ── Connector-token key rotation (C3) ───────────────────────────────────────────


async def _rotate_connector_tokens() -> dict:
    """Re-encrypt every stored connector token under the current primary key.

    Run after prepending a new secret to CONNECTOR_ENCRYPTION_KEYS. Idempotent:
    tokens already under the primary key are rewritten harmlessly; tokens no
    configured key can decrypt are counted as failed and left untouched (the
    user must re-authorise the connector).
    """
    from sqlalchemy import select

    from smritikosh.connectors.oauth import rotate_ciphertext
    from smritikosh.db.models import UserConnector
    from smritikosh.db.postgres import db_session

    rotated = 0
    failed = 0
    skipped = 0
    async with db_session() as pg:
        rows = (await pg.execute(select(UserConnector))).scalars().all()
        for connector in rows:
            if not connector.encrypted_tokens:
                skipped += 1
                continue
            try:
                connector.encrypted_tokens = rotate_ciphertext(connector.encrypted_tokens)
                rotated += 1
            except Exception:
                logger.warning(
                    "Connector token rotation failed — no configured key decrypts it "
                    "(user must re-authorise): user=%s provider=%s",
                    connector.user_id,
                    connector.provider,
                )
                failed += 1
        await pg.commit()

    logger.info(
        "Connector key rotation complete: rotated=%d failed=%d skipped=%d",
        rotated, failed, skipped,
    )
    return {"rotated": rotated, "failed": failed, "skipped": skipped, "total": len(rows)}


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


# ── Prediction outcome recording (E4) ───────────────────────────────────────────


async def _record_prediction_outcome(
    prediction_id: str, actual_event_ids: list[str]
) -> dict:
    """Score one memory prediction against what /context actually surfaced.

    Shared by the ARQ task and the in-process fallback. The engine opens its
    own session; scoring an already-scored or missing prediction is a no-op.
    """
    from smritikosh.api.deps import get_prediction_engine

    engine = get_prediction_engine()
    hit_rate = await engine.record_outcome_by_id(prediction_id, actual_event_ids)
    return {"prediction_id": prediction_id, "hit_rate": hit_rate}


# ── ARQ task wrappers ───────────────────────────────────────────────────────────


async def process_media(ctx, media_id: str) -> str:
    """ARQ task: process one uploaded media file."""
    return await _process_media_record(media_id)


async def re_embed_events(ctx, migration_id: str | None = None) -> dict:
    """ARQ task: process one chunk of a resumable embedding migration (H1).

    Called without a migration_id it creates or resumes one (kept for
    backward compatibility with jobs enqueued before chunking existed).
    After each chunk it re-enqueues itself, so a worker restart between
    chunks loses nothing and no single task ever exceeds the job timeout.
    """
    from smritikosh.tasks.queue import enqueue

    if migration_id is None:
        migration_id, _total, _resumed = await _create_or_resume_migration()

    outcome = await _run_embedding_migration_chunk(migration_id)
    if outcome == "continue":
        job = await enqueue("re_embed_events", migration_id)
        if job is None:
            # Queue vanished mid-migration (Redis outage) — finish inline
            # rather than stranding a half-done migration.
            return await _run_embedding_migration_inline(migration_id)
    return {"migration_id": migration_id, "outcome": outcome}


async def reconsolidate_recalled(
    ctx, event_ids: list[str], query: str, user_id: str, app_id: str = "default"
) -> dict:
    """ARQ task: reconsolidate events recalled by a /context call."""
    return await _reconsolidate_recalled(event_ids, query, user_id, app_id)


async def rotate_connector_tokens(ctx) -> dict:
    """ARQ task: re-encrypt all connector tokens under the primary key (C3)."""
    return await _rotate_connector_tokens()


async def record_prediction_outcome(
    ctx, prediction_id: str, actual_event_ids: list[str]
) -> dict:
    """ARQ task: score a memory prediction against surfaced events (E4)."""
    return await _record_prediction_outcome(prediction_id, actual_event_ids)


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

    functions = [
        process_media,
        re_embed_events,
        reconsolidate_recalled,
        rotate_connector_tokens,
        record_prediction_outcome,
    ]
    on_startup = _on_startup
    on_shutdown = _on_shutdown
    max_tries = 3                 # retry transient failures
    job_timeout = 3600            # media transcription of a long recording can be slow


# Bind the Redis connection only when configured. Importing this module from the
# API in a Redis-less dev setup must not fail — the taskworker is the only thing
# that needs REDIS_URL, and it is never run without it.
if queue_enabled():
    WorkerSettings.redis_settings = redis_settings()
