"""
Admin routes — manual triggers for background memory maintenance jobs.

POST /admin/consolidate    Run memory consolidation for one user or all active users.
POST /admin/prune          Run synaptic pruning for one user or all users.
POST /admin/cluster        Run memory clustering for one user or all users.
POST /admin/mine-beliefs   Run belief mining for one user or all users.

These endpoints expose the same logic that the MemoryScheduler runs automatically,
useful for debugging, testing, or forcing an immediate maintenance cycle.
"""

import asyncio
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_audit_logger, get_llm, get_reconsolidation_engine
from smritikosh.api.schemas import (
    AdminJobRequest,
    AdminJobResponse,
    AdminJobResult,
    AdminUserItem,
    AdminUserPatch,
    AdminUsersResponse,
    EmbeddingHealthResponse,
    ReconsolidateRequest,
    ReconsolidateResponse,
    ReEmbedResponse,
)
from smritikosh.auth.deps import require_admin
from smritikosh.config import settings
from smritikosh.db.models import AppUser, Event
from smritikosh.db.postgres import db_session, get_session
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.processing.scheduler import MemoryScheduler
from smritikosh.processing.synaptic_pruner import PruningThresholds

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _get_scheduler(request: Request) -> MemoryScheduler:
    """Retrieve the MemoryScheduler stored on app.state during startup."""
    scheduler: MemoryScheduler | None = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialised.")
    return scheduler


# ── Consolidation ──────────────────────────────────────────────────────────────


@router.post("/consolidate", response_model=AdminJobResponse)
async def trigger_consolidation(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run memory consolidation immediately.

    If ``user_id`` is provided, runs for that user only.
    If omitted, discovers all users with unconsolidated events and processes each.
    """
    if body.user_id:
        result = await scheduler.run_consolidation_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_consolidation_for_all_users()

    return AdminJobResponse(
        job="consolidation",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=(
                    r.skip_reason
                    if r.skipped
                    else f"consolidated={r.events_consolidated} facts={r.facts_distilled}"
                ),
            )
            for r in results
        ],
    )


# ── Pruning ────────────────────────────────────────────────────────────────────


@router.post("/prune", response_model=AdminJobResponse)
async def trigger_pruning(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run synaptic pruning immediately.

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    override = None
    if any(v is not None for v in (body.min_age_days, body.importance_threshold, body.min_recall_count)):
        from smritikosh.processing.synaptic_pruner import DEFAULT_IMPORTANCE_THRESHOLD, DEFAULT_MIN_AGE_DAYS, DEFAULT_MIN_RECALL_COUNT
        override = PruningThresholds(
            importance_threshold=body.importance_threshold if body.importance_threshold is not None else DEFAULT_IMPORTANCE_THRESHOLD,
            min_recall_count=body.min_recall_count if body.min_recall_count is not None else DEFAULT_MIN_RECALL_COUNT,
            min_age_days=body.min_age_days if body.min_age_days is not None else DEFAULT_MIN_AGE_DAYS,
        )

    if body.user_id:
        result = await scheduler.run_pruning_now(
            user_id=body.user_id, app_id=body.app_id, override_thresholds=override
        )
        results = [result]
    else:
        results = await scheduler.run_pruning_for_all_users(override_thresholds=override)

    return AdminJobResponse(
        job="pruning",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=f"evaluated={r.events_evaluated} pruned={r.events_pruned}",
            )
            for r in results
        ],
    )


# ── Clustering ─────────────────────────────────────────────────────────────────


@router.post("/cluster", response_model=AdminJobResponse)
async def trigger_clustering(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run memory clustering immediately.

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    if body.user_id:
        result = await scheduler.run_clustering_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_clustering_for_all_users()

    return AdminJobResponse(
        job="clustering",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=r.skip_reason if r.skipped else f"clusters={r.clusters_found} events_clustered={r.events_clustered}",
            )
            for r in results
        ],
    )


# ── Reconsolidation ───────────────────────────────────────────────────────────


@router.post("/reconsolidate", response_model=ReconsolidateResponse)
async def trigger_reconsolidation(
    body: ReconsolidateRequest,
    engine: Annotated[ReconsolidationEngine, Depends(get_reconsolidation_engine)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> ReconsolidateResponse:
    """
    Manually reconsolidate a specific memory event.

    Useful for testing or forcing an update on a high-value event.
    The engine applies the same gate conditions as automatic reconsolidation
    (recall_count threshold, importance threshold, cooldown).

    Set ``query`` to the context in which the memory was recalled.
    """
    result = await engine.reconsolidate_event(
        event_id_str=body.event_id,
        query=body.query,
        user_id=body.user_id,
        force=body.force,
    )
    return ReconsolidateResponse(
        event_id=result.event_id,
        user_id=result.user_id,
        updated=result.updated,
        skipped=result.skipped,
        skip_reason=result.skip_reason,
        old_summary=result.old_summary,
        new_summary=result.new_summary,
    )


# ── Belief mining ──────────────────────────────────────────────────────────────


@router.post("/mine-beliefs", response_model=AdminJobResponse)
async def trigger_belief_mining(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run belief mining immediately.

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    if body.user_id:
        result = await scheduler.run_belief_mining_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_belief_mining_for_all_users()

    return AdminJobResponse(
        job="belief_mining",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=r.skip_reason if r.skipped else f"beliefs_upserted={r.beliefs_upserted}",
            )
            for r in results
        ],
    )


# ── Cross-system synthesis ─────────────────────────────────────────────────────


@router.post("/synthesize", response_model=AdminJobResponse)
async def trigger_synthesis(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run cross-system synthesis immediately.

    Correlates connector signals (calendar, email, Slack) with recent
    episodic events to infer durable behavioral patterns.

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    if body.user_id:
        result = await scheduler.run_synthesis_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_synthesis_for_all_users()

    return AdminJobResponse(
        job="cross_system_synthesis",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=(
                    r.skip_reason
                    if r.skipped
                    else (
                        f"sources={','.join(r.connector_sources_found)} "
                        f"facts_synthesized={r.facts_synthesized} "
                        f"facts_pending={r.facts_pending}"
                    )
                ),
            )
            for r in results
        ],
    )


# ── User management ────────────────────────────────────────────────────────────


@router.get("/users", response_model=AdminUsersResponse)
async def list_users(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    role: Optional[str] = None,
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> AdminUsersResponse:
    """
    Return a paginated list of all registered users.

    Requires admin role.
    """
    q = select(AppUser).order_by(AppUser.created_at.desc())
    if role:
        q = q.where(AppUser.role == role)

    total_result = await pg.execute(select(func.count()).select_from(q.subquery()))
    total = total_result.scalar_one()

    users_result = await pg.execute(q.limit(limit).offset(offset))
    users = users_result.scalars().all()

    return AdminUsersResponse(
        users=[
            AdminUserItem(
                username=u.username,
                email=u.email,
                role=u.role,
                app_ids=u.app_ids,
                is_active=u.is_active,
                created_at=u.created_at.isoformat() if u.created_at else "",
                updated_at=u.updated_at.isoformat() if u.updated_at else "",
            )
            for u in users
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/users/{username}", response_model=AdminUserItem)
async def get_user(
    username: str,
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> AdminUserItem:
    """Return a single user by username. Requires admin role."""
    result = await pg.execute(select(AppUser).where(AppUser.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return AdminUserItem(
        username=user.username,
        email=user.email,
        role=user.role,
        app_ids=user.app_ids,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else "",
        updated_at=user.updated_at.isoformat() if user.updated_at else "",
    )


@router.patch("/users/{username}", response_model=AdminUserItem)
async def patch_user(
    username: str,
    body: AdminUserPatch,
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> AdminUserItem:
    """
    Update a user's ``is_active`` flag or ``role``.

    Only the fields included in the request body are updated.
    Requires admin role.
    """
    result = await pg.execute(select(AppUser).where(AppUser.username == username))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role is not None:
        if body.role not in ("admin", "user"):
            raise HTTPException(status_code=422, detail="role must be 'admin' or 'user'.")
        user.role = body.role
    if body.app_ids is not None:
        user.app_ids = body.app_ids

    await pg.flush()

    return AdminUserItem(
        username=user.username,
        email=user.email,
        role=user.role,
        app_ids=user.app_ids,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else "",
        updated_at=user.updated_at.isoformat() if user.updated_at else "",
    )


# ── Embedding health ───────────────────────────────────────────────────────────


@router.get("/embedding-health", response_model=EmbeddingHealthResponse)
async def embedding_health(
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> EmbeddingHealthResponse:
    """
    Report how many stored embeddings match the currently configured dimension.

    ``stale_events`` counts embeddings whose vector_dims() != EMBEDDING_DIMENSIONS.
    Any non-zero value means the database contains vectors from a previous model
    and hybrid search will produce incorrect similarity scores.

    Run POST /admin/re-embed to fix stale embeddings.
    """
    configured_dim = settings.embedding_dimensions

    total_result = await pg.execute(
        text("SELECT COUNT(*) FROM events WHERE embedding IS NOT NULL")
    )
    total_embedded: int = total_result.scalar_one()

    null_result = await pg.execute(
        text("SELECT COUNT(*) FROM events WHERE embedding IS NULL")
    )
    null_embeddings: int = null_result.scalar_one()

    stale_result = await pg.execute(
        text(
            "SELECT COUNT(*) FROM events "
            "WHERE embedding IS NOT NULL AND vector_dims(embedding) != :dim"
        ),
        {"dim": configured_dim},
    )
    stale_events: int = stale_result.scalar_one()

    return EmbeddingHealthResponse(
        configured_dim=configured_dim,
        total_embedded=total_embedded,
        stale_events=stale_events,
        null_embeddings=null_embeddings,
        healthy=(stale_events == 0),
    )


@router.post("/re-embed", response_model=ReEmbedResponse)
async def trigger_re_embed(
    background_tasks: BackgroundTasks,
    pg: AsyncSession = Depends(get_session),
    llm: LLMAdapter = Depends(get_llm),
    _admin: dict = Depends(require_admin),
    audit=Depends(get_audit_logger),
) -> ReEmbedResponse:
    """
    Re-embed all events whose embedding dimension doesn't match EMBEDDING_DIMENSIONS,
    plus any events that were stored without an embedding.

    The job runs as a FastAPI background task — the response returns immediately
    with the number of events queued.  Check GET /admin/embedding-health afterward
    to confirm all stale embeddings are resolved.
    """
    from smritikosh.audit.logger import AuditEvent, EventType

    configured_dim = settings.embedding_dimensions

    rows = await pg.execute(
        text(
            "SELECT id, raw_text FROM events "
            "WHERE embedding IS NULL "
            "   OR vector_dims(embedding) != :dim "
            "ORDER BY created_at ASC"
        ),
        {"dim": configured_dim},
    )
    stale = rows.fetchall()

    if not stale:
        if audit:
            await audit.emit(AuditEvent(
                event_type=EventType.EMBEDDING_REEMBED_QUEUED,
                user_id=_admin["sub"],
                app_id="__system__",
                payload={"queued": 0, "configured_dim": configured_dim, "triggered_by": _admin["sub"]},
            ))
        return ReEmbedResponse(status="ok", queued=0, message="No stale embeddings found.")

    stale_snapshot = [(str(row.id), row.raw_text) for row in stale]
    background_tasks.add_task(_re_embed_events, stale_snapshot, llm, audit)

    if audit:
        await audit.emit(AuditEvent(
            event_type=EventType.EMBEDDING_REEMBED_QUEUED,
            user_id=_admin["sub"],
            app_id="__system__",
            payload={"queued": len(stale_snapshot), "configured_dim": configured_dim, "triggered_by": _admin["sub"]},
        ))

    logger.info("Re-embed queued: %d events", len(stale_snapshot))
    return ReEmbedResponse(status="started", queued=len(stale_snapshot))


async def _re_embed_events(events: list[tuple[str, str]], llm: LLMAdapter, audit=None) -> None:
    """Background task: re-embed each stale event and update the DB."""
    import uuid as _uuid
    from datetime import datetime, timezone
    from sqlalchemy import update as _update

    from smritikosh.audit.logger import AuditEvent, EventType

    success = 0
    errors = 0
    async with db_session() as session:
        for event_id_str, raw_text in events:
            try:
                embedding = await llm.embed(raw_text)
                await session.execute(
                    _update(Event)
                    .where(Event.id == _uuid.UUID(event_id_str))
                    .values(embedding=embedding, updated_at=datetime.now(timezone.utc))
                )
                success += 1
            except Exception as exc:
                logger.warning("Re-embed failed for event %s: %s", event_id_str, exc)
                errors += 1

    logger.info(
        "Re-embed complete: success=%d errors=%d total=%d",
        success,
        errors,
        len(events),
    )

    if audit:
        await audit.emit_sync(AuditEvent(
            event_type=EventType.EMBEDDING_REEMBED_COMPLETE,
            user_id="__system__",
            app_id="__system__",
            payload={"success": success, "errors": errors, "total": len(events)},
        ))
