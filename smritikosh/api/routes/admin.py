"""
Admin routes — manual triggers for background memory maintenance jobs.

POST /admin/consolidate    Run memory consolidation for one user or all active users.
POST /admin/prune          Run synaptic pruning for one user or all users.
POST /admin/cluster        Run memory clustering for one user or all users.
POST /admin/mine-beliefs   Run belief mining for one user or all users.

These endpoints expose the same logic that the MemoryScheduler runs automatically,
useful for debugging, testing, or forcing an immediate maintenance cycle.
"""

import logging
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_audit_logger, get_reconsolidation_engine
from smritikosh.api.quotas import quota_usage_snapshot
from smritikosh.api.schemas import (
    AdminJobRequest,
    AdminJobResponse,
    AdminJobResult,
    AdminUserItem,
    AdminUserPatch,
    AdminUsersResponse,
    EmbeddingHealthResponse,
    EmbeddingMigrationItem,
    EmbeddingMigrationStatusResponse,
    ReconsolidateRequest,
    ReconsolidateResponse,
    ReEmbedResponse,
)
from smritikosh.auth.deps import require_admin
from smritikosh.config import settings
from smritikosh.db.models import AppUser, LlmUsage, UserQuota
from smritikosh.db.postgres import get_session
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


# ── Reflection cycles (E4) ─────────────────────────────────────────────────────


@router.post("/reflect", response_model=AdminJobResponse)
async def trigger_reflection(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run reflection cycles immediately (drift/contradiction detection).

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    if body.user_id:
        result = await scheduler.run_reflection_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_reflection_for_all_users()

    return AdminJobResponse(
        job="reflection",
        users_processed=len(results),
        results=[
            AdminJobResult(
                user_id=r.user_id,
                app_id=r.app_id,
                skipped=r.skipped,
                detail=(
                    r.skip_reason
                    if r.skipped
                    else f"insights_found={r.insights_found} stored={r.insights_stored}"
                ),
            )
            for r in results
        ],
    )


@router.post("/nudge", response_model=AdminJobResponse)
async def trigger_nudge(
    body: AdminJobRequest,
    scheduler: Annotated[MemoryScheduler, Depends(_get_scheduler)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> AdminJobResponse:
    """
    Run Life OS nudge cycles immediately (digest fresh reflection insights).

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users. LLM-free.
    """
    if body.user_id:
        result = await scheduler.run_lifeos_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_lifeos_for_all_users()

    return AdminJobResponse(
        job="lifeos",
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
                        f"insights={r.insights_included} channel={r.channel} "
                        f"delivered={r.delivered}"
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


def _migration_item(m) -> EmbeddingMigrationItem:
    return EmbeddingMigrationItem(
        migration_id=str(m.id),
        status=m.status,
        target_model=m.target_model,
        target_dim=m.target_dim,
        total=m.total,
        processed=m.processed,
        errors=m.errors,
        progress_pct=round(100.0 * m.processed / m.total, 2) if m.total else 100.0,
        started_at=m.started_at.isoformat() if m.started_at else "",
        updated_at=m.updated_at.isoformat() if m.updated_at else "",
        finished_at=m.finished_at.isoformat() if m.finished_at else None,
        error_message=m.error_message,
    )


@router.post("/re-embed", response_model=ReEmbedResponse)
async def trigger_re_embed(
    background_tasks: BackgroundTasks,
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
    audit=Depends(get_audit_logger),
) -> ReEmbedResponse:
    """
    Re-embed all events whose embedding dimension doesn't match EMBEDDING_DIMENSIONS,
    plus any events that were stored without an embedding.

    The run is a resumable, chunked embedding migration (item H1): progress and
    a keyset cursor are committed after every RE_EMBED_BATCH_SIZE events, so a
    crash or deploy loses at most one chunk. If a migration is already running,
    this call resumes it (from its cursor) instead of starting a second one.
    Track progress with GET /admin/re-embed/status; cancel with DELETE /admin/re-embed.
    """
    from smritikosh.audit.logger import AuditEvent, EventType
    from smritikosh.tasks import enqueue
    from smritikosh.tasks.jobs import (
        _create_or_resume_migration,
        _run_embedding_migration_inline,
    )

    configured_dim = settings.embedding_dimensions

    row = await pg.execute(
        text(
            "SELECT count(*) AS n FROM events "
            "WHERE embedding IS NULL OR vector_dims(embedding) != :dim"
        ),
        {"dim": configured_dim},
    )
    stale = int(row.scalar() or 0)

    if stale == 0:
        if audit:
            await audit.emit(AuditEvent(
                event_type=EventType.EMBEDDING_REEMBED_QUEUED,
                user_id=_admin["sub"],
                app_id="__system__",
                payload={"queued": 0, "configured_dim": configured_dim, "triggered_by": _admin["sub"]},
            ))
        return ReEmbedResponse(status="ok", queued=0, message="No stale embeddings found.")

    migration_id, total, resumed = await _create_or_resume_migration()

    # Durable queue when available; in-process chunk loop otherwise.
    job = await enqueue("re_embed_events", migration_id)
    if job is None:
        background_tasks.add_task(_run_embedding_migration_inline, migration_id)

    if audit:
        await audit.emit(AuditEvent(
            event_type=EventType.EMBEDDING_REEMBED_QUEUED,
            user_id=_admin["sub"],
            app_id="__system__",
            payload={
                "queued": stale,
                "configured_dim": configured_dim,
                "migration_id": migration_id,
                "resumed": resumed,
                "triggered_by": _admin["sub"],
            },
        ))

    logger.info(
        "Re-embed %s: %d stale events (migration %s)",
        "resumed" if resumed else "started", stale, migration_id,
    )
    return ReEmbedResponse(
        status="resumed" if resumed else "started",
        queued=stale,
        migration_id=migration_id,
    )


@router.get("/re-embed/status", response_model=EmbeddingMigrationStatusResponse)
async def re_embed_status(
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> EmbeddingMigrationStatusResponse:
    """Progress of the running embedding migration plus recent run history (H1)."""
    from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus

    result = await pg.execute(
        select(EmbeddingMigration)
        .order_by(EmbeddingMigration.started_at.desc())
        .limit(5)
    )
    migrations = list(result.scalars().all())
    current = next(
        (m for m in migrations if m.status == EmbeddingMigrationStatus.RUNNING), None
    )
    return EmbeddingMigrationStatusResponse(
        current=_migration_item(current) if current else None,
        history=[_migration_item(m) for m in migrations],
    )


@router.delete("/re-embed", response_model=EmbeddingMigrationStatusResponse)
async def cancel_re_embed(
    pg: AsyncSession = Depends(get_session),
    _admin: dict = Depends(require_admin),
) -> EmbeddingMigrationStatusResponse:
    """Cancel the running embedding migration. The in-flight chunk finishes;
    the next chunk sees the cancelled status and stops. Re-POST /admin/re-embed
    to start a fresh run over whatever is still stale."""
    from datetime import datetime, timezone

    from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus

    result = await pg.execute(
        select(EmbeddingMigration).where(
            EmbeddingMigration.status == EmbeddingMigrationStatus.RUNNING
        )
    )
    cancelled = 0
    for m in result.scalars().all():
        m.status = EmbeddingMigrationStatus.CANCELLED
        m.finished_at = datetime.now(timezone.utc)
        cancelled += 1
    await pg.flush()
    if cancelled:
        logger.info("Cancelled %d running embedding migration(s)", cancelled)
    return await re_embed_status(pg=pg, _admin=_admin)


@router.get("/llm-usage", summary="LLM token/cost accounting")
async def get_llm_usage(
    pg: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict, Depends(require_admin)],
    days: Annotated[int, Query(ge=1, le=365, description="Reporting window in days")] = 30,
    group_by: Annotated[
        str, Query(description="Aggregation key: model | source | user | kind")
    ] = "model",
) -> dict:
    """
    Aggregate LLM spend from the llm_usage table (item D1).

    Rows are written per billed API call by the LLM adapter, attributed to
    (user_id, app_id, source) via the ambient llm_context. Calls made before
    accounting was enabled (or by providers that return no usage data, e.g.
    some local models) are absent — treat totals as a lower bound for cost,
    not an exact invoice.
    """
    group_columns = {
        "model": LlmUsage.model,
        "source": LlmUsage.source,
        "user": LlmUsage.user_id,
        "kind": LlmUsage.kind,
    }
    column = group_columns.get(group_by)
    if column is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid group_by '{group_by}'. Must be one of: {', '.join(group_columns)}.",
        )

    since = text("NOW() - make_interval(days => :days)")
    result = await pg.execute(
        select(
            column.label("key"),
            func.count().label("calls"),
            func.sum(LlmUsage.prompt_tokens).label("prompt_tokens"),
            func.sum(LlmUsage.completion_tokens).label("completion_tokens"),
            func.sum(LlmUsage.cost_usd).label("cost_usd"),
        )
        .where(LlmUsage.created_at >= since)
        .group_by(column)
        .order_by(func.sum(LlmUsage.cost_usd).desc()),
        {"days": days},
    )
    groups = [
        {
            "key": row.key or "(none)",
            "calls": int(row.calls or 0),
            "prompt_tokens": int(row.prompt_tokens or 0),
            "completion_tokens": int(row.completion_tokens or 0),
            "cost_usd": round(float(row.cost_usd or 0.0), 6),
        }
        for row in result
    ]
    return {
        "days": days,
        "group_by": group_by,
        "groups": groups,
        "total_calls": sum(g["calls"] for g in groups),
        "total_prompt_tokens": sum(g["prompt_tokens"] for g in groups),
        "total_completion_tokens": sum(g["completion_tokens"] for g in groups),
        "total_cost_usd": round(sum(g["cost_usd"] for g in groups), 6),
    }


# ── Usage quotas (D2) ──────────────────────────────────────────────────────────


class QuotaUpdateRequest(BaseModel):
    """All fields optional: null clears the override (config default applies)."""

    app_id: str = "default"
    daily_event_limit: Optional[int] = Field(None, ge=0)
    monthly_event_limit: Optional[int] = Field(None, ge=0)
    daily_token_limit: Optional[int] = Field(None, ge=0)
    monthly_token_limit: Optional[int] = Field(None, ge=0)
    note: Optional[str] = None


@router.get("/quotas/{user_id}", summary="Effective quota + current usage")
async def get_quota(
    user_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict, Depends(require_admin)],
    app_id: Annotated[str, Query()] = "default",
) -> dict:
    """
    Return the tenant's effective limits (override merged over config
    defaults; null = unlimited) and consumption in the current UTC day/month
    windows. Token usage comes from the llm_usage accounting table.
    """
    return await quota_usage_snapshot(pg, user_id, app_id)


@router.put("/quotas/{user_id}", summary="Set per-tenant quota overrides")
async def put_quota(
    user_id: str,
    body: QuotaUpdateRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> dict:
    """
    Upsert the (user_id, app_id) quota row. A null field clears that override
    so the QUOTA_DEFAULT_* config value applies again; 0 there = unlimited.
    Takes effect on the tenant's next request — no restart needed.
    """
    result = await pg.execute(
        select(UserQuota).where(
            UserQuota.user_id == user_id, UserQuota.app_id == body.app_id
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        row = UserQuota(user_id=user_id, app_id=body.app_id)
        pg.add(row)

    row.daily_event_limit = body.daily_event_limit
    row.monthly_event_limit = body.monthly_event_limit
    row.daily_token_limit = body.daily_token_limit
    row.monthly_token_limit = body.monthly_token_limit
    row.note = body.note
    await pg.flush()

    logger.info(
        "Quota updated",
        extra={"user_id": user_id, "app_id": body.app_id, "by": _admin["sub"]},
    )
    return await quota_usage_snapshot(pg, user_id, body.app_id)


@router.delete("/quotas/{user_id}", summary="Remove per-tenant quota overrides")
async def delete_quota(
    user_id: str,
    pg: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict, Depends(require_admin)],
    app_id: Annotated[str, Query()] = "default",
) -> dict:
    """Delete the override row; the tenant reverts to the config defaults."""
    result = await pg.execute(
        select(UserQuota).where(
            UserQuota.user_id == user_id, UserQuota.app_id == app_id
        )
    )
    row = result.scalar_one_or_none()
    deleted = row is not None
    if deleted:
        await pg.delete(row)
        await pg.flush()
        logger.info(
            "Quota override removed",
            extra={"user_id": user_id, "app_id": app_id, "by": _admin["sub"]},
        )
    return {"user_id": user_id, "app_id": app_id, "deleted": deleted}


# ── Connector encryption-key rotation (C3) ─────────────────────────────────────


@router.post("/rotate-connector-keys")
async def rotate_connector_keys(
    _admin: Annotated[dict, Depends(require_admin)],
) -> dict:
    """
    Re-encrypt every stored connector token under the current primary key.

    Rotation procedure:
      1. Prepend the new secret: CONNECTOR_ENCRYPTION_KEYS="new,old" (restart)
      2. Call this endpoint — tokens are re-encrypted under the new key
      3. Drop the old secret: CONNECTOR_ENCRYPTION_KEYS="new" (restart)

    Runs on the durable queue when Redis is configured; inline otherwise.
    Tokens that no configured key can decrypt are left untouched and counted
    as failed — those users must re-authorise their connector.
    """
    from smritikosh.tasks import enqueue
    from smritikosh.tasks.jobs import _rotate_connector_tokens

    job = await enqueue("rotate_connector_tokens")
    if job is not None:
        return {"queued": True, "job_id": job.job_id}
    result = await _rotate_connector_tokens()
    return {"queued": False, **result}
