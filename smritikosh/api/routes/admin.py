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
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from smritikosh.api.deps import get_reconsolidation_engine
from smritikosh.api.schemas import (
    AdminJobRequest,
    AdminJobResponse,
    AdminJobResult,
    ReconsolidateRequest,
    ReconsolidateResponse,
)
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.processing.scheduler import MemoryScheduler

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
) -> AdminJobResponse:
    """
    Run synaptic pruning immediately.

    If ``user_id`` is provided, runs for that user only.
    If omitted, runs for all users.
    """
    if body.user_id:
        result = await scheduler.run_pruning_now(
            user_id=body.user_id, app_id=body.app_id
        )
        results = [result]
    else:
        results = await scheduler.run_pruning_for_all_users()

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
