"""
Cognitive agent layer routes (item E4).

POST /agent/decision                        Memory-grounded decision recommendation
                                            with cited evidence (DecisionAgent).
GET  /cognition/predictions/{user_id}       Recent predict-observe-learn cycles +
                                            rolling accuracy (PredictionEngine).
GET  /cognition/reflections/{user_id}       Insights from reflection cycles
                                            (drift, contradictions, stale beliefs).
POST /cognition/reflections/{user_id}/{id}/ack   Acknowledge an insight so it
                                            stops surfacing as a nudge.
"""

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import (
    get_decision_agent,
    get_neo4j_session,
    get_prediction_engine,
    get_reflection_agent,
)
from smritikosh.api.quotas import enforce_token_quota
from smritikosh.api.ratelimit import limiter
from smritikosh.api.schemas import (
    BeliefAlignmentItem,
    DecisionRequest,
    DecisionResponse,
    PredictionItem,
    PredictionListResponse,
    ReflectionAckResponse,
    ReflectionItem,
    ReflectionListResponse,
)
from smritikosh.auth.deps import assert_app_access, assert_self_or_admin, get_current_user
from smritikosh.config import settings
from smritikosh.db.postgres import get_session
from smritikosh.llm.usage import llm_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["cognition"])


# ── Decision agent ─────────────────────────────────────────────────────────────


@router.post("/agent/decision", response_model=DecisionResponse)
@limiter.limit(lambda: settings.rate_limit_context or "10000/minute")
async def agent_decision(
    request: Request,
    body: DecisionRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> DecisionResponse:
    """
    Reason over the user's memory about a decision and return a recommendation
    with cited evidence (event IDs), belief-alignment analysis, risks, and the
    open questions memory cannot answer. The decision + recommendation is also
    logged back as an episodic event so future reasoning can learn from it.
    """
    assert_self_or_admin(current_user, body.user_id)
    resolved_app_ids = body.app_ids or current_user.get("app_ids")
    app_id = resolved_app_ids[0] if resolved_app_ids else "default"
    await enforce_token_quota(pg, body.user_id, app_id)

    agent = get_decision_agent()
    try:
        with llm_context(user_id=body.user_id, app_id=app_id, source="decision_agent"):
            result = await agent.decide(
                pg,
                neo,
                user_id=body.user_id,
                decision=body.decision,
                options=body.options,
                app_ids=resolved_app_ids,
            )
    except Exception as exc:
        logger.exception("DecisionAgent failed", extra={"user_id": body.user_id})
        raise HTTPException(status_code=500, detail=f"Decision synthesis failed: {exc}") from exc

    return DecisionResponse(
        user_id=result.user_id,
        app_id=result.app_id,
        decision=result.decision,
        recommendation=result.recommendation,
        reasoning=result.reasoning,
        confidence=result.confidence,
        belief_alignment=[
            BeliefAlignmentItem(belief=ba.belief, alignment=ba.alignment, note=ba.note)
            for ba in result.belief_alignment
        ],
        risks=result.risks,
        cited_event_ids=result.cited_event_ids,
        open_questions=result.open_questions,
        memories_considered=result.memories_considered,
        logged_event_id=result.logged_event_id,
        skipped=result.skipped,
        skip_reason=result.skip_reason,
    )


# ── Predictions ────────────────────────────────────────────────────────────────


@router.get("/cognition/predictions/{user_id}", response_model=PredictionListResponse)
async def list_predictions(
    user_id: str,
    app_id: Annotated[str, Query()] = "default",
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    days: Annotated[int, Query(ge=1, le=365)] = 30,
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> PredictionListResponse:
    """Recent predict-observe-learn cycles and the rolling prediction accuracy."""
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)

    engine = get_prediction_engine()
    accuracy = await engine.accuracy(pg, user_id, app_id, days=days)
    predictions = await engine.recent_predictions(pg, user_id, app_id, limit=limit)

    return PredictionListResponse(
        user_id=user_id,
        app_id=app_id,
        accuracy=accuracy,
        predictions=[
            PredictionItem(
                prediction_id=str(p.id),
                query_preview=p.query_preview,
                intent=p.intent,
                predicted_event_ids=[str(i) for i in (p.predicted_event_ids or [])],
                predicted_cluster_ids=[int(i) for i in (p.predicted_cluster_ids or [])],
                actual_event_ids=[str(i) for i in (p.actual_event_ids or [])],
                hit_rate=p.hit_rate,
                created_at=p.created_at.isoformat() if p.created_at else "",
                scored_at=p.scored_at.isoformat() if p.scored_at else None,
            )
            for p in predictions
        ],
    )


# ── Reflections ────────────────────────────────────────────────────────────────


@router.get("/cognition/reflections/{user_id}", response_model=ReflectionListResponse)
async def list_reflections(
    user_id: str,
    app_id: Annotated[str, Query()] = "default",
    include_acknowledged: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> ReflectionListResponse:
    """Insights from reflection cycles: drift, contradictions, stale beliefs."""
    assert_self_or_admin(current_user, user_id)
    assert_app_access(current_user, app_id)

    agent = get_reflection_agent()
    rows = await agent.list_reflections(
        pg, user_id, app_id, include_acknowledged=include_acknowledged, limit=limit
    )
    return ReflectionListResponse(
        user_id=user_id,
        app_id=app_id,
        reflections=[
            ReflectionItem(
                reflection_id=str(r.id),
                kind=r.kind,
                insight=r.insight,
                severity=r.severity,
                evidence=r.evidence or {},
                acknowledged=r.acknowledged,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
            for r in rows
        ],
    )


@router.post(
    "/cognition/reflections/{user_id}/{reflection_id}/ack",
    response_model=ReflectionAckResponse,
)
async def acknowledge_reflection(
    user_id: str,
    reflection_id: str,
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> ReflectionAckResponse:
    """Acknowledge an insight so it stops re-surfacing (and stops being
    re-fed to future reflection cycles as an open item)."""
    assert_self_or_admin(current_user, user_id)
    try:
        rid = uuid.UUID(reflection_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="reflection_id must be a UUID.")

    agent = get_reflection_agent()
    ok = await agent.acknowledge(pg, user_id, rid)
    if not ok:
        raise HTTPException(status_code=404, detail="Reflection not found.")
    return ReflectionAckResponse(reflection_id=reflection_id, acknowledged=True)
