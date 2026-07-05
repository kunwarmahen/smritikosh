"""
PredictionEngine — the Predict-Observe-Learn loop (item E4, FUTURE.md #7).

Before /context assembles retrieval, the engine predicts which memories the
query will surface. After retrieval, the actual surfaced event IDs are
recorded against the prediction and a hit rate computed. The delta is the
learning signal:

    hits   (predicted AND surfaced)      → importance_score nudged up
    misses (predicted, NOT surfaced)     → importance_score nudged down

Over many cycles, memories that are *predictably useful* for a user rise in
importance (and therefore in hybrid-search ranking), while memories the
engine keeps over-predicting sink — the retrieval layer specialises to the
person, which no general model can do.

Prediction is deliberately LLM-free: two indexed Postgres queries (cluster
affinity + recall history), so the loop adds ~zero latency and zero cost to
/context. Outcome recording runs post-response on the task queue.

Usage:
    engine = PredictionEngine(episodic=episodic)

    prediction = await engine.predict(pg, user_id="u1", query="...", intent="career")
    ctx = await builder.build(...)
    await engine.record_outcome(pg, prediction.prediction_id,
                                [str(sr.event.id) for sr in ctx.similar_events])
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh import metrics
from smritikosh.db.models import Event, MemoryPrediction

logger = logging.getLogger(__name__)

# How many events / clusters one prediction names.
PREDICTED_EVENTS = 5
PREDICTED_CLUSTERS = 3

# Learning-rate defaults: small nudges, so no single cycle dominates the
# Amygdala/feedback signals that also move importance_score.
HIT_IMPORTANCE_BUMP = 0.02
MISS_IMPORTANCE_DECAY = 0.01


@dataclass
class Prediction:
    prediction_id: str
    user_id: str
    app_id: str
    predicted_event_ids: list[str] = field(default_factory=list)
    predicted_cluster_ids: list[int] = field(default_factory=list)


class PredictionEngine:
    """
    Predicts and scores which memories a query will surface.

    Args:
        hit_bump:   importance_score increase for correctly predicted events.
        miss_decay: importance_score decrease for predicted-but-unused events.
    """

    def __init__(
        self,
        *,
        hit_bump: float = HIT_IMPORTANCE_BUMP,
        miss_decay: float = MISS_IMPORTANCE_DECAY,
    ) -> None:
        self.hit_bump = hit_bump
        self.miss_decay = miss_decay

    # ── Predict (before retrieval) ─────────────────────────────────────────

    async def predict(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        query: str,
        intent: str = "general",
        app_ids: list[str] | None = None,
    ) -> Prediction:
        """
        Predict the memories this query will surface, and persist the guess.

        Heuristic (no LLM, no embedding):
          - clusters: the user's clusters ranked by accumulated recall — the
            themes this user keeps coming back to.
          - events: the user's most recalled-and-important events, biased
            toward the predicted clusters.
        """
        app_id = app_ids[0] if app_ids else "default"

        cluster_rows = await session.execute(
            select(Event.cluster_id, func.sum(Event.recall_count).label("recalls"))
            .where(
                Event.user_id == user_id,
                Event.cluster_id.is_not(None),
                *( [Event.app_id.in_(app_ids)] if app_ids is not None else [] ),
            )
            .group_by(Event.cluster_id)
            .order_by(text("recalls DESC"))
            .limit(PREDICTED_CLUSTERS)
        )
        cluster_ids = [int(r.cluster_id) for r in cluster_rows]

        event_query = (
            select(Event.id)
            .where(
                Event.user_id == user_id,
                *( [Event.app_id.in_(app_ids)] if app_ids is not None else [] ),
            )
            .limit(PREDICTED_EVENTS)
        )
        # Predictably-useful first: cluster affinity leads (when clusters
        # exist), recall history dominates within that, importance breaks ties.
        if cluster_ids:
            event_query = event_query.order_by(
                Event.cluster_id.in_(cluster_ids).desc(),
                Event.recall_count.desc(),
                Event.importance_score.desc(),
            )
        else:
            event_query = event_query.order_by(
                Event.recall_count.desc(),
                Event.importance_score.desc(),
            )
        event_rows = await session.execute(event_query)
        event_ids = [str(r.id) for r in event_rows]

        record = MemoryPrediction(
            user_id=user_id,
            app_id=app_id,
            query_preview=query[:300],
            intent=str(intent),
            predicted_event_ids=event_ids,
            predicted_cluster_ids=cluster_ids,
        )
        session.add(record)
        await session.flush()

        return Prediction(
            prediction_id=str(record.id),
            user_id=user_id,
            app_id=app_id,
            predicted_event_ids=event_ids,
            predicted_cluster_ids=cluster_ids,
        )

    # ── Observe + Learn (after retrieval) ──────────────────────────────────

    async def record_outcome(
        self,
        session: AsyncSession,
        prediction_id: str,
        actual_event_ids: list[str],
    ) -> float | None:
        """
        Score a prediction against what retrieval actually surfaced and apply
        the learning nudges. Returns the hit rate (None if the prediction row
        is gone or was already scored).
        """
        record = await session.get(MemoryPrediction, uuid.UUID(prediction_id))
        if record is None or record.scored_at is not None:
            return None

        predicted = set(record.predicted_event_ids or [])
        actual = set(actual_event_ids)
        hits = predicted & actual
        misses = predicted - actual
        hit_rate = (len(hits) / len(actual)) if actual else 0.0

        record.actual_event_ids = sorted(actual)
        record.hit_rate = hit_rate
        record.scored_at = datetime.now(timezone.utc)

        # Learn: predictably-useful memories rise; over-predicted ones sink.
        if hits and self.hit_bump > 0:
            await session.execute(
                update(Event)
                .where(Event.id.in_([uuid.UUID(i) for i in hits]))
                .values(
                    importance_score=func.least(
                        1.0, Event.importance_score + self.hit_bump
                    )
                )
            )
        if misses and self.miss_decay > 0:
            await session.execute(
                update(Event)
                .where(Event.id.in_([uuid.UUID(i) for i in misses]))
                .values(
                    importance_score=func.greatest(
                        0.0, Event.importance_score - self.miss_decay
                    )
                )
            )

        metrics.PREDICTION_HIT_RATE.observe(hit_rate)
        logger.debug(
            "Prediction scored",
            extra={
                "prediction_id": prediction_id,
                "hit_rate": hit_rate,
                "hits": len(hits),
                "misses": len(misses),
            },
        )
        return hit_rate

    async def record_outcome_by_id(
        self, prediction_id: str, actual_event_ids: list[str]
    ) -> float | None:
        """Queue-safe entry point: opens its own session (IDs cross the
        process boundary; the row is re-read in the worker)."""
        from smritikosh.db.postgres import db_session

        async with db_session() as pg:
            return await self.record_outcome(pg, prediction_id, actual_event_ids)

    # ── Read ───────────────────────────────────────────────────────────────

    async def recent_predictions(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        limit: int = 20,
    ) -> list[MemoryPrediction]:
        result = await session.execute(
            select(MemoryPrediction)
            .where(
                MemoryPrediction.user_id == user_id,
                MemoryPrediction.app_id == app_id,
            )
            .order_by(MemoryPrediction.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def accuracy(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        days: int = 30,
    ) -> dict:
        """Rolling prediction accuracy for a user: avg hit rate + sample size."""
        since = datetime.now(timezone.utc) - timedelta(days=days)
        result = await session.execute(
            select(
                func.avg(MemoryPrediction.hit_rate).label("avg_hit_rate"),
                func.count().label("scored"),
            ).where(
                MemoryPrediction.user_id == user_id,
                MemoryPrediction.app_id == app_id,
                MemoryPrediction.hit_rate.is_not(None),
                MemoryPrediction.created_at >= since,
            )
        )
        row = result.one()
        return {
            "days": days,
            "scored_predictions": int(row.scored or 0),
            "avg_hit_rate": round(float(row.avg_hit_rate), 4) if row.avg_hit_rate is not None else None,
        }
