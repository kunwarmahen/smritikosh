"""
ReinforcementLoop — closes the feedback loop between user signals and memory quality.

Mirrors synaptic strengthening/weakening in the brain:
  - Positive feedback (memory was useful) → raise importance_score.
  - Negative feedback (memory was irrelevant) → lower importance_score.
  - Neutral feedback → log the signal without adjusting the score.

This directly influences future hybrid_search rankings because importance_score
is one of the four weighted terms in the retrieval formula.

Usage:
    loop = ReinforcementLoop()

    async with db_session() as session:
        feedback, new_score = await loop.submit(
            session,
            event_id=event_id,
            user_id="u1",
            feedback_type=FeedbackType.POSITIVE,
            comment="This context was spot on!",
        )
        await session.commit()
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event, FeedbackType, MemoryFeedback

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

POSITIVE_DELTA = 0.10   # boost applied to importance_score on positive feedback
NEGATIVE_DELTA = 0.10   # penalty applied on negative feedback
SCORE_MIN = 0.0
SCORE_MAX = 1.0


# ── ReinforcementLoop ─────────────────────────────────────────────────────────


class ReinforcementLoop:
    """
    Stores user feedback and updates the associated event's importance_score.

    All methods accept an AsyncSession — callers own transaction boundaries.
    """

    # ── Write ──────────────────────────────────────────────────────────────

    async def submit(
        self,
        session: AsyncSession,
        *,
        event_id: uuid.UUID,
        user_id: str,
        app_id: str = "default",
        feedback_type: FeedbackType,
        comment: str | None = None,
    ) -> tuple[MemoryFeedback, float]:
        """
        Record feedback and adjust the event's importance_score.

        Steps:
            1. Verify the event exists (raises ValueError if not).
            2. Insert a MemoryFeedback record.
            3. If POSITIVE/NEGATIVE, update importance_score (clamped to [0, 1]).
            4. Flush so the feedback record gets its id assigned.

        Returns:
            (MemoryFeedback, new_importance_score)
            For NEUTRAL feedback the score is unchanged and returned as-is.

        Raises:
            ValueError: if event_id does not exist in the events table.
        """
        event = await session.get(Event, event_id)
        if event is None or event.user_id != user_id or event.app_id != app_id:
            raise ValueError(f"Event {event_id!s} not found or access denied.")

        feedback = MemoryFeedback(
            event_id=event_id,
            user_id=user_id,
            app_id=app_id,
            feedback_type=str(feedback_type),
            comment=comment,
        )
        session.add(feedback)

        new_score = apply_delta(event.importance_score, feedback_type)
        if new_score != event.importance_score:
            await session.execute(
                update(Event)
                .where(Event.id == event_id, Event.user_id == user_id, Event.app_id == app_id)
                .values(
                    importance_score=new_score,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            logger.debug(
                "Importance score updated via feedback",
                extra={
                    "user_id": user_id,
                    "event_id": str(event_id),
                    "feedback_type": str(feedback_type),
                    "old_score": event.importance_score,
                    "new_score": new_score,
                },
            )

        await session.flush()
        return feedback, new_score

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_feedback(
        self,
        session: AsyncSession,
        event_id: uuid.UUID,
        user_id: str | None = None,
        app_id: str = "default",
    ) -> list[MemoryFeedback]:
        """Return all feedback records for an event, newest first (multi-tenant safe)."""
        query = select(MemoryFeedback).where(MemoryFeedback.event_id == event_id)
        if user_id is not None:
            query = query.where(MemoryFeedback.user_id == user_id, MemoryFeedback.app_id == app_id)
        result = await session.execute(
            query.order_by(MemoryFeedback.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_user_feedback(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        limit: int = 50,
    ) -> list[MemoryFeedback]:
        """Return recent feedback submitted by a user."""
        result = await session.execute(
            select(MemoryFeedback)
            .where(
                MemoryFeedback.user_id == user_id,
                MemoryFeedback.app_id == app_id,
            )
            .order_by(MemoryFeedback.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())


# ── Pure helpers ──────────────────────────────────────────────────────────────


def apply_delta(current_score: float, feedback_type: FeedbackType) -> float:
    """
    Compute the new importance_score after applying a feedback signal.

    POSITIVE  →  min(1.0, score + POSITIVE_DELTA)
    NEGATIVE  →  max(0.0, score - NEGATIVE_DELTA)
    NEUTRAL   →  unchanged

    Always clamps to [SCORE_MIN, SCORE_MAX].
    """
    if feedback_type == FeedbackType.POSITIVE:
        return min(SCORE_MAX, current_score + POSITIVE_DELTA)
    if feedback_type == FeedbackType.NEGATIVE:
        return max(SCORE_MIN, current_score - NEGATIVE_DELTA)
    return current_score
