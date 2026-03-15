"""
POST /feedback — submit user feedback on a recalled memory event.
"""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_reinforcement
from smritikosh.api.schemas import FeedbackRequest, FeedbackResponse
from smritikosh.db.models import FeedbackType
from smritikosh.db.postgres import get_session
from smritikosh.processing.reinforcement import ReinforcementLoop

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(
    request: FeedbackRequest,
    pg: AsyncSession = Depends(get_session),
    loop: ReinforcementLoop = Depends(get_reinforcement),
) -> FeedbackResponse:
    """
    Record user feedback on a recalled memory and adjust its importance score.

    - ``positive`` feedback boosts the event's importance_score by 0.10.
    - ``negative`` feedback reduces it by 0.10.
    - ``neutral`` records the signal without changing the score.

    The updated importance_score influences all future hybrid_search rankings
    for this event.
    """
    try:
        event_id = uuid.UUID(request.event_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid event_id UUID format.")

    try:
        feedback_type = FeedbackType(request.feedback_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feedback_type '{request.feedback_type}'. "
                   f"Must be one of: positive, negative, neutral.",
        )

    try:
        feedback, new_score = await loop.submit(
            pg,
            event_id=event_id,
            user_id=request.user_id,
            app_id=request.app_id,
            feedback_type=feedback_type,
            comment=request.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(
            "Feedback submission failed",
            extra={"user_id": request.user_id, "event_id": request.event_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return FeedbackResponse(
        feedback_id=str(feedback.id),
        event_id=request.event_id,
        new_importance_score=new_score,
    )
