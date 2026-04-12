"""
Context route — retrieve assembled memory context for an LLM call.

POST /context   Build MemoryContext from all memory systems and return it
                as a prompt-ready string + OpenAI-style message list.

After the response is returned, a background task reconsolidates the top
recalled event — updating its summary to incorporate the new recall context.
This mirrors human memory reconsolidation (recalled memories are re-saved
with new associations) without adding latency to the API response.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.ratelimit import limiter
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_context_builder, get_reconsolidation_engine
from smritikosh.config import settings
from smritikosh.api.schemas import ContextRequest, ContextResponse
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)
router = APIRouter(tags=["context"])


@router.post("/context", response_model=ContextResponse)
@limiter.limit(lambda: settings.rate_limit_context or "10000/minute")
async def get_context(
    request: Request,
    body: ContextRequest,
    background_tasks: BackgroundTasks,
    builder: Annotated[ContextBuilder, Depends(get_context_builder)],
    reconsolidation: Annotated[ReconsolidationEngine, Depends(get_reconsolidation_engine)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> ContextResponse:
    """
    Assemble memory context for a user query.

    Concurrently fetches:
      - Semantically similar past events (hybrid vector + recency + importance search)
      - User identity profile (Neo4j facts: preferences, interests, roles …)
      - Recent event timeline
      - Matching behavioral rules (procedural memory)

    Returns:
      - context_text: structured markdown, ready to prepend to your LLM system prompt
      - messages: OpenAI-style [{role: system, content: ...}] — append user turn and call LLM
      - reconsolidation_scheduled: true if a background memory update was queued

    After the response is sent, the top recalled event is silently reconsolidated
    in the background (its summary is refined with the current query context).
    Partial context is returned even if one memory system is unavailable.
    """
    assert_self_or_admin(current_user, body.user_id)
    resolved_app_ids = body.app_ids or current_user.get("app_ids")
    try:
        ctx = await builder.build(
            pg,
            neo,
            user_id=body.user_id,
            query=body.query,
            app_ids=resolved_app_ids,
            from_date=body.from_date,
            to_date=body.to_date,
        )
    except Exception as exc:
        logger.exception("ContextBuilder failed", extra={"user_id": body.user_id})
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {exc}") from exc

    # Schedule background reconsolidation for the top recalled event.
    # The ReconsolidationEngine opens its own session — the request session
    # may already be closed by the time the background task runs.
    reconsolidation_scheduled = False
    if ctx.similar_events:
        background_tasks.add_task(
            reconsolidation.reconsolidate_after_recall,
            ctx.similar_events,
            body.query,
            body.user_id,
        )
        reconsolidation_scheduled = True

    return ContextResponse(
        user_id=body.user_id,
        query=body.query,
        context_text=ctx.as_prompt_text(),
        messages=ctx.as_messages(),
        total_memories=ctx.total_memories(),
        embedding_failed=ctx.embedding_failed,
        intent=ctx.intent,
        reconsolidation_scheduled=reconsolidation_scheduled,
    )
