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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_context_builder, get_reconsolidation_engine
from smritikosh.api.schemas import ContextRequest, ContextResponse
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.retrieval.context_builder import ContextBuilder

logger = logging.getLogger(__name__)
router = APIRouter(tags=["context"])


@router.post("/context", response_model=ContextResponse)
async def get_context(
    request: ContextRequest,
    background_tasks: BackgroundTasks,
    builder: Annotated[ContextBuilder, Depends(get_context_builder)],
    reconsolidation: Annotated[ReconsolidationEngine, Depends(get_reconsolidation_engine)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
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
    try:
        ctx = await builder.build(
            pg,
            neo,
            user_id=request.user_id,
            query=request.query,
            app_id=request.app_id,
            from_date=request.from_date,
            to_date=request.to_date,
        )
    except Exception as exc:
        logger.exception("ContextBuilder failed", extra={"user_id": request.user_id})
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {exc}") from exc

    # Schedule background reconsolidation for the top recalled event.
    # The ReconsolidationEngine opens its own session — the request session
    # may already be closed by the time the background task runs.
    reconsolidation_scheduled = False
    if ctx.similar_events:
        background_tasks.add_task(
            reconsolidation.reconsolidate_after_recall,
            ctx.similar_events,
            request.query,
            request.user_id,
        )
        reconsolidation_scheduled = True

    return ContextResponse(
        user_id=request.user_id,
        query=request.query,
        context_text=ctx.as_prompt_text(),
        messages=ctx.as_messages(),
        total_memories=ctx.total_memories(),
        embedding_failed=ctx.embedding_failed,
        intent=ctx.intent,
        reconsolidation_scheduled=reconsolidation_scheduled,
    )
