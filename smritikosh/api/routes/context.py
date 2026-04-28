"""
Context route — retrieve assembled memory context for an LLM call.

POST /context        Build MemoryContext from all memory systems and return it
                     as a prompt-ready string + OpenAI-style message list.

POST /context/stream Same as above but returns Server-Sent Events so clients
                     can start rendering as each layer (procedures, recent,
                     similar events, identity) arrives, rather than waiting for
                     the full assembly.

After the response is returned, a background task reconsolidates the top
recalled event — updating its summary to incorporate the new recall context.
This mirrors human memory reconsolidation (recalled memories are re-saved
with new associations) without adding latency to the API response.
"""

import asyncio
import json
import logging
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.ratelimit import limiter
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_context_builder, get_reconsolidation_engine
from smritikosh.config import settings
from smritikosh.api.schemas import ContextRequest, ContextResponse, ProcedureItem
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.processing.reconsolidation import ReconsolidationEngine
from smritikosh.retrieval.context_builder import ContextBuilder, MemoryContext

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
        _app_id = resolved_app_ids[0] if resolved_app_ids else "default"
        background_tasks.add_task(
            reconsolidation.reconsolidate_after_recall,
            ctx.similar_events,
            body.query,
            body.user_id,
            _app_id,
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
        procedures=[
            ProcedureItem(
                procedure_id=str(p.id),
                trigger=p.trigger,
                instruction=p.instruction,
                category=p.category,
                priority=p.priority,
                is_active=p.is_active,
                hit_count=p.hit_count or 0,
            )
            for p in ctx.procedures
        ],
    )


# ── Streaming endpoint ────────────────────────────────────────────────────────

def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@router.post("/context/stream")
@limiter.limit(lambda: settings.rate_limit_context or "10000/minute")
async def stream_context(
    request: Request,
    body: ContextRequest,
    builder: Annotated[ContextBuilder, Depends(get_context_builder)],
    reconsolidation: Annotated[ReconsolidationEngine, Depends(get_reconsolidation_engine)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> StreamingResponse:
    """
    Streaming variant of POST /context — returns Server-Sent Events.

    Each memory layer is emitted as soon as it is ready, so clients can
    start rendering before the full context is assembled:

      event: intent
      data: {"intent": "TECHNICAL", "via_llm": false}

      event: procedures
      data: {"data": [...rules...], "count": 2}

      event: recent
      data: {"data": [...events...], "count": 5}

      event: similar
      data: {"data": [...events...], "count": 5, "embedding_failed": false}

      event: identity
      data: {"facts": [...], "summary": "..."}

      event: done
      data: {"context_text": "...", "messages": [...], "total_memories": 12,
             "reconsolidation_scheduled": true}

    On error an `error` event is emitted and the stream closes.

    Usage (JavaScript):
        const es = new EventSource('/context/stream', { method: 'POST', body: ... });
        es.addEventListener('similar', (e) => renderMemories(JSON.parse(e.data)));
        es.addEventListener('done', (e) => injectContext(JSON.parse(e.data)));
    """
    assert_self_or_admin(current_user, body.user_id)
    resolved_app_ids = body.app_ids or current_user.get("app_ids")

    async def generate() -> AsyncIterator[str]:
        try:
            # ── 1. Intent classification ──────────────────────────────────
            intent_result = (
                await builder.intent_classifier.classify_async(body.query)
                if builder.intent_classifier is not None else None
            )
            detected_intent = intent_result.intent if intent_result else "GENERAL"
            yield _sse("intent", {
                "intent": str(detected_intent),
                "via_llm": intent_result.via_llm if intent_result else False,
            })

            # ── 2. Embed query ────────────────────────────────────────────
            embedding, embedding_failed = await builder._embed_query(body.query, body.user_id)

            # ── 3. Start Neo4j profile in background ─────────────────────
            neo4j_app_id = resolved_app_ids[0] if resolved_app_ids else "default"
            profile_task = asyncio.create_task(
                builder.semantic.get_user_profile(
                    neo, body.user_id, neo4j_app_id,
                    min_confidence=builder.min_profile_confidence,
                )
            )

            # ── 4. Procedures (PG) ───────────────────────────────────────��
            try:
                procedures_raw = (
                    await builder.procedural.search_by_query(
                        pg, body.user_id, body.query,
                        app_ids=resolved_app_ids, top_k=builder.top_k_procedures,
                    )
                    if builder.procedural is not None else []
                )
            except Exception:
                procedures_raw = []

            yield _sse("procedures", {
                "data": [
                    {
                        "procedure_id": str(p.id),
                        "trigger": p.trigger,
                        "instruction": p.instruction,
                        "category": p.category,
                        "priority": p.priority,
                    }
                    for p in procedures_raw
                ],
                "count": len(procedures_raw),
            })

            # ── 5. Recent events (PG) ─────────────────────────────────────
            try:
                recent_raw = await builder.episodic.get_recent(
                    pg, body.user_id, resolved_app_ids, limit=builder.recent_limit,
                    from_date=body.from_date, to_date=body.to_date,
                )
            except Exception:
                recent_raw = []

            yield _sse("recent", {
                "data": [
                    {"event_id": str(e.id), "raw_text": e.raw_text,
                     "importance_score": e.importance_score,
                     "created_at": e.created_at.isoformat() if e.created_at else None}
                    for e in recent_raw
                ],
                "count": len(recent_raw),
            })

            # ── 6. Similar events / hybrid search (PG) ────────────────────
            try:
                similar_raw = (
                    await builder.episodic.hybrid_search(
                        pg, body.user_id, embedding, app_ids=resolved_app_ids,
                        top_k=builder.top_k_similar,
                        weights_override=intent_result.weights if intent_result else None,
                        from_date=body.from_date, to_date=body.to_date,
                    )
                    if embedding is not None else []
                )
            except Exception:
                similar_raw = []

            yield _sse("similar", {
                "data": [
                    {"event_id": str(sr.event.id), "raw_text": sr.event.raw_text,
                     "score": sr.score, "importance_score": sr.event.importance_score,
                     "created_at": sr.event.created_at.isoformat() if sr.event.created_at else None}
                    for sr in similar_raw
                ],
                "count": len(similar_raw),
                "embedding_failed": embedding_failed,
            })

            # ── 7. Identity / Neo4j profile ───────────────────────────────
            try:
                profile = await profile_task
            except Exception:
                profile = None

            facts_out = []
            summary_out = ""
            if profile:
                facts_out = [
                    {"category": f.category, "key": f.key, "value": f.value,
                     "confidence": f.confidence}
                    for f in (profile.facts or [])
                ]
                summary_out = profile.summary or ""

            yield _sse("identity", {"facts": facts_out, "summary": summary_out})

            # ── 8. Assemble full context + reconsolidation ────────────────
            ctx = MemoryContext(
                user_id=body.user_id,
                query=body.query,
                intent=str(detected_intent),
                similar_events=similar_raw if isinstance(similar_raw, list) else [],
                user_profile=profile if not isinstance(profile, Exception) else None,
                recent_events=recent_raw if isinstance(recent_raw, list) else [],
                procedures=procedures_raw if isinstance(procedures_raw, list) else [],
                embedding_failed=embedding_failed,
            )

            # Recall increment
            if ctx.similar_events:
                try:
                    await builder.episodic.increment_recall(
                        pg, [sr.event.id for sr in ctx.similar_events]
                    )
                except Exception:
                    pass

            reconsolidation_scheduled = False
            if ctx.similar_events:
                _app_id = resolved_app_ids[0] if resolved_app_ids else "default"
                asyncio.create_task(
                    reconsolidation.reconsolidate_after_recall(
                        ctx.similar_events, body.query, body.user_id, _app_id,
                    )
                )
                reconsolidation_scheduled = True

            yield _sse("done", {
                "context_text": ctx.as_prompt_text(),
                "messages": ctx.as_messages(),
                "total_memories": ctx.total_memories(),
                "embedding_failed": embedding_failed,
                "intent": str(detected_intent),
                "reconsolidation_scheduled": reconsolidation_scheduled,
                "procedures": [
                    {"procedure_id": str(p.id), "trigger": p.trigger,
                     "instruction": p.instruction, "category": p.category,
                     "priority": p.priority}
                    for p in (procedures_raw if isinstance(procedures_raw, list) else [])
                ],
            })

        except Exception as exc:
            logger.exception("Streaming context failed", extra={"user_id": body.user_id})
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable Nginx buffering
    })
