"""
Session ingest routes — passive memory extraction from conversation transcripts.

POST /ingest/session
    Post a full or partial conversation transcript. The endpoint:
    1. Checks idempotency (session_id already processed → return cached result)
    2. Filters to user turns only + strips sentinel blocks
    3. Optionally runs TriggerDetector pre-filter (skip LLM if no triggers)
    4. Calls LLM with delta-extraction prompt (only NEW or CONTRADICTING facts)
    5. Upserts surviving facts to Neo4j (SemanticMemory)
    6. Stores one episodic event summarising the session (EpisodicMemory)
    7. Records the session in processed_sessions (for idempotency / streaming)

POST /ingest/transcript
    Alias for /ingest/session — accepts the same body, for backwards compat.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncSession as NeoSession
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_hippocampus, get_llm, get_semantic
from smritikosh.auth.deps import assert_self_or_admin, require_write_scope
from smritikosh.db.models import ProcessedSession, SourceType
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.hippocampus import Hippocampus, _EXTRACTION_SCHEMA, _EXTRACTION_EXAMPLE
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.transcript_utils import build_delta_prompt, prepare_transcript
from smritikosh.processing.trigger_detector import TriggerDetector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

_trigger_detector = TriggerDetector()


# ── Request / Response models ─────────────────────────────────────────────────


class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Turn content")


class SessionIngestRequest(BaseModel):
    user_id: str = Field(..., description="User who had the conversation")
    app_id: str = Field("default", description="Application namespace")
    session_id: str = Field(..., description="Idempotency key — re-posting same session is a no-op")
    turns: list[ConversationTurn] = Field(..., description="Conversation turns in order")
    partial: bool = Field(False, description="True for mid-session streaming windows; False (default) for final session close")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional: timestamp, device, channel, etc.")
    use_trigger_filter: bool = Field(True, description="Skip LLM extraction if no high-signal phrases are detected (cost saver)")
    dry_run: bool = Field(False, description="Run extraction but do not persist anything — useful for debugging extraction quality")


class SessionIngestResponse(BaseModel):
    session_id: str
    user_id: str
    app_id: str
    turns_processed: int
    facts_extracted: int
    skipped_duplicates: int
    extraction_skipped: bool    # True if trigger filter fired and no triggers found
    already_processed: bool     # True if this session_id was already complete
    partial: bool
    dry_run: bool = False
    extracted_facts: list[dict] = Field(default_factory=list, description="Populated in dry_run mode — shows what would be stored")


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post("/session", response_model=SessionIngestResponse, status_code=201)
async def ingest_session(
    request: SessionIngestRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    llm: Annotated[LLMAdapter, Depends(get_llm)],
    semantic: Annotated[SemanticMemory, Depends(get_semantic)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(require_write_scope)],
) -> SessionIngestResponse:
    """
    Extract memories from a conversation session transcript.

    The endpoint is idempotent: posting the same session_id twice is safe.
    For streaming (long-lived sessions), set partial=True and post windows as
    they accumulate. The server tracks the last processed turn index, so each
    partial post only processes new turns.

    Anti-contamination steps applied automatically:
    - Only user turns are analysed (assistant turns discarded)
    - Injected context sentinels are stripped before extraction
    - Extraction LLM receives existing facts and is told to extract only NEW info
    - Optional trigger-word pre-filter skips the LLM entirely for low-signal windows
    """
    assert_self_or_admin(current_user, request.user_id)

    turns_raw = [t.model_dump() for t in request.turns]

    # ── 1. Idempotency check (skip in dry_run) ────────────────────────────────
    existing_session = None
    if not request.dry_run:
        existing_session = await _get_processed_session(
            pg, request.user_id, request.app_id, request.session_id
        )

    if existing_session and not existing_session.is_partial and not request.partial:
        # Complete session already processed — return cached result
        return SessionIngestResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            app_id=request.app_id,
            turns_processed=existing_session.turns_count,
            facts_extracted=existing_session.facts_extracted,
            skipped_duplicates=0,
            extraction_skipped=False,
            already_processed=True,
            partial=False,
        )

    last_turn_index = existing_session.last_turn_index if existing_session else 0

    # ── 2. Prepare transcript (filter + strip sentinels) ──────────────────────
    transcript = prepare_transcript(turns_raw, last_turn_index=last_turn_index)

    if transcript.turns_count == 0:
        # Nothing new to process
        return SessionIngestResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            app_id=request.app_id,
            turns_processed=0,
            facts_extracted=0,
            skipped_duplicates=0,
            extraction_skipped=True,
            already_processed=False,
            partial=request.partial,
        )

    # ── 3. Trigger-word pre-filter ────────────────────────────────────────────
    extraction_skipped = False
    if request.use_trigger_filter:
        has_triggers = _trigger_detector.any_triggered(transcript.user_turns)
        if not has_triggers:
            extraction_skipped = True
            logger.debug(
                "No triggers found in session window — skipping LLM extraction",
                extra={"session_id": request.session_id, "user_id": request.user_id},
            )
            # Still update the turn index so we don't re-scan these turns (skip in dry_run)
            if not request.dry_run:
                await _upsert_processed_session(
                    pg, request.user_id, request.app_id, request.session_id,
                    turns_count=len(turns_raw),
                    facts_extracted=existing_session.facts_extracted if existing_session else 0,
                    last_turn_index=len(transcript.user_turns) + last_turn_index,
                    is_partial=request.partial,
                )
            return SessionIngestResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                app_id=request.app_id,
                turns_processed=transcript.turns_count,
                facts_extracted=0,
                skipped_duplicates=0,
                extraction_skipped=True,
                already_processed=False,
                partial=request.partial,
                dry_run=request.dry_run,
            )

    # ── 4. Fetch existing facts for delta extraction ──────────────────────────
    profile = await semantic.get_user_profile(neo, request.user_id, request.app_id, min_confidence=0.5)
    existing_facts = profile.facts[:30] if profile else []

    # ── 5. LLM delta extraction ───────────────────────────────────────────────
    prompt = build_delta_prompt(
        transcript.user_turns, existing_facts, last_turn_index=0  # already windowed
    )

    trigger_phrases = _trigger_detector.collect_all_phrases(transcript.user_turns)
    source_type = (
        SourceType.TRIGGER_WORD if trigger_phrases else SourceType.PASSIVE_DISTILLATION
    )
    source_meta: dict[str, Any] = {
        "session_id": request.session_id,
        "partial": request.partial,
        "turns_window": [last_turn_index, last_turn_index + transcript.turns_count],
    }
    if trigger_phrases:
        source_meta["trigger_phrases"] = trigger_phrases

    extracted_facts: list[dict] = []
    try:
        result = await llm.extract_structured(
            prompt=prompt,
            schema_description=_EXTRACTION_SCHEMA,
            example_output=_EXTRACTION_EXAMPLE,
        )
        extracted_facts = result.get("facts", [])
    except Exception:
        logger.exception(
            "LLM extraction failed for session",
            extra={"session_id": request.session_id, "user_id": request.user_id},
        )
        # Don't abort — store the session event without facts

    # ── 6. Store episodic event (skip in dry_run) ─────────────────────────────
    if not request.dry_run:
        session_summary = (
            f"Session {request.session_id}: {transcript.turns_count} user turns. "
            f"Extracted {len(extracted_facts)} facts."
        )
        try:
            await hippocampus.encode(
                pg,
                neo,
                user_id=request.user_id,
                raw_text=transcript.combined_text[:2000],
                app_id=request.app_id,
                metadata={**request.metadata, "session_id": request.session_id, "summary": session_summary},
                source_type=source_type,
                source_meta=source_meta,
            )
        except Exception:
            logger.exception(
                "Failed to store episodic event for session",
                extra={"session_id": request.session_id},
            )

    # ── 7. Record processed session (skip in dry_run) ─────────────────────────
    if not request.dry_run:
        total_facts = (existing_session.facts_extracted if existing_session else 0) + len(extracted_facts)
        await _upsert_processed_session(
            pg, request.user_id, request.app_id, request.session_id,
            turns_count=len(turns_raw),
            facts_extracted=total_facts,
            last_turn_index=last_turn_index + transcript.turns_count,
            is_partial=request.partial,
        )

    logger.info(
        "Session ingest %s",
        "dry_run complete" if request.dry_run else "complete",
        extra={
            "session_id": request.session_id,
            "user_id": request.user_id,
            "turns": transcript.turns_count,
            "facts": len(extracted_facts),
            "source_type": source_type,
            "dry_run": request.dry_run,
        },
    )

    return SessionIngestResponse(
        session_id=request.session_id,
        user_id=request.user_id,
        app_id=request.app_id,
        turns_processed=transcript.turns_count,
        facts_extracted=len(extracted_facts),
        skipped_duplicates=0,
        extraction_skipped=extraction_skipped,
        already_processed=False,
        partial=request.partial,
        dry_run=request.dry_run,
        extracted_facts=extracted_facts if request.dry_run else [],
    )


# ── Alias endpoint ────────────────────────────────────────────────────────────


@router.post("/transcript", response_model=SessionIngestResponse, status_code=201)
async def ingest_transcript(
    request: SessionIngestRequest,
    hippocampus: Annotated[Hippocampus, Depends(get_hippocampus)],
    llm: Annotated[LLMAdapter, Depends(get_llm)],
    semantic: Annotated[SemanticMemory, Depends(get_semantic)],
    pg: Annotated[AsyncSession, Depends(get_session)],
    neo: Annotated[NeoSession, Depends(get_neo4j_session)],
    current_user: Annotated[dict, Depends(require_write_scope)],
) -> SessionIngestResponse:
    """Alias for POST /ingest/session — same behaviour, kept for backwards compat."""
    return await ingest_session(
        request, hippocampus, llm, semantic, pg, neo, current_user
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_processed_session(
    pg: AsyncSession,
    user_id: str,
    app_id: str,
    session_id: str,
) -> ProcessedSession | None:
    result = await pg.execute(
        select(ProcessedSession).where(
            ProcessedSession.user_id == user_id,
            ProcessedSession.app_id == app_id,
            ProcessedSession.session_id == session_id,
        )
    )
    return result.scalar_one_or_none()


async def _upsert_processed_session(
    pg: AsyncSession,
    user_id: str,
    app_id: str,
    session_id: str,
    *,
    turns_count: int,
    facts_extracted: int,
    last_turn_index: int,
    is_partial: bool,
) -> ProcessedSession:
    from datetime import datetime, timezone
    existing = await _get_processed_session(pg, user_id, app_id, session_id)
    if existing:
        existing.turns_count = turns_count
        existing.facts_extracted = facts_extracted
        existing.last_turn_index = last_turn_index
        existing.is_partial = is_partial
        existing.processed_at = datetime.now(timezone.utc)
        return existing
    record = ProcessedSession(
        user_id=user_id,
        app_id=app_id,
        session_id=session_id,
        turns_count=turns_count,
        facts_extracted=facts_extracted,
        last_turn_index=last_turn_index,
        is_partial=is_partial,
    )
    pg.add(record)
    await pg.flush()
    return record
