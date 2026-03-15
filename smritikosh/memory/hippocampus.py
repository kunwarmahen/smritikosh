"""
Hippocampus — memory intake coordinator.

In the human brain, the hippocampus is the gateway to long-term memory:
it receives sensory input, decides what is worth storing, and routes
encoded memories to the appropriate cortical regions.

Here it orchestrates the full encoding pipeline for a single interaction:

    raw text
        │
        ├─► Amygdala.score()              → importance_score     (sync)
        │
        ├─► LLMAdapter.embed()            → embedding            ┐ parallel
        └─► LLMAdapter.extract_structured → extracted facts      ┘ (asyncio.gather)
                │
                ├─► EpisodicMemory.store()    → Event   (Postgres)
                └─► SemanticMemory.upsert_fact() × N    (Neo4j)
                        │
                        └─► EncodedMemory  (returned to caller)

Design principles:
  - Embedding and fact extraction run concurrently to minimise latency.
  - If extraction fails, the event is still stored (embedding is the fallback).
  - If Neo4j is unavailable, Postgres write still succeeds.
  - Hippocampus never commits — callers own transaction boundaries.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from neo4j import AsyncSession as NeoSession

from smritikosh.db.models import Event
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.semantic import FactRecord, SemanticMemory
from smritikosh.processing.amygdala import Amygdala

logger = logging.getLogger(__name__)


# ── Fact extraction prompt constants ─────────────────────────────────────────

_EXTRACTION_SCHEMA = (
    "facts: list of objects with fields: "
    "category (one of: preference, interest, role, project, skill, goal, relationship), "
    "key (short snake_case label e.g. ui_color, current_project), "
    "value (concise string), "
    "confidence (float 0.0–1.0). "
    "Only extract clear, durable facts — skip questions, hypotheticals, and filler."
)

_EXTRACTION_EXAMPLE = {
    "facts": [
        {"category": "preference", "key": "ui_color",        "value": "green",            "confidence": 0.9},
        {"category": "interest",   "key": "domain",          "value": "AI agents",        "confidence": 0.95},
        {"category": "role",       "key": "current",         "value": "entrepreneur",     "confidence": 1.0},
        {"category": "project",    "key": "active",          "value": "smritikosh",       "confidence": 0.85},
        {"category": "skill",      "key": "rag",             "value": "experienced",      "confidence": 0.8},
    ]
}


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class EncodedMemory:
    """
    Result of a single Hippocampus.encode() call.

    Contains everything that was stored so the caller can log, audit, or
    immediately use the results without extra DB queries.
    """
    event: Event
    facts: list[FactRecord] = field(default_factory=list)
    importance_score: float = 1.0
    extraction_failed: bool = False   # True if LLM extraction threw an error


# ── Hippocampus ───────────────────────────────────────────────────────────────

class Hippocampus:
    """
    Central intake coordinator — the entry point for all memory writes.

    Inject dependencies (LLMAdapter, EpisodicMemory, SemanticMemory, Amygdala)
    rather than instantiating them internally, making the class testable with
    lightweight mocks.

    Usage:
        hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await hippo.encode(
                pg, neo,
                user_id="u1",
                raw_text="I decided to build an AI memory startup",
            )
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        amygdala: Amygdala | None = None,
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.amygdala = amygdala or Amygdala()

    # ── Primary entry point ────────────────────────────────────────────────

    async def encode(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        raw_text: str,
        app_id: str = "default",
        metadata: dict | None = None,
    ) -> EncodedMemory:
        """
        Full memory encoding pipeline for one interaction.

        Steps (in order):
            1. Score importance with Amygdala (sync, no I/O).
            2. Concurrently: generate embedding + extract structured facts.
            3. Store event to EpisodicMemory (Postgres).
            4. Upsert each extracted fact to SemanticMemory (Neo4j).

        Failures are handled gracefully:
            - Extraction errors → event stored with empty facts list.
            - Embedding errors  → event stored without embedding vector.
            - Neo4j errors      → event stored; fact upserts skipped with warning.
        """
        # ── 1. Importance scoring (sync) ──────────────────────────────────
        importance_score = self.amygdala.score(raw_text)
        logger.debug(
            "Amygdala scored event",
            extra={"user_id": user_id, "score": importance_score, "text_preview": raw_text[:80]},
        )

        # ── 2. Embed + extract in parallel ────────────────────────────────
        embedding, extracted_facts, extraction_failed = await self._embed_and_extract(
            raw_text, user_id
        )

        # ── 3. Store episodic event ───────────────────────────────────────
        event = await self.episodic.store(
            pg_session,
            user_id=user_id,
            app_id=app_id,
            raw_text=raw_text,
            embedding=embedding,
            importance_score=importance_score,
            metadata=metadata,
        )
        logger.info(
            "Episodic event stored",
            extra={"event_id": str(event.id), "user_id": user_id, "facts_extracted": len(extracted_facts)},
        )

        # ── 4. Upsert semantic facts ──────────────────────────────────────
        stored_facts = await self._upsert_facts(
            neo_session, user_id, app_id, extracted_facts
        )

        return EncodedMemory(
            event=event,
            facts=stored_facts,
            importance_score=importance_score,
            extraction_failed=extraction_failed,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _embed_and_extract(
        self, raw_text: str, user_id: str
    ) -> tuple[list[float] | None, list[dict], bool]:
        """
        Run embedding generation and fact extraction concurrently.

        Returns:
            (embedding, extracted_fact_dicts, extraction_failed)
        """
        embed_task = self.llm.embed(raw_text)
        extract_task = self.llm.extract_structured(
            prompt=f"Extract structured facts from this user interaction:\n\n{raw_text}",
            schema_description=_EXTRACTION_SCHEMA,
            example_output=_EXTRACTION_EXAMPLE,
        )

        results = await asyncio.gather(embed_task, extract_task, return_exceptions=True)

        # ── Embedding result ──────────────────────────────────────────────
        embedding: list[float] | None = None
        if isinstance(results[0], Exception):
            logger.warning(
                "Embedding generation failed — event stored without vector",
                extra={"user_id": user_id, "error": str(results[0])},
            )
        else:
            embedding = results[0]

        # ── Extraction result ─────────────────────────────────────────────
        extracted_facts: list[dict] = []
        extraction_failed = False
        if isinstance(results[1], Exception):
            logger.warning(
                "Fact extraction failed — event stored with no semantic facts",
                extra={"user_id": user_id, "error": str(results[1])},
            )
            extraction_failed = True
        else:
            extracted_facts = results[1].get("facts", [])

        return embedding, extracted_facts, extraction_failed

    async def _upsert_facts(
        self,
        neo_session: NeoSession,
        user_id: str,
        app_id: str,
        fact_dicts: list[dict],
    ) -> list[FactRecord]:
        """Upsert each extracted fact dict to SemanticMemory. Skips invalid entries."""
        stored: list[FactRecord] = []
        for fd in fact_dicts:
            try:
                fact = await self.semantic.upsert_fact(
                    neo_session,
                    user_id=user_id,
                    app_id=app_id,
                    category=fd["category"],
                    key=fd["key"],
                    value=fd["value"],
                    confidence=float(fd.get("confidence", 1.0)),
                )
                stored.append(fact)
            except (KeyError, ValueError) as exc:
                # Log and skip — a bad fact dict should not abort the whole pipeline
                logger.warning(
                    "Skipping invalid fact dict",
                    extra={"fact": fd, "error": str(exc)},
                )
        return stored
