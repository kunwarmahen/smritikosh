"""
Consolidator — memory consolidation background process.

Mirrors the brain's memory consolidation during sleep:
  - Retrieves unconsolidated episodic events for a user.
  - Groups them into time-ordered batches.
  - Compresses each batch into a summary + distilled facts via LLM.
  - Marks original events as consolidated in EpisodicMemory.
  - Upserts distilled facts to SemanticMemory (reinforcing confidence).

Result: 10 raw events → 1 consolidated event + updated knowledge graph.
Memory shrinks and improves over time rather than growing indefinitely.

Run: periodically via the Scheduler (e.g. every hour).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event, RelationType
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.memory.semantic import FactRecord, SemanticMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_EVENTS_TO_CONSOLIDATE = 5     # skip if fewer unconsolidated events exist
BATCH_SIZE = 10                    # events processed per LLM call

_CONSOLIDATION_SCHEMA = (
    "summary (string), "
    "facts: list of objects with: "
    "category (preference|interest|role|project|skill|goal|relationship), "
    "key (short snake_case label), value (concise string), confidence (0.0–1.0). "
    "links: optional list of objects with: "
    "from_index (0-based int matching the interaction number), "
    "to_index (0-based int), "
    "relation_type (caused|preceded|related|contradicts). "
    "Only include clear, durable facts and unambiguous causal or temporal relationships."
)

_CONSOLIDATION_EXAMPLE = {
    "summary": "User is building an AI memory startup called smritikosh, prefers green UI.",
    "facts": [
        {"category": "project",    "key": "active",   "value": "smritikosh",  "confidence": 0.95},
        {"category": "preference", "key": "ui_color", "value": "green",       "confidence": 0.9},
    ],
    "links": [
        {"from_index": 0, "to_index": 1, "relation_type": "preceded"},
    ],
}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class ConsolidationResult:
    user_id: str
    app_id: str
    events_processed: int = 0
    events_consolidated: int = 0
    facts_distilled: int = 0
    links_created: int = 0
    batches: int = 0
    skipped: bool = False          # True if fewer than MIN_EVENTS_TO_CONSOLIDATE
    skip_reason: str = ""


# ── Consolidator ──────────────────────────────────────────────────────────────

class Consolidator:
    """
    Compresses unconsolidated episodic events into stable semantic facts.

    Injected dependencies (LLM, EpisodicMemory, SemanticMemory) are the same
    instances used by Hippocampus and ContextBuilder — no extra resources needed.

    Usage:
        consolidator = Consolidator(llm=llm, episodic=episodic, semantic=semantic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await consolidator.run(pg, neo, user_id="u1")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        narrative: NarrativeMemory | None = None,
        batch_size: int = BATCH_SIZE,
        min_events: int = MIN_EVENTS_TO_CONSOLIDATE,
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.narrative = narrative
        self.batch_size = batch_size
        self.min_events = min_events

    # ── Primary entry point ────────────────────────────────────────────────

    async def run(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> ConsolidationResult:
        """
        Run one consolidation cycle for a single user.

        Steps:
            1. Fetch unconsolidated events (oldest first).
            2. Guard: skip if fewer than min_events.
            3. Split into batches of batch_size.
            4. For each batch: compress → mark consolidated → upsert facts.
        """
        result = ConsolidationResult(user_id=user_id, app_id=app_id)

        # ── 1. Fetch unconsolidated events ────────────────────────────────
        events = await self.episodic.get_unconsolidated(
            pg_session, user_id, app_id=app_id, limit=self.batch_size * 10
        )
        result.events_processed = len(events)

        # ── 2. Guard ──────────────────────────────────────────────────────
        if len(events) < self.min_events:
            result.skipped = True
            result.skip_reason = (
                f"Only {len(events)} unconsolidated events — "
                f"need at least {self.min_events}."
            )
            logger.debug(
                "Consolidation skipped",
                extra={"user_id": user_id, "reason": result.skip_reason},
            )
            return result

        # ── 3. Batch and process ──────────────────────────────────────────
        batches = _split_batches(events, self.batch_size)
        result.batches = len(batches)

        for batch in batches:
            consolidated, facts, links = await self._consolidate_batch(
                pg_session, neo_session, user_id, app_id, batch
            )
            result.events_consolidated += consolidated
            result.facts_distilled += facts
            result.links_created += links

        logger.info(
            "Consolidation complete",
            extra={
                "user_id": user_id,
                "events_processed": result.events_processed,
                "events_consolidated": result.events_consolidated,
                "facts_distilled": result.facts_distilled,
                "links_created": result.links_created,
                "batches": result.batches,
            },
        )
        return result

    # ── Batch processing ───────────────────────────────────────────────────

    async def _consolidate_batch(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        user_id: str,
        app_id: str,
        batch: list[Event],
    ) -> tuple[int, int, int]:
        """
        Consolidate one batch of events.
        Returns (events_consolidated, facts_distilled, links_created).
        """
        prompt = _build_consolidation_prompt(batch)

        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_CONSOLIDATION_SCHEMA,
                example_output=_CONSOLIDATION_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Consolidation LLM call failed — batch skipped",
                extra={"user_id": user_id, "batch_size": len(batch), "error": str(exc)},
            )
            return 0, 0, 0

        summary = extracted.get("summary", "")
        fact_dicts = extracted.get("facts", [])
        link_dicts = extracted.get("links", [])

        # Mark events consolidated in Postgres
        event_ids = [e.id for e in batch]
        await self.episodic.mark_consolidated(
            pg_session, event_ids, summary=summary or None
        )

        # Upsert distilled facts to Neo4j
        facts_stored = 0
        for fd in fact_dicts:
            try:
                await self.semantic.upsert_fact(
                    neo_session,
                    user_id=user_id,
                    app_id=app_id,
                    category=fd["category"],
                    key=fd["key"],
                    value=fd["value"],
                    confidence=float(fd.get("confidence", 0.8)),
                )
                facts_stored += 1
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid distilled fact",
                    extra={"fact": fd, "error": str(exc)},
                )

        # Create narrative links between events in this batch
        links_created = 0
        if self.narrative and link_dicts:
            for ld in link_dicts:
                try:
                    from_idx = int(ld["from_index"])
                    to_idx = int(ld["to_index"])
                    rel_type = RelationType(ld["relation_type"])
                    if (
                        0 <= from_idx < len(batch)
                        and 0 <= to_idx < len(batch)
                        and from_idx != to_idx
                    ):
                        await self.narrative.create_link(
                            pg_session,
                            from_event_id=batch[from_idx].id,
                            to_event_id=batch[to_idx].id,
                            relation_type=rel_type,
                        )
                        links_created += 1
                except (KeyError, ValueError) as exc:
                    logger.warning(
                        "Skipping invalid narrative link",
                        extra={"link": ld, "error": str(exc)},
                    )

        return len(batch), facts_stored, links_created


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_batches(events: list[Event], batch_size: int) -> list[list[Event]]:
    """Split events into time-ordered batches of batch_size."""
    return [events[i : i + batch_size] for i in range(0, len(events), batch_size)]


def _build_consolidation_prompt(batch: list[Event]) -> str:
    """Build the LLM prompt for a batch of events."""
    lines = ["Consolidate the following user interactions into a summary and extract facts.\n"]
    lines.append("Interactions:")
    for i, event in enumerate(batch, 1):
        date = _format_date(event.created_at)
        text = (event.summary or event.raw_text)[:300]
        lines.append(f"{i}. [{date}] {text}")
    lines.append(
        "\nReturn JSON with: summary (1-2 sentences) and facts (list of structured facts)."
    )
    return "\n".join(lines)


def _format_date(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")
