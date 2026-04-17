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
import re
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
    "summary (string): 1-2 sentence overview of the entire batch. "
    "event_summaries (list of strings): one concise sentence per interaction, "
    "in the same order as the input list — must have exactly the same length. "
    "facts: list of objects with: "
    "category — must be exactly one of: "
    "identity (name/age/gender/nationality/languages), "
    "location (city/country/timezone/places), "
    "role (job title/profession/career stage), "
    "skill (professional or personal ability/expertise), "
    "education (degrees/schools/certifications), "
    "project (active work initiative), "
    "goal (aspiration/target/deadline), "
    "interest (topic/domain they follow or are curious about), "
    "hobby (active leisure pursuit — sport/art/game), "
    "habit (routine/recurring behaviour/ritual), "
    "preference (stated taste — food/aesthetic/UI/style), "
    "personality (self-described trait/tendency), "
    "relationship (family member/friend/partner/colleague), "
    "pet (animal companion), "
    "health (medical condition/medication/allergy/disability), "
    "diet (dietary restriction/food allergy/eating pattern), "
    "belief (opinion/worldview/philosophical or political stance), "
    "value (core principle/priority/ethic), "
    "religion (spiritual or religious affiliation/practice), "
    "finance (financial situation/budget constraint/money goal), "
    "lifestyle (overall life pattern — nomadic/minimalist/urban), "
    "event (life milestone/anniversary/upcoming appointment), "
    "tool (software/app/platform/tech stack used). "
    "key (short snake_case label), value (concise string), confidence (0.0–1.0), "
    "source_indices (list of 0-based ints): ONLY the indices of interactions where "
    "this fact is explicitly stated or unmistakably implied in that specific text. "
    "Do NOT include an index just because the interaction is in the same batch. "
    "Most facts should cite exactly 1 source; cite multiple only when each "
    "interaction independently and directly mentions the same fact. "
    "links: optional list of objects with: "
    "from_index (0-based int matching the interaction number), "
    "to_index (0-based int), "
    "relation_type (caused|preceded|related|contradicts). "
    "Only include clear, durable facts and unambiguous causal or temporal relationships."
)

_CONSOLIDATION_EXAMPLE = {
    "summary": "User is building an AI memory startup called smritikosh, prefers green UI, and is vegetarian.",
    "event_summaries": [
        "User introduced smritikosh, an AI memory startup.",
        "User mentioned a preference for green UI colours.",
        "User mentioned they are vegetarian.",
    ],
    "facts": [
        {"category": "project",    "key": "active",      "value": "smritikosh", "confidence": 0.95, "source_indices": [0]},
        {"category": "preference", "key": "ui_color",    "value": "green",      "confidence": 0.9,  "source_indices": [1]},
        {"category": "diet",       "key": "restriction", "value": "vegetarian", "confidence": 0.95, "source_indices": [2]},
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
    embeddings_updated: int = 0    # batches where summary was re-embedded
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
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.narrative = narrative
        self.batch_size = batch_size
        self.min_events = min_events
        self.audit = audit

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
            pg_session, user_id, app_ids=[app_id], limit=self.batch_size * 10
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
            consolidated, facts, links, batch_summary, batch_fact_list, embedding_updated = await self._consolidate_batch(
                pg_session, neo_session, user_id, app_id, batch
            )
            result.events_consolidated += consolidated
            result.facts_distilled += facts
            result.links_created += links
            result.embeddings_updated += 1 if embedding_updated else 0

            if self.audit and consolidated:
                from smritikosh.audit.logger import AuditEvent, EventType
                await self.audit.emit(AuditEvent(
                    event_type=EventType.MEMORY_CONSOLIDATED,
                    user_id=user_id,
                    app_id=app_id,
                    payload={
                        "events_in_batch": len(batch),
                        "event_ids": [str(e.id) for e in batch],
                        "summary": batch_summary,
                        "facts_distilled": facts,
                        "links_created": links,
                        "facts": batch_fact_list,
                        "summary_embedding_updated": embedding_updated,
                    },
                ))

        logger.info(
            "Consolidation complete",
            extra={
                "user_id": user_id,
                "events_processed": result.events_processed,
                "events_consolidated": result.events_consolidated,
                "facts_distilled": result.facts_distilled,
                "links_created": result.links_created,
                "batches": result.batches,
                "embeddings_updated": result.embeddings_updated,
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
    ) -> tuple[int, int, int, str, list, bool]:
        """
        Consolidate one batch of events.
        Returns (events_consolidated, facts_distilled, links_created, summary, fact_list, embedding_updated).
        """
        profile = await self.semantic.get_user_profile(
            neo_session, user_id, app_id, min_confidence=0.5
        )
        existing_facts = (profile.facts if profile else [])[:20]

        prompt = _build_consolidation_prompt(batch, existing_facts)

        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_CONSOLIDATION_SCHEMA,
                example_output=_CONSOLIDATION_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Consolidation LLM call failed — batch skipped: %s",
                exc,
                extra={"user_id": user_id, "batch_size": len(batch)},
            )
            return 0, 0, 0, "", [], False

        summary = extracted.get("summary", "")
        event_summaries: list[str] = extracted.get("event_summaries", [])
        fact_dicts = extracted.get("facts", [])
        link_dicts = extracted.get("links", [])

        # Mark all events consolidated; write per-event summaries individually
        # so each event keeps a summary of its own content, not the whole batch.
        event_ids = [e.id for e in batch]
        await self.episodic.mark_consolidated(pg_session, event_ids)
        for i, event in enumerate(batch):
            per_summary = event_summaries[i] if i < len(event_summaries) else None
            if per_summary:
                await self.episodic.update_summary(pg_session, event.id, per_summary)

        # Re-embed the consolidated summary so hybrid search uses the clean,
        # distilled signal rather than the original noisy raw_text embeddings.
        # Only the first (oldest) event in the batch is updated — it acts as the
        # canonical representative; other events retain their original embeddings
        # for result diversity in hybrid search.
        embedding_updated = False
        if summary:
            try:
                summary_embedding = await self.llm.embed(summary)
                await self.episodic.update_embedding(
                    pg_session, batch[0].id, summary_embedding
                )
                embedding_updated = True
                logger.debug(
                    "Summary embedding updated",
                    extra={"user_id": user_id, "event_id": str(batch[0].id)},
                )
            except Exception as exc:
                logger.warning(
                    "Summary re-embedding failed — original embedding retained: %s",
                    exc,
                    extra={"user_id": user_id, "event_id": str(batch[0].id)},
                )

        # Upsert distilled facts to Neo4j — link each fact only to the specific
        # events that mention it, using source_indices from the LLM response.
        batch_event_ids = [str(e.id) for e in batch]
        facts_stored = 0
        for fd in fact_dicts:
            try:
                raw_indices = [
                    i for i in (fd.get("source_indices") or [])
                    if isinstance(i, int) and 0 <= i < len(batch)
                ]
                verified_indices = _filter_source_indices(
                    fd.get("key", ""), fd.get("value", ""), raw_indices, batch
                )
                fact_source_ids = [batch_event_ids[i] for i in verified_indices]
                await self.semantic.upsert_fact(
                    neo_session,
                    user_id=user_id,
                    app_id=app_id,
                    category=fd["category"],
                    key=fd["key"],
                    value=fd["value"],
                    confidence=float(fd.get("confidence", 0.8)),
                    source_event_ids=fact_source_ids,
                )
                facts_stored += 1
            except (KeyError, ValueError) as exc:
                logger.warning(
                    "Skipping invalid distilled fact: %s",
                    exc,
                    extra={"fact": fd},
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

        return len(batch), facts_stored, links_created, summary, fact_dicts, embedding_updated


# ── Helpers ───────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "my", "me", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "their", "to", "of", "in", "at", "for", "on", "with", "by",
    "and", "or", "but", "not", "so", "if", "as", "this", "that", "from",
    "have", "has", "had", "do", "did", "does", "will", "would", "could",
    "should", "may", "might", "very", "just", "also", "more", "than",
})


def _keywords(text: str) -> frozenset[str]:
    return frozenset(
        w.lower() for w in re.split(r'\W+', text)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    )


def _filter_source_indices(
    fact_key: str,
    fact_value: str,
    raw_indices: list[int],
    batch: list[Event],
) -> list[int]:
    """
    Verify LLM-provided source indices by checking keyword overlap between
    the fact and each cited event's text. Prevents over-attribution when the
    LLM lists batch-mates that don't actually mention the fact.

    Falls back to the single best-scoring event if nothing passes, rather
    than silently accepting the full unverified list.
    """
    fact_kw = _keywords(f"{fact_key} {fact_value}")
    if not fact_kw or not raw_indices:
        return raw_indices

    scores: list[tuple[int, int]] = []
    for i in raw_indices:
        if 0 <= i < len(batch):
            event_text = (batch[i].raw_text or "") + " " + (batch[i].summary or "")
            overlap = len(fact_kw & _keywords(event_text))
            scores.append((i, overlap))

    verified = [i for i, score in scores if score > 0]
    if verified:
        return verified

    # No event matched — return the best-scoring candidate rather than all
    if scores:
        logger.debug(
            "Source index verification: no keyword overlap found, using best-scoring event",
            extra={"fact_key": fact_key, "fact_value": fact_value},
        )
        return [max(scores, key=lambda x: x[1])[0]]
    return []


def _split_batches(events: list[Event], batch_size: int) -> list[list[Event]]:
    """Split events into time-ordered batches of batch_size."""
    return [events[i : i + batch_size] for i in range(0, len(events), batch_size)]


def _build_consolidation_prompt(
    batch: list[Event], existing_facts: list[FactRecord] | None = None
) -> str:
    """Build the LLM prompt for a batch of events."""
    lines = ["Consolidate the following user interactions into a summary and extract facts.\n"]
    lines.append("Interactions:")
    for i, event in enumerate(batch, 1):
        date = _format_date(event.created_at)
        text = (event.summary or event.raw_text)[:300]
        lines.append(f"{i}. [{date}] {text}")

    if existing_facts:
        lines.append(
            "\nAlready known facts — REUSE the exact category+key for the same concept. "
            "Only create a new key when the concept is genuinely not covered below:"
        )
        for f in existing_facts:
            lines.append(f"  - {f.category}/{f.key}: {f.value}")

    lines.append(
        f"\nReturn JSON with: summary (1-2 sentence overview of all {len(batch)} interactions), "
        f"event_summaries (list of exactly {len(batch)} concise one-sentence summaries, "
        "one per interaction in order), and facts (list of structured facts)."
    )
    return "\n".join(lines)


def _format_date(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d")
