"""
BeliefMiner — infers durable beliefs and values from accumulated memory.

Mirrors the brain's capacity for abstract belief formation: after enough
experiences accumulate, the mind distills recurring patterns into stable
worldview assumptions and core values.

Unlike SemanticMemory facts (directly extracted from statements), beliefs
are second-order inferences drawn by the LLM from the *pattern* of events
and facts — what the user consistently chooses, values, and assumes.

Pipeline:
    EpisodicMemory (consolidated events with summaries)  ┐
    SemanticMemory (structured facts from Neo4j)          ├── context
                                                          ┘
        │
        └─► LLMAdapter.extract_structured()   (belief inference)
                │
                └─► upsert into user_beliefs table
                        (reinforces evidence_count on repeat inference)

Run: periodically via the Scheduler (e.g. every 12 hours).
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event, UserBelief
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.semantic import FactRecord, SemanticMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CONSOLIDATED_EVENTS = 3    # minimum evidence needed to attempt belief mining
MAX_EVENTS_IN_PROMPT = 20      # cap on events sent to LLM to control token use
MAX_FACTS_IN_PROMPT = 15       # cap on facts sent to LLM

_BELIEF_SCHEMA = (
    "beliefs: list of objects with: "
    "statement (string, one concise sentence starting with a verb like 'believes', "
    "'values', 'assumes', 'prefers', 'thinks'), "
    "category (worldview|value|attitude|assumption), "
    "confidence (float 0.0–1.0)"
)
_BELIEF_EXAMPLE = {
    "beliefs": [
        {
            "statement": "believes iterative development beats big-bang launches",
            "category": "value",
            "confidence": 0.85,
        },
        {
            "statement": "assumes AI will transform knowledge work within 5 years",
            "category": "worldview",
            "confidence": 0.9,
        },
    ]
}

_VALID_CATEGORIES = {"worldview", "value", "attitude", "assumption"}


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class MiningResult:
    user_id: str
    app_id: str
    beliefs_found: int = 0
    beliefs_upserted: int = 0
    skipped: bool = False
    skip_reason: str = ""


# ── BeliefMiner ───────────────────────────────────────────────────────────────


class BeliefMiner:
    """
    Infers and persists user beliefs from consolidated episodic events and facts.

    Usage:
        miner = BeliefMiner(llm=llm, semantic=semantic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await miner.mine(pg, neo, user_id="u1")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        semantic: SemanticMemory,
        *,
        min_events: int = MIN_CONSOLIDATED_EVENTS,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.semantic = semantic
        self.min_events = min_events
        self.audit = audit

    # ── Primary entry point ────────────────────────────────────────────────

    async def mine(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> MiningResult:
        """
        Run one belief-mining cycle for a single user.

        Steps:
            1. Fetch consolidated events with summaries from Postgres.
            2. Guard: skip if fewer than min_events.
            3. Fetch semantic facts from Neo4j.
            4. Build LLM prompt and extract belief list.
            5. Upsert each valid belief to user_beliefs.
        """
        result = MiningResult(user_id=user_id, app_id=app_id)

        # ── 1. Fetch consolidated events with summaries ───────────────────
        events = await _fetch_consolidated_events(
            pg_session, user_id, app_id, limit=MAX_EVENTS_IN_PROMPT
        )

        # ── 2. Guard ──────────────────────────────────────────────────────
        if len(events) < self.min_events:
            result.skipped = True
            result.skip_reason = (
                f"Only {len(events)} consolidated events — "
                f"need at least {self.min_events}."
            )
            logger.debug(
                "Belief mining skipped",
                extra={"user_id": user_id, "reason": result.skip_reason},
            )
            return result

        # ── 3. Fetch semantic facts and existing beliefs ──────────────────
        profile = await self.semantic.get_user_profile(
            neo_session, user_id, app_id, min_confidence=0.5
        )
        facts = (profile.facts if profile else [])[:MAX_FACTS_IN_PROMPT]

        existing_beliefs = await self.get_beliefs(pg_session, user_id, app_id)

        # Capture event IDs before the LLM call — these become the evidence
        # sources stored alongside each belief for provenance and auditability.
        event_ids = [str(e.id) for e in events]

        # ── 4. Extract beliefs via LLM ────────────────────────────────────
        prompt = _build_belief_prompt(facts, events, existing_beliefs)
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_BELIEF_SCHEMA,
                example_output=_BELIEF_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Belief mining LLM call failed",
                extra={"user_id": user_id, "error": str(exc)},
            )
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            return result

        belief_dicts = extracted.get("beliefs", [])
        result.beliefs_found = len(belief_dicts)

        # ── 5. Upsert valid beliefs ────────────────────────────────────────
        now = datetime.now(timezone.utc)
        for bd in belief_dicts:
            try:
                upserted = await _upsert_belief(pg_session, user_id, app_id, bd, now, event_ids)
                if upserted:
                    result.beliefs_upserted += 1
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping invalid belief",
                    extra={"belief": bd, "error": str(exc)},
                )

        logger.info(
            "Belief mining complete",
            extra={
                "user_id": user_id,
                "beliefs_found": result.beliefs_found,
                "beliefs_upserted": result.beliefs_upserted,
            },
        )

        if self.audit and result.beliefs_upserted:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.BELIEF_MINED,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "beliefs_found": result.beliefs_found,
                    "beliefs_upserted": result.beliefs_upserted,
                    "beliefs": [
                        {
                            "statement": bd.get("statement", ""),
                            "category": bd.get("category", ""),
                            "confidence": bd.get("confidence", 0.0),
                        }
                        for bd in belief_dicts
                    ],
                },
            ))

        return result

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_beliefs(
        self,
        pg_session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        min_confidence: float = 0.5,
    ) -> list[UserBelief]:
        """Fetch all beliefs for a user above the confidence threshold."""
        result = await pg_session.execute(
            select(UserBelief)
            .where(
                UserBelief.user_id == user_id,
                UserBelief.app_id == app_id,
                UserBelief.confidence >= min_confidence,
            )
            .order_by(UserBelief.confidence.desc())
        )
        return list(result.scalars().all())


# ── Private helpers ───────────────────────────────────────────────────────────


async def _fetch_consolidated_events(
    session: AsyncSession, user_id: str, app_id: str, limit: int
) -> list[Event]:
    """Fetch consolidated events that have summaries, newest first."""
    result = await session.execute(
        select(Event)
        .where(
            Event.user_id == user_id,
            Event.app_id == app_id,
            Event.consolidated.is_(True),
            Event.summary.is_not(None),
        )
        .order_by(Event.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def _build_belief_prompt(
    facts: list[FactRecord],
    events: list[Event],
    existing_beliefs: list | None = None,
) -> str:
    """Build the LLM prompt for belief inference."""
    lines = [
        "Analyze this user's known facts and recent memory summaries.\n"
        "Infer 3–7 core beliefs, values, or worldview assumptions that "
        "consistently appear across their experiences.\n"
        "Each belief must be a concise one-sentence statement starting with "
        "a verb (believes, values, assumes, prefers, thinks).\n"
    ]

    if existing_beliefs:
        lines.append(
            "ALREADY RECORDED BELIEFS — do NOT repeat or rephrase these. "
            "Only return beliefs that are genuinely new or meaningfully different:"
        )
        for b in existing_beliefs:
            lines.append(f"  - [{b.category}] {b.statement}")
        lines.append("")

    if facts:
        lines.append("Known facts:")
        for f in facts:
            lines.append(
                f"  - {f.category}/{f.key}: {f.value} "
                f"(confidence={f.confidence:.2f})"
            )
        lines.append("")

    if events:
        lines.append("Recent memory summaries:")
        for i, e in enumerate(events, 1):
            text = (e.summary or e.raw_text)[:150]
            lines.append(f"  {i}. {text}")

    return "\n".join(lines)


async def _upsert_belief(
    session: AsyncSession,
    user_id: str,
    app_id: str,
    bd: dict,
    now: datetime,
    event_ids: list[str],
) -> bool:
    """
    Upsert one belief dict into user_beliefs.

    On INSERT: stores event_ids as evidence sources.
    On CONFLICT: merges old and new event IDs (deduped, capped at 50) so the
    full provenance trail grows across mining cycles without unbounded growth.

    Returns True if the upsert was executed, False if validation failed.
    """
    statement = str(bd["statement"]).strip()
    category = str(bd["category"]).strip().lower()
    confidence = float(bd["confidence"])

    if not statement:
        return False
    if category not in _VALID_CATEGORIES:
        category = "assumption"   # safe fallback for unexpected values
    confidence = max(0.0, min(1.0, confidence))

    stmt = (
        pg_insert(UserBelief)
        .values(
            id=uuid.uuid4(),
            user_id=user_id,
            app_id=app_id,
            statement=statement,
            category=category,
            confidence=confidence,
            evidence_count=1,
            evidence_event_ids=event_ids,
            first_inferred_at=now,
            last_updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_user_belief",
            set_={
                "confidence": confidence,
                "evidence_count": UserBelief.evidence_count + 1,
                # Merge existing IDs with new IDs: concatenate, deduplicate,
                # cap at 50 to prevent unbounded growth over many mining cycles.
                "evidence_event_ids": text(
                    "(SELECT jsonb_agg(DISTINCT val) "
                    "FROM ("
                    "  SELECT val "
                    "  FROM jsonb_array_elements_text("
                    "    user_beliefs.evidence_event_ids || EXCLUDED.evidence_event_ids"
                    "  ) val "
                    "  LIMIT 50"
                    ") sub)"
                ),
                "last_updated_at": now,
            },
        )
    )
    await session.execute(stmt)
    return True
