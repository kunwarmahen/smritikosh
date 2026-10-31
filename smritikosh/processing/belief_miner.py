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

What keeps beliefs few and meaningful, each one a failure seen in real
data (554 beliefs from 87 memories, nearly all at confidence 1.00):

  * NO QUOTA. The model was told to infer "3–7" beliefs every run, so
    every run invented some. Now it may return none, and is told to.
  * EVIDENCE OR NOTHING. Each belief cites the memories that show it (by
    their number in the prompt). Fewer than MIN_EVIDENCE distinct ones,
    and it is dropped; the ones cited -- not every memory in the prompt --
    are its evidence.
  * CONFIDENCE FROM EVIDENCE. The model's own number was stored and came
    back 1.0. Confidence now grows with the distinct memories behind a
    belief (``evidence_confidence``).
  * DUPLICATES BY MEANING. A rewording was a new row, since the unique
    key is the exact text. A new belief that means the same as one
    already held (embedding similarity >= SAME_MEANING) reinforces that
    one instead; one that means the same as a retracted belief is
    dropped.
  * ``tidy`` folds the near-duplicates already stored into one each.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import SYSTEM_SOURCE_TYPES, BeliefStatus, Event, UserBelief
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.semantic import FactRecord, SemanticMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_CONSOLIDATED_EVENTS = 3    # minimum evidence needed to attempt belief mining
MAX_EVENTS_IN_PROMPT = 20      # cap on events sent to LLM to control token use
MAX_FACTS_IN_PROMPT = 15       # cap on facts sent to LLM
MAX_BELIEFS_IN_PROMPT = 30     # beliefs already held, shown so they are not repeated
MIN_EVIDENCE = 3               # distinct memories a belief must cite
SAME_MEANING = 0.88            # cosine similarity at which two beliefs are one

_BELIEF_SCHEMA = (
    "beliefs: list of 0 to 5 objects with: "
    "statement (string, one concise sentence starting with a verb like 'believes', "
    "'values', 'assumes', 'prefers', 'thinks'), "
    "category (worldview|value|attitude|assumption), "
    f"evidence (list of ints): the numbers of at least {MIN_EVIDENCE} different "
    "memory summaries below that each, on their own, show this belief. "
    "An empty list of beliefs is the right answer when nothing new is clearly "
    "shown by several memories."
)
_BELIEF_EXAMPLE = {
    "beliefs": [
        {
            "statement": "believes iterative development beats big-bang launches",
            "category": "value",
            "evidence": [2, 5, 9],
        },
    ]
}


def evidence_confidence(distinct_memories: int) -> float:
    """How sure a belief is, from how many distinct memories show it:
    3 → 0.70, 4 → 0.80, 5 or more → 0.90, never 1.0 -- it is an inference."""
    return round(min(0.9, 0.4 + 0.1 * max(0, distinct_memories)), 2)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0

_VALID_CATEGORIES = {"worldview", "value", "attitude", "assumption"}


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class MiningResult:
    user_id: str
    app_id: str
    beliefs_found: int = 0
    beliefs_upserted: int = 0
    beliefs_reinforced: int = 0    # meant the same as one already held
    beliefs_unsupported: int = 0   # cited fewer than MIN_EVIDENCE memories
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
        rejected_beliefs = await self._get_rejected(pg_session, user_id, app_id)
        rejected_statements = {b.statement.strip().lower() for b in rejected_beliefs}

        # ── 4. Extract beliefs via LLM ────────────────────────────────────
        shown = sorted(existing_beliefs, key=lambda b: (-(b.evidence_count or 0),
                                                        -(b.confidence or 0)))
        prompt = _build_belief_prompt(facts, events, shown[:MAX_BELIEFS_IN_PROMPT],
                                      rejected_beliefs)
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_BELIEF_SCHEMA,
                example_output=_BELIEF_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Belief mining LLM call failed: %s",
                exc,
                extra={"user_id": user_id},
            )
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            return result

        belief_dicts = extracted.get("beliefs", []) or []
        result.beliefs_found = len(belief_dicts)

        # ── 5. Keep the supported ones; a repeat reinforces ───────────────
        # Rejected statements are guarded three ways: verbatim here, by
        # meaning below, and at the SQL layer (the upsert's conflict action
        # carries WHERE status != 'rejected'), so a retracted belief can
        # never be resurrected by re-mining (E2).
        meanings = _Meanings(self.llm)
        now = datetime.now(timezone.utc)
        for bd in belief_dicts:
            try:
                statement = str(bd.get("statement", "")).strip()
                if statement.lower() in rejected_statements:
                    logger.debug(
                        "Skipping re-mined rejected belief",
                        extra={"user_id": user_id, "statement": statement},
                    )
                    continue
                cited = _cited(bd.get("evidence"), events)
                if len(cited) < MIN_EVIDENCE:
                    result.beliefs_unsupported += 1
                    continue
                if statement and await meanings.match(statement, rejected_beliefs):
                    continue
                same = await meanings.match(statement, existing_beliefs) if statement else None
                if same is not None:
                    await _reinforce(pg_session, same, cited, now)
                    result.beliefs_reinforced += 1
                    continue
                bd = {**bd, "confidence": evidence_confidence(len(cited))}
                upserted = await _upsert_belief(pg_session, user_id, app_id, bd, now, cited)
                if upserted:
                    result.beliefs_upserted += 1
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning(
                    "Skipping invalid belief: %s",
                    exc,
                    extra={"belief": bd},
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
        include_rejected: bool = False,
    ) -> list[UserBelief]:
        """Fetch beliefs for a user above the confidence threshold.

        Rejected (user-retracted) beliefs are excluded by default — they must
        not reach identity synthesis, context assembly, or re-mining prompts.
        """
        q = (
            select(UserBelief)
            .where(
                UserBelief.user_id == user_id,
                UserBelief.app_id == app_id,
                UserBelief.confidence >= min_confidence,
            )
            .order_by(UserBelief.confidence.desc())
        )
        if not include_rejected:
            q = q.where(UserBelief.status != BeliefStatus.REJECTED)
        result = await pg_session.execute(q)
        return list(result.scalars().all())

    @staticmethod
    async def _get_rejected(
        pg_session: AsyncSession, user_id: str, app_id: str
    ) -> list[UserBelief]:
        """All user-retracted beliefs, regardless of confidence."""
        result = await pg_session.execute(
            select(UserBelief).where(
                UserBelief.user_id == user_id,
                UserBelief.app_id == app_id,
                UserBelief.status == BeliefStatus.REJECTED,
            )
        )
        return list(result.scalars().all())


# ── Private helpers ───────────────────────────────────────────────────────────


def _cited(evidence, events: list[Event]) -> list[str]:
    """The distinct event ids a belief cites, from its 1-based numbers."""
    ids: list[str] = []
    for n in evidence or []:
        try:
            i = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(events):
            eid = str(events[i - 1].id)
            if eid not in ids:
                ids.append(eid)
    return ids


class _Meanings:
    """Embeddings for one mining run, so each statement is embedded once.
    An embedding that fails turns matching by meaning off for the run --
    exact-text matching (the unique key) still holds."""

    def __init__(self, llm: LLMAdapter) -> None:
        self.llm = llm
        self.cache: dict[str, list[float]] = {}
        self.off = False

    async def vector(self, text_: str) -> list[float] | None:
        if self.off:
            return None
        if text_ not in self.cache:
            try:
                vec = await self.llm.embed(text_)
            except Exception as exc:
                logger.warning("Belief embedding failed, matching by text only: %s", exc)
                self.off = True
                return None
            if not isinstance(vec, list) or not vec:
                self.off = True
                return None
            self.cache[text_] = vec
        return self.cache[text_]

    async def match(self, statement: str, beliefs: list) -> object | None:
        """The belief in ``beliefs`` that means the same, or None."""
        target = await self.vector(statement)
        if target is None:
            return None
        best, best_score = None, SAME_MEANING
        for b in beliefs:
            vec = await self.vector(b.statement)
            if vec is None:
                return None
            score = _cosine(target, vec)
            if score >= best_score:
                best, best_score = b, score
        return best


async def _reinforce(session: AsyncSession, belief: UserBelief, cited: list[str],
                     now: datetime) -> None:
    """One more mining run found this belief: its evidence grows, and its
    confidence with the distinct memories behind it."""
    merged = list(dict.fromkeys([*(belief.evidence_event_ids or []), *cited]))[:50]
    belief.evidence_event_ids = merged
    belief.evidence_count = len(merged)
    belief.confidence = evidence_confidence(len(merged))
    belief.last_updated_at = now
    await session.flush()


async def tidy_beliefs(
    session: AsyncSession, llm: LLMAdapter, *, user_id: str, app_id: str = "default",
) -> int:
    """Fold this user's active beliefs that mean the same into one each.

    The belief with the most evidence keeps; the others' evidence joins it
    and they are deleted. Retracted beliefs are left alone -- they are
    what stops a belief coming back. Returns how many were folded in.
    """
    rows = await session.execute(
        select(UserBelief).where(
            UserBelief.user_id == user_id,
            UserBelief.app_id == app_id,
            UserBelief.status != BeliefStatus.REJECTED,
        )
    )
    beliefs = sorted(rows.scalars().all(),
                     key=lambda b: (-(b.evidence_count or 0), -(b.confidence or 0)))
    meanings = _Meanings(llm)
    kept: list[UserBelief] = []
    folded = 0
    now = datetime.now(timezone.utc)
    for b in beliefs:
        same = await meanings.match(b.statement, kept)
        if meanings.off:
            return folded
        if same is None:
            kept.append(b)
            continue
        merged = list(dict.fromkeys([*(same.evidence_event_ids or []),
                                     *(b.evidence_event_ids or [])]))[:50]
        same.evidence_event_ids = merged
        same.last_updated_at = now
        await session.delete(b)
        folded += 1
    await session.flush()
    return folded


async def _fetch_consolidated_events(
    session: AsyncSession, user_id: str, app_id: str, limit: int
) -> list[Event]:
    """Fetch consolidated events, newest first. A labelled statement has no
    summary (memory/labels.py) and is read as written; the system's own
    findings are not evidence of what the user believes."""
    result = await session.execute(
        select(Event)
        .where(
            Event.user_id == user_id,
            Event.app_id == app_id,
            Event.consolidated.is_(True),
            Event.source_type.not_in(list(SYSTEM_SOURCE_TYPES)),
        )
        .order_by(Event.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


def _build_belief_prompt(
    facts: list[FactRecord],
    events: list[Event],
    existing_beliefs: list | None = None,
    rejected_beliefs: list | None = None,
) -> str:
    """Build the LLM prompt for belief inference."""
    lines = [
        "Analyze this user's known facts and recent memory summaries.\n"
        "Infer at most 5 core beliefs, values, or worldview assumptions that "
        f"several different memories show -- at least {MIN_EVIDENCE} each, cited "
        "by number. A preference or fact already stated plainly is not a "
        "belief. If nothing new is clearly shown, return no beliefs.\n"
        "Each belief must be a concise one-sentence statement starting with "
        "a verb (believes, values, assumes, prefers, thinks).\n"
    ]

    if rejected_beliefs:
        lines.append(
            "REJECTED BELIEFS — the user has explicitly retracted these as wrong. "
            "NEVER return these or any rephrasing of them:"
        )
        for b in rejected_beliefs:
            lines.append(f"  - [{b.category}] {b.statement}")
        lines.append("")

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
    A conflict against a REJECTED row is a no-op (WHERE guard) — a retracted
    belief must never be reactivated or have its confidence bumped by mining.

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
            evidence_count=max(1, len(event_ids)),   # the distinct memories cited
            evidence_event_ids=event_ids,
            first_inferred_at=now,
            last_updated_at=now,
        )
        .on_conflict_do_update(
            constraint="uq_user_belief",
            set_={
                "confidence": confidence,
                # the distinct memories behind it, as _reinforce counts them
                "evidence_count": text(
                    "LEAST(50, (SELECT count(DISTINCT val) "
                    "FROM jsonb_array_elements_text("
                    "  user_beliefs.evidence_event_ids || EXCLUDED.evidence_event_ids"
                    ") val))"
                ),
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
            where=(UserBelief.status != BeliefStatus.REJECTED),
        )
    )
    await session.execute(stmt)
    return True
