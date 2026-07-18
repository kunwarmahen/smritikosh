"""
MeetingPrepAgent — context-aware meeting briefs (E4, FUTURE.md #3).

Before a meeting: retrieve everything memory holds about the attendees —
who they are, prior interactions, open commitments — plus the user's own
goals, and synthesise a one-page brief with cited evidence.

After the meeting: `debrief()` feeds the user's notes back through the full
Hippocampus encoding pipeline (importance scoring, embedding, fact
extraction), closing FUTURE.md's loop: memory in → agent action → new
memory out.

Design decisions:
    - Retrieval fans out per attendee (one embed + one hybrid search each,
      concurrently) rather than one blended query — a brief about three
      people needs three evidence pools, not one averaged one.
    - One synthesis LLM call regardless of attendee count; per-attendee
      sections come back structured and citations are validated against the
      actually-retrieved events (hallucinated ids dropped).
    - The brief itself is logged as an episodic event
      (source_type=agent_meeting_prep) so the post-meeting debrief — and any
      later retrieval — can see what was prepared.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh import metrics
from smritikosh.db.models import SourceType
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory, SearchResult
from smritikosh.memory.hippocampus import Hippocampus
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

MAX_ATTENDEES = 10
EVENTS_PER_ATTENDEE = 5
TOPIC_EVENTS = 5
RECENT_EVENTS = 10

_PREP_SCHEMA = (
    "attendee_briefs: list of objects, one per listed attendee, with: "
    "name (string, copied exactly from the attendee list), "
    "known_facts (list of strings — what the user's memory says about this "
    "person; empty list if memory holds nothing), "
    "history (list of strings — concrete prior interactions, with dates when "
    "the memory includes them), "
    "open_commitments (list of strings — unresolved promises, asks, or "
    "follow-ups in either direction). "
    "talking_points (list of 2-4 strings, REQUIRED and never empty — what "
    "the user should raise, given their goal and the shared history). "
    "questions_to_ask (list of 1-3 strings, REQUIRED and never empty — gaps "
    "in the user's memory worth filling during the meeting). "
    "watch_outs (list of strings — sensitivities, past friction, or risks; "
    "empty only if memory shows none). "
    "cited_event_ids: list of strings, REQUIRED — ids copied exactly from "
    "the [event <id>] tags of every memory used in this brief."
)

_PREP_EXAMPLE = {
    "attendee_briefs": [
        {
            "name": "Priya",
            "known_facts": ["CTO at Acme; prefers written proposals before calls."],
            "history": ["2026-05-12: demo call — asked about SOC2 status."],
            "open_commitments": ["You promised to send the security whitepaper."],
        }
    ],
    "talking_points": ["Lead with the SOC2 progress since the demo."],
    "questions_to_ask": ["Whether Acme's pilot budget was approved."],
    "watch_outs": ["Priya pushed back on pricing last time — avoid re-anchoring high."],
    "cited_event_ids": ["6f1e..."],
}


@dataclass
class AttendeeBrief:
    name: str
    known_facts: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    open_commitments: list[str] = field(default_factory=list)


@dataclass
class MeetingPrepResult:
    user_id: str
    app_id: str
    attendees: list[str]
    topic: str = ""
    attendee_briefs: list[AttendeeBrief] = field(default_factory=list)
    talking_points: list[str] = field(default_factory=list)
    questions_to_ask: list[str] = field(default_factory=list)
    watch_outs: list[str] = field(default_factory=list)
    cited_event_ids: list[str] = field(default_factory=list)
    memories_considered: int = 0
    logged_event_id: str | None = None
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class DebriefResult:
    user_id: str
    app_id: str
    event_id: str | None = None
    facts_extracted: int = 0
    extraction_failed: bool = False


class MeetingPrepAgent:
    """
    Pre-meeting brief synthesis + post-meeting debrief ingestion.

    Usage:
        agent = MeetingPrepAgent(llm=llm, episodic=episodic,
                                 semantic=semantic, hippocampus=hippocampus)

        async with db_session() as pg, neo4j_session() as neo:
            brief = await agent.prepare(pg, neo, user_id="u1",
                                        attendees=["Priya"], topic="pilot renewal")
            ...
            await agent.debrief(pg, neo, user_id="u1",
                                notes="Priya confirmed the pilot budget ...")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        hippocampus: Hippocampus,
        *,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.hippocampus = hippocampus
        self.audit = audit

    # ── Pre-meeting brief ──────────────────────────────────────────────────

    async def prepare(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        attendees: list[str],
        topic: str = "",
        goal: str = "",
        app_ids: list[str] | None = None,
    ) -> MeetingPrepResult:
        app_id = app_ids[0] if app_ids else "default"
        attendees = [a.strip() for a in attendees if a.strip()][:MAX_ATTENDEES]
        result = MeetingPrepResult(
            user_id=user_id, app_id=app_id, attendees=attendees, topic=topic
        )
        if not attendees:
            result.skipped = True
            result.skip_reason = "No attendees given."
            return result

        # ── 1. Fan out retrieval: one evidence pool per attendee + topic ───
        queries = [f"meeting with {name}" + (f" about {topic}" if topic else "")
                   for name in attendees]
        if topic:
            queries.append(topic)

        pools = await asyncio.gather(
            *(self._search_pool(pg_session, user_id, q, app_ids) for q in queries),
        )
        profile = await self.semantic.get_user_profile(
            neo_session, user_id, app_id, min_confidence=0.5
        )
        recent = await self.episodic.get_recent(
            pg_session, user_id, app_ids or [app_id], limit=RECENT_EVENTS
        )

        # Dedup events across pools, preserving per-attendee grouping for the prompt
        attendee_pools = dict(zip(attendees, pools))
        topic_pool = pools[len(attendees)] if topic else []
        seen: set[str] = set()
        all_events = []
        for pool in [*attendee_pools.values(), topic_pool, [
            SearchResult(event=e, hybrid_score=0.0) for e in recent
        ]]:
            for sr in pool:
                eid = str(sr.event.id)
                if eid not in seen:
                    seen.add(eid)
                    all_events.append(sr.event)
        result.memories_considered = len(all_events)

        if not all_events and not (profile and profile.facts):
            result.skipped = True
            result.skip_reason = (
                "No memory stored for this user — a brief would be generic. "
                "Add memories first."
            )
            metrics.AGENT_RUNS.labels(agent="meeting_prep", outcome="skipped").inc()
            return result

        # ── 2. One synthesis call over the pooled evidence ─────────────────
        prompt = _build_prep_prompt(
            attendees, topic, goal, profile, attendee_pools, topic_pool, recent
        )
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_PREP_SCHEMA,
                example_output=_PREP_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Meeting prep synthesis failed: %s", exc, extra={"user_id": user_id}
            )
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            metrics.AGENT_RUNS.labels(agent="meeting_prep", outcome="error").inc()
            return result

        # Some local models nest the meeting-level lists inside each attendee
        # object instead of at the top level — hoist them so neither shape
        # loses content.
        nested: dict[str, list[str]] = {
            "talking_points": [], "questions_to_ask": [], "watch_outs": [],
            "cited_event_ids": [],
        }
        for ab in extracted.get("attendee_briefs", []):
            try:
                result.attendee_briefs.append(AttendeeBrief(
                    name=str(ab["name"]).strip(),
                    known_facts=[str(x) for x in ab.get("known_facts", []) if str(x).strip()],
                    history=[str(x) for x in ab.get("history", []) if str(x).strip()],
                    open_commitments=[
                        str(x) for x in ab.get("open_commitments", []) if str(x).strip()
                    ],
                ))
                for key, bucket in nested.items():
                    bucket.extend(str(x) for x in ab.get(key, []) if str(x).strip())
            except (KeyError, TypeError, AttributeError):
                continue

        def _merged(key: str) -> list[str]:
            top = [str(x) for x in extracted.get(key, []) if str(x).strip()]
            return top + [x for x in nested[key] if x not in top]

        result.talking_points = _merged("talking_points")
        result.questions_to_ask = _merged("questions_to_ask")
        result.watch_outs = _merged("watch_outs")
        result.cited_event_ids = [c for c in _merged("cited_event_ids") if c in seen]

        # ── 3. Close the loop: the brief becomes memory + audit ────────────
        if result.attendee_briefs or result.talking_points:
            try:
                summary_text = _render_brief_summary(result)
                embedding = None
                try:
                    embedding = await self.llm.embed(summary_text)
                except Exception:
                    logger.warning("Brief embedding failed — storing without one")
                event = await self.episodic.store(
                    pg_session,
                    user_id=user_id,
                    app_id=app_id,
                    raw_text=summary_text,
                    embedding=embedding,
                    importance_score=0.6,
                    source_type=SourceType.AGENT_MEETING_PREP,
                    source_meta={
                        "agent": "meeting_prep",
                        "attendees": attendees,
                        "topic": topic,
                        "cited_event_ids": result.cited_event_ids,
                    },
                )
                result.logged_event_id = str(event.id)
            except Exception:
                logger.exception("Failed to log meeting brief as episodic event")

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.AGENT_MEETING_PREP,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "attendees": attendees,
                    "topic": topic[:200],
                    "briefed_attendees": [b.name for b in result.attendee_briefs],
                    "talking_points_count": len(result.talking_points),
                    "cited_event_ids": result.cited_event_ids,
                    "memories_considered": result.memories_considered,
                    "logged_event_id": result.logged_event_id,
                },
            ))

        metrics.AGENT_RUNS.labels(agent="meeting_prep", outcome="success").inc()
        logger.info(
            "Meeting brief synthesised",
            extra={
                "user_id": user_id,
                "attendees": len(attendees),
                "citations": len(result.cited_event_ids),
                "memories_considered": result.memories_considered,
            },
        )
        return result

    async def _search_pool(
        self,
        pg_session: AsyncSession,
        user_id: str,
        query: str,
        app_ids: list[str] | None,
    ) -> list[SearchResult]:
        """Embed one query and fetch its hybrid-search pool (empty on failure)."""
        try:
            embedding = await self.llm.embed(query)
        except Exception as exc:
            logger.warning("Prep pool embedding failed for %r: %s", query, exc)
            return []
        try:
            return await self.episodic.hybrid_search(
                pg_session, user_id, embedding,
                app_ids=app_ids, top_k=EVENTS_PER_ATTENDEE,
            )
        except Exception as exc:
            logger.warning("Prep pool search failed for %r: %s", query, exc)
            return []

    # ── Post-meeting debrief ───────────────────────────────────────────────

    async def debrief(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        notes: str,
        attendees: list[str] | None = None,
        app_ids: list[str] | None = None,
    ) -> DebriefResult:
        """
        Feed post-meeting notes through the full encoding pipeline — the new
        facts and the episodic record become retrievable memory immediately.
        """
        app_id = app_ids[0] if app_ids else "default"
        encoded = await self.hippocampus.encode(
            pg_session,
            neo_session,
            user_id=user_id,
            raw_text=notes,
            app_id=app_id,
            source_type=SourceType.MEETING_DEBRIEF,
            source_meta={
                "agent": "meeting_prep",
                "attendees": attendees or [],
            },
        )
        metrics.AGENT_RUNS.labels(agent="meeting_debrief", outcome="success").inc()
        return DebriefResult(
            user_id=user_id,
            app_id=app_id,
            event_id=str(encoded.event.id),
            facts_extracted=len(encoded.facts),
            extraction_failed=encoded.extraction_failed,
        )


# ── Prompt / summary builders ─────────────────────────────────────────────────


def _build_prep_prompt(
    attendees: list[str],
    topic: str,
    goal: str,
    profile,
    attendee_pools: dict[str, list[SearchResult]],
    topic_pool: list[SearchResult],
    recent,
) -> str:
    lines = [
        "You are a meeting-preparation assistant. Using ONLY this user's "
        "memory below, produce a brief for their upcoming meeting: what they "
        "know about each attendee, prior interactions, open commitments, "
        "talking points, and questions worth asking. Do not invent facts — "
        "if memory holds nothing about an attendee, say so with empty lists. "
        "Cite the events you rely on by their [event <id>] tags.\n",
        "ATTENDEES: " + ", ".join(attendees),
    ]
    if topic:
        lines.append(f"MEETING TOPIC: {topic}")
    if goal:
        lines.append(f"THE USER'S GOAL FOR THIS MEETING: {goal}")
    lines.append("")

    if profile and profile.facts:
        lines.append("WHO THE USER IS:")
        lines.append(profile.as_text_summary())
        lines.append("")

    for name, pool in attendee_pools.items():
        lines.append(f"MEMORIES RELATED TO {name.upper()}:")
        if pool:
            for sr in pool:
                text = (sr.event.summary or sr.event.raw_text)[:250]
                lines.append(f"  [event {sr.event.id}] {text}")
        else:
            lines.append("  (none found)")
        lines.append("")

    if topic_pool:
        lines.append("MEMORIES RELATED TO THE TOPIC:")
        for sr in topic_pool:
            text = (sr.event.summary or sr.event.raw_text)[:250]
            lines.append(f"  [event {sr.event.id}] {text}")
        lines.append("")

    if recent:
        lines.append("RECENT ACTIVITY (for context):")
        for e in recent:
            date = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
            text = (e.summary or e.raw_text)[:150]
            lines.append(f"  [event {e.id}] [{date}] {text}")
        lines.append("")

    return "\n".join(lines)


def _render_brief_summary(result: MeetingPrepResult) -> str:
    """Compact text form of the brief for the episodic log."""
    lines = [
        "Meeting brief prepared for: " + ", ".join(result.attendees)
        + (f" (topic: {result.topic})" if result.topic else "")
    ]
    for b in result.attendee_briefs:
        if b.open_commitments:
            lines.append(f"Open with {b.name}: " + "; ".join(b.open_commitments))
    if result.talking_points:
        lines.append("Talking points: " + "; ".join(result.talking_points))
    return "\n".join(lines)
