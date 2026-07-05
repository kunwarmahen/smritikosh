"""
ReflectionAgent — periodic self-consistency cycles (E4, FUTURE.md #9).

Runs on the scheduler (default daily). For each active user it asks one
question: "is this user's stated identity consistent with their recent
behaviour?" — comparing goals/facts/beliefs against the recent event stream
and surfacing:

    drift          — a stated goal with no supporting recent activity
    contradiction  — recent behaviour that conflicts with a belief or fact
    stale_belief   — a belief nothing recent supports any more
    observation    — a notable pattern that fits none of the above

Insights persist in the `reflections` table (dashboard/API surface, with an
acknowledge flag so nudges don't repeat forever), and each cycle logs a
summary of its own reasoning as an episodic event (source_type=
agent_reflection) — future cycles see their predecessors' conclusions, per
FUTURE.md's "logs its own reasoning" note.

Cost control: one LLM call per user per cycle, gated on a minimum number of
recent events (REFLECTION_MIN_EVENTS) so quiet users cost nothing.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh import metrics
from smritikosh.db.models import (
    BeliefStatus,
    Reflection,
    ReflectionKind,
    SourceType,
    UserBelief,
)
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

MAX_EVENTS_IN_PROMPT = 30
MAX_FACTS_IN_PROMPT = 20
MAX_BELIEFS_IN_PROMPT = 10
MAX_INSIGHTS_PER_CYCLE = 5

_VALID_KINDS = {k.value for k in ReflectionKind}
_VALID_SEVERITIES = {"info", "notice", "warning"}

_REFLECTION_SCHEMA = (
    "insights: list of objects (0 to 5 items — return an empty list when the "
    "user's identity and behaviour are consistent; do NOT invent problems) with: "
    "kind (drift|contradiction|stale_belief|observation), "
    "insight (string: one or two sentences, concrete and specific, addressed "
    "to the user as 'you'), "
    "severity (info|notice|warning), "
    "evidence_event_ids (list of strings copied exactly from the [event <id>] "
    "tags of the events that support this insight)."
)

_REFLECTION_EXAMPLE = {
    "insights": [
        {
            "kind": "drift",
            "insight": "You stated a goal of launching in Q2, but no project-related events have been logged in 18 days.",
            "severity": "notice",
            "evidence_event_ids": ["6f1e..."],
        }
    ]
}


@dataclass
class ReflectionResult:
    user_id: str
    app_id: str
    insights_found: int = 0
    insights_stored: int = 0
    skipped: bool = False
    skip_reason: str = ""


class ReflectionAgent:
    """
    Periodic identity-vs-behaviour consistency checker.

    Usage:
        agent = ReflectionAgent(llm=llm, semantic=semantic, episodic=episodic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await agent.reflect(pg, neo, user_id="u1")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        semantic: SemanticMemory,
        episodic: EpisodicMemory,
        *,
        min_events: int = 5,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.semantic = semantic
        self.episodic = episodic
        self.min_events = min_events
        self.audit = audit

    async def reflect(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> ReflectionResult:
        """Run one reflection cycle for a single user."""
        result = ReflectionResult(user_id=user_id, app_id=app_id)

        # ── 1. Recent behaviour (excluding this agent's own past summaries
        # from the guard count, so reflection can't keep itself "active") ──
        recent = await self.episodic.get_recent(
            pg_session, user_id, [app_id], limit=MAX_EVENTS_IN_PROMPT
        )
        substantive = [e for e in recent if e.source_type != SourceType.AGENT_REFLECTION]
        if len(substantive) < self.min_events:
            result.skipped = True
            result.skip_reason = (
                f"Only {len(substantive)} recent events — need at least {self.min_events}."
            )
            return result

        # ── 2. Stated identity: facts, beliefs, open insights ──────────────
        profile = await self.semantic.get_user_profile(
            neo_session, user_id, app_id, min_confidence=0.5
        )
        facts = (profile.facts if profile else [])[:MAX_FACTS_IN_PROMPT]

        belief_rows = await pg_session.execute(
            select(UserBelief)
            .where(
                UserBelief.user_id == user_id,
                UserBelief.app_id == app_id,
                UserBelief.status != BeliefStatus.REJECTED,
            )
            .order_by(UserBelief.confidence.desc())
            .limit(MAX_BELIEFS_IN_PROMPT)
        )
        beliefs = list(belief_rows.scalars().all())

        open_rows = await pg_session.execute(
            select(Reflection)
            .where(
                Reflection.user_id == user_id,
                Reflection.app_id == app_id,
                Reflection.acknowledged.is_(False),
            )
            .order_by(Reflection.created_at.desc())
            .limit(10)
        )
        open_insights = list(open_rows.scalars().all())

        # ── 3. One LLM pass ────────────────────────────────────────────────
        prompt = _build_reflection_prompt(facts, beliefs, recent, open_insights)
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_REFLECTION_SCHEMA,
                example_output=_REFLECTION_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Reflection LLM call failed: %s", exc, extra={"user_id": user_id}
            )
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            metrics.AGENT_RUNS.labels(agent="reflection", outcome="error").inc()
            return result

        insight_dicts = extracted.get("insights", [])[:MAX_INSIGHTS_PER_CYCLE]
        result.insights_found = len(insight_dicts)

        # ── 4. Persist insights (validated) ────────────────────────────────
        valid_event_ids = {str(e.id) for e in recent}
        stored: list[Reflection] = []
        for d in insight_dicts:
            try:
                insight_text = str(d["insight"]).strip()
                if not insight_text:
                    continue
                kind = str(d.get("kind", "")).lower().strip()
                if kind not in _VALID_KINDS:
                    kind = ReflectionKind.OBSERVATION
                severity = str(d.get("severity", "info")).lower().strip()
                if severity not in _VALID_SEVERITIES:
                    severity = "info"
                evidence_ids = [
                    str(i) for i in d.get("evidence_event_ids", [])
                    if str(i) in valid_event_ids
                ]
                row = Reflection(
                    user_id=user_id,
                    app_id=app_id,
                    kind=kind,
                    insight=insight_text,
                    severity=severity,
                    evidence={"event_ids": evidence_ids},
                )
                pg_session.add(row)
                stored.append(row)
            except (KeyError, TypeError, AttributeError) as exc:
                logger.warning("Skipping invalid reflection insight: %s", exc, extra={"insight": d})
        await pg_session.flush()
        result.insights_stored = len(stored)

        # ── 5. Log the cycle's reasoning as memory (FUTURE.md note) ────────
        if stored:
            try:
                summary = "Reflection cycle findings:\n" + "\n".join(
                    f"- [{r.kind}/{r.severity}] {r.insight}" for r in stored
                )
                await self.episodic.store(
                    pg_session,
                    user_id=user_id,
                    app_id=app_id,
                    raw_text=summary,
                    importance_score=0.5,
                    source_type=SourceType.AGENT_REFLECTION,
                    source_meta={
                        "agent": "reflection",
                        "reflection_ids": [str(r.id) for r in stored],
                    },
                )
            except Exception:
                logger.exception("Failed to log reflection cycle as episodic event")

        if self.audit and stored:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.AGENT_REFLECTION,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "insights_found": result.insights_found,
                    "insights_stored": result.insights_stored,
                    "insights": [
                        {"kind": r.kind, "severity": r.severity, "insight": r.insight}
                        for r in stored
                    ],
                },
            ))

        metrics.AGENT_RUNS.labels(agent="reflection", outcome="success").inc()
        logger.info(
            "Reflection cycle complete",
            extra={
                "user_id": user_id,
                "insights_found": result.insights_found,
                "insights_stored": result.insights_stored,
            },
        )
        return result

    # ── Read / acknowledge ─────────────────────────────────────────────────

    async def list_reflections(
        self,
        pg_session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> list[Reflection]:
        q = (
            select(Reflection)
            .where(Reflection.user_id == user_id, Reflection.app_id == app_id)
            .order_by(Reflection.created_at.desc())
            .limit(limit)
        )
        if not include_acknowledged:
            q = q.where(Reflection.acknowledged.is_(False))
        result = await pg_session.execute(q)
        return list(result.scalars().all())

    async def acknowledge(
        self,
        pg_session: AsyncSession,
        user_id: str,
        reflection_id,
    ) -> bool:
        """Mark an insight acknowledged (stops it re-surfacing as a nudge)."""
        row = await pg_session.get(Reflection, reflection_id)
        if row is None or row.user_id != user_id:
            return False
        row.acknowledged = True
        return True


# ── Prompt builder ────────────────────────────────────────────────────────────


def _build_reflection_prompt(facts, beliefs, events, open_insights) -> str:
    now = datetime.now(timezone.utc)
    lines = [
        "You are a reflection agent for a personal memory system. Compare this "
        "user's STATED identity (facts, goals, beliefs) against their RECENT "
        "BEHAVIOUR (logged events) and report inconsistencies: goals with no "
        "recent supporting activity (drift), behaviour conflicting with stated "
        "beliefs (contradiction), beliefs nothing recent supports (stale_belief), "
        "or other notable patterns (observation).\n"
        "Be conservative: only report what the evidence clearly shows. If "
        "identity and behaviour are consistent, return an empty insights list.\n",
        f"TODAY: {now.strftime('%Y-%m-%d')}",
        "",
    ]

    if open_insights:
        lines.append(
            "ALREADY SURFACED (unacknowledged) — do NOT repeat these; only add "
            "genuinely new insights or material changes:"
        )
        for r in open_insights:
            lines.append(f"  - [{r.kind}] {r.insight}")
        lines.append("")

    if facts:
        lines.append("STATED IDENTITY (facts, goals):")
        for f in facts:
            lines.append(f"  - {f.category}/{f.key}: {f.value}")
        lines.append("")

    if beliefs:
        lines.append("INFERRED BELIEFS:")
        for b in beliefs:
            lines.append(f"  - [{b.category}] {b.statement} (confidence {b.confidence:.2f})")
        lines.append("")

    if events:
        lines.append("RECENT BEHAVIOUR (newest first):")
        for e in events:
            date = e.created_at.strftime("%Y-%m-%d") if e.created_at else "unknown"
            text = (e.summary or e.raw_text)[:200]
            lines.append(f"  [event {e.id}] [{date}] {text}")

    return "\n".join(lines)
