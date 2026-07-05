"""
DecisionAgent — memory-grounded decision recommendations (E4, FUTURE.md #1).

The user describes a decision; the agent assembles their memory at the
COMPLEX meta-cognition tier (similar events, identity facts, narrative
chains, core beliefs), then one structured LLM call reasons over each
dimension and produces a recommendation with **cited memory**:

    - belief_alignment: which of the user's beliefs each option supports or
      conflicts with (the Values dimension)
    - risks: downside scenarios grounded in past events (the Risk dimension)
    - cited_event_ids / cited_beliefs: the exact memory records the reasoning
      used — every recommendation is auditable

Two closing-the-loop behaviours from FUTURE.md's implementation notes:
    - the decision + recommendation is itself logged as an episodic event
      (source_type=agent_decision), so future retrieval and reflection can
      learn from past decisions;
    - an audit event (agent.decision) records the full reasoning payload.

The single-synthesis design is deliberate: the full Multi-Agent Deliberation
Council (FUTURE.md #4 — separate risk/values/devil's-advocate agents) layers
on top of this later; the API contract already carries its dimensions.
"""

import logging
from dataclasses import dataclass, field

from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh import metrics
from smritikosh.db.models import SourceType
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.retrieval.context_builder import ContextBuilder
from smritikosh.retrieval.intent_classifier import ComplexityTier

logger = logging.getLogger(__name__)

_DECISION_SCHEMA = (
    "recommendation (string): the recommended course of action in 1-3 sentences. "
    "reasoning (string): the core argument, grounded in the user's memory. "
    "confidence (float 0.0-1.0): how strongly the user's memory supports this. "
    "belief_alignment: list of objects with: belief (string, one of the user's "
    "listed beliefs verbatim), alignment (supports|conflicts|neutral), "
    "note (string, one sentence). "
    "risks: list of strings, each one concrete downside scenario. "
    "cited_event_ids: list of strings — ONLY ids copied exactly from the "
    "[event <id>] tags in the provided memories that materially informed the "
    "recommendation. "
    "open_questions: list of strings — information the user's memory does NOT "
    "contain that would change the recommendation."
)

_DECISION_EXAMPLE = {
    "recommendation": "Take the offer, but negotiate a 3-month remote trial first.",
    "reasoning": "Your logged priorities put autonomy above salary, and past role changes went well when you kept an exit ramp.",
    "confidence": 0.72,
    "belief_alignment": [
        {
            "belief": "values family above career",
            "alignment": "conflicts",
            "note": "The role requires relocation away from family.",
        }
    ],
    "risks": ["Relocation cost is unrecoverable if the role ends within a year."],
    "cited_event_ids": ["6f1e...", "a2b3..."],
    "open_questions": ["Whether the team allows permanent remote work."],
}


@dataclass
class BeliefAlignment:
    belief: str
    alignment: str          # supports | conflicts | neutral
    note: str = ""


@dataclass
class DecisionResult:
    user_id: str
    app_id: str
    decision: str
    recommendation: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    belief_alignment: list[BeliefAlignment] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    cited_event_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    memories_considered: int = 0
    logged_event_id: str | None = None
    skipped: bool = False
    skip_reason: str = ""


class DecisionAgent:
    """
    Orchestrates memory retrieval + one structured synthesis for a decision.

    Usage:
        agent = DecisionAgent(llm=llm, context_builder=builder, episodic=episodic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await agent.decide(pg, neo, user_id="u1",
                                        decision="Should I take the Berlin offer?")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        context_builder: ContextBuilder,
        episodic: EpisodicMemory,
        *,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.context_builder = context_builder
        self.episodic = episodic
        self.audit = audit

    async def decide(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        decision: str,
        options: list[str] | None = None,
        app_ids: list[str] | None = None,
    ) -> DecisionResult:
        app_id = app_ids[0] if app_ids else "default"
        result = DecisionResult(user_id=user_id, app_id=app_id, decision=decision)

        # ── 1. Assemble memory at the COMPLEX tier (chains + beliefs) ──────
        ctx = await self.context_builder.build(
            pg_session,
            neo_session,
            user_id=user_id,
            query=decision,
            app_ids=app_ids,
            complexity_override=ComplexityTier.COMPLEX,
        )
        result.memories_considered = ctx.total_memories()

        if ctx.is_empty():
            result.skipped = True
            result.skip_reason = (
                "No memory stored for this user — a recommendation would be "
                "generic, not personal. Add memories first."
            )
            metrics.AGENT_RUNS.labels(agent="decision", outcome="skipped").inc()
            return result

        # ── 2. One structured synthesis over the assembled memory ─────────
        prompt = _build_decision_prompt(decision, options, ctx)
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_DECISION_SCHEMA,
                example_output=_DECISION_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Decision synthesis LLM call failed: %s", exc,
                extra={"user_id": user_id},
            )
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            metrics.AGENT_RUNS.labels(agent="decision", outcome="error").inc()
            return result

        result.recommendation = str(extracted.get("recommendation", "")).strip()
        result.reasoning = str(extracted.get("reasoning", "")).strip()
        try:
            result.confidence = max(0.0, min(1.0, float(extracted.get("confidence", 0.5))))
        except (TypeError, ValueError):
            result.confidence = 0.5
        result.risks = [str(r) for r in extracted.get("risks", []) if str(r).strip()]
        result.open_questions = [
            str(q) for q in extracted.get("open_questions", []) if str(q).strip()
        ]
        for ba in extracted.get("belief_alignment", []):
            try:
                alignment = str(ba.get("alignment", "neutral")).lower()
                if alignment not in ("supports", "conflicts", "neutral"):
                    alignment = "neutral"
                result.belief_alignment.append(BeliefAlignment(
                    belief=str(ba["belief"]),
                    alignment=alignment,
                    note=str(ba.get("note", "")),
                ))
            except (KeyError, TypeError, AttributeError):
                continue

        # Citations must be auditable: keep only ids that were actually in
        # the provided memory (the LLM may hallucinate or truncate ids).
        provided_ids = {str(sr.event.id) for sr in ctx.similar_events}
        provided_ids.update(str(e.id) for e in ctx.recent_events)
        result.cited_event_ids = [
            str(c) for c in extracted.get("cited_event_ids", []) if str(c) in provided_ids
        ]

        # ── 3. Close the loop: the decision becomes memory + audit ─────────
        if result.recommendation:
            try:
                summary_text = (
                    f"Decision considered: {decision}\n"
                    f"Agent recommendation: {result.recommendation}"
                )
                embedding = None
                try:
                    embedding = await self.llm.embed(summary_text)
                except Exception:
                    logger.warning("Decision event embedding failed — storing without one")
                event = await self.episodic.store(
                    pg_session,
                    user_id=user_id,
                    app_id=app_id,
                    raw_text=summary_text,
                    embedding=embedding,
                    importance_score=0.8,
                    source_type=SourceType.AGENT_DECISION,
                    source_meta={
                        "agent": "decision",
                        "confidence": result.confidence,
                        "cited_event_ids": result.cited_event_ids,
                    },
                )
                result.logged_event_id = str(event.id)
            except Exception:
                logger.exception("Failed to log decision as episodic event")

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.AGENT_DECISION,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "decision_preview": decision[:300],
                    "recommendation": result.recommendation,
                    "confidence": result.confidence,
                    "cited_event_ids": result.cited_event_ids,
                    "belief_alignment": [
                        {"belief": ba.belief, "alignment": ba.alignment}
                        for ba in result.belief_alignment
                    ],
                    "risks_count": len(result.risks),
                    "memories_considered": result.memories_considered,
                    "logged_event_id": result.logged_event_id,
                },
            ))

        metrics.AGENT_RUNS.labels(agent="decision", outcome="success").inc()
        logger.info(
            "Decision synthesised",
            extra={
                "user_id": user_id,
                "confidence": result.confidence,
                "citations": len(result.cited_event_ids),
                "memories_considered": result.memories_considered,
            },
        )
        return result


# ── Prompt builder ────────────────────────────────────────────────────────────


def _build_decision_prompt(decision: str, options: list[str] | None, ctx) -> str:
    """Render the decision + assembled memory (with citable event id tags)."""
    lines = [
        "You are a personal decision advisor. Reason ONLY from this user's "
        "memory below — do not invent facts about them. Weigh the decision "
        "against their beliefs and values, surface concrete risks from their "
        "history, and cite the events you rely on by their [event <id>] tags.\n",
        f"DECISION: {decision}",
    ]
    if options:
        lines.append("OPTIONS:")
        lines.extend(f"  {i}. {opt}" for i, opt in enumerate(options, 1))
    lines.append("")

    if ctx.user_profile and ctx.user_profile.facts:
        lines.append("WHO THIS USER IS:")
        lines.append(ctx.user_profile.as_text_summary())
        lines.append("")

    if ctx.beliefs:
        lines.append("CORE BELIEFS & VALUES:")
        for b in ctx.beliefs:
            lines.append(f"  - [{b.category}] {b.statement} (confidence {b.confidence:.2f})")
        lines.append("")

    if ctx.similar_events:
        lines.append("RELEVANT PAST MEMORIES:")
        for sr in ctx.similar_events:
            text = (sr.event.summary or sr.event.raw_text)[:250]
            lines.append(f"  [event {sr.event.id}] {text}")
        lines.append("")

    if ctx.recent_events:
        lines.append("RECENT ACTIVITY:")
        for e in ctx.recent_events:
            text = (e.summary or e.raw_text)[:200]
            lines.append(f"  [event {e.id}] {text}")
        lines.append("")

    if ctx.narrative_chains:
        lines.append("HOW PAST EVENTS UNFOLDED (causal chains):")
        for chain in ctx.narrative_chains:
            parts = [(e.summary or e.raw_text)[:80] for e in chain]
            lines.append("  " + " → ".join(parts))
        lines.append("")

    return "\n".join(lines)
