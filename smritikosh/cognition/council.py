"""
CouncilAgent — multi-agent deliberation for high-stakes decisions (E4, FUTURE.md #4).

Where the DecisionAgent runs one synthesis pass, the council convenes four
specialist perspectives over the SAME assembled memory, then a judge
synthesises the deliberation:

    risk             — downside scenarios grounded in the user's history
    values           — alignment with stated beliefs and identity
    pattern          — similar past decisions and how they actually unfolded
    devils_advocate  — the strongest case AGAINST the emerging consensus

Design decisions:
    - Memory is assembled ONCE (ContextBuilder at the COMPLEX tier) and shared
      by all specialists — the council multiplies reasoning passes, not
      retrieval cost. Specialists run concurrently.
    - A failed specialist is dropped, not fatal: the judge deliberates over
      whoever showed up (minimum two opinions, else the run is skipped).
    - Citations are validated per-specialist against the actually-provided
      memories; the final cited set is the union of surviving citations —
      the judge synthesises argument, it does not add new evidence.
    - The deliberation closes the loop like the DecisionAgent: verdict logged
      back as an episodic event (source_type=agent_council) and audited
      (agent.council) with every opinion, so the full reasoning chain is
      visible per FUTURE.md ("User sees the full reasoning chain").
"""

import asyncio
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

MIN_OPINIONS_FOR_VERDICT = 2

_VALID_POSITIONS = {"support", "oppose", "conditional"}

# role → the specialist's charter, injected into its prompt
COUNCIL_ROLES: dict[str, str] = {
    "risk": (
        "You are the RISK specialist. Evaluate downside scenarios for this "
        "decision against the user's actual history: past failures, costs "
        "that proved unrecoverable, and the risk tolerance their logged "
        "behaviour demonstrates. Concrete worst cases only — no generic risk."
    ),
    "values": (
        "You are the VALUES specialist. Judge ONLY whether this decision "
        "aligns with the user's stated beliefs, identity facts, and "
        "priorities. Name each belief it supports or violates. Ignore "
        "practicality — that is another specialist's job."
    ),
    "pattern": (
        "You are the PATTERN specialist. Find the user's most similar past "
        "decisions and situations in the provided memories and report how "
        "they actually unfolded. Argue from precedent: what happened last "
        "time they faced something like this?"
    ),
    "devils_advocate": (
        "You are the DEVIL'S ADVOCATE. Whatever the user seems inclined to "
        "do, build the strongest honest case against it using their own "
        "memories. Attack weak assumptions. If the case against is genuinely "
        "weak, say so — do not manufacture objections."
    ),
}

_OPINION_SCHEMA = (
    "position (string: support|oppose|conditional): your overall stance on "
    "the decision from your specialist perspective. "
    "argument (string): your case in 2-4 sentences, grounded in the provided "
    "memories. "
    "confidence (float 0.0-1.0): how strongly the user's memory supports "
    "your argument. "
    "cited_event_ids: list of strings — ONLY ids copied exactly from the "
    "[event <id>] tags of memories that materially support your argument."
)

_OPINION_EXAMPLE = {
    "position": "conditional",
    "argument": "Your last relocation drained savings for a year; this one only works if the offer covers moving costs.",
    "confidence": 0.7,
    "cited_event_ids": ["6f1e..."],
}

_VERDICT_SCHEMA = (
    "recommendation (string): the recommended course of action in 1-3 "
    "sentences, synthesised from the council's opinions. "
    "reasoning (string): how you weighed the opinions against each other — "
    "which arguments won and why. "
    "confidence (float 0.0-1.0): overall confidence given the agreement or "
    "disagreement between specialists. "
    "dissent (string): the strongest surviving objection to your "
    "recommendation, in one or two sentences (empty string if the council "
    "was unanimous). "
    "open_questions: list of strings — information the user's memory does "
    "NOT contain that would change the verdict."
)

_VERDICT_EXAMPLE = {
    "recommendation": "Decline the offer and counter with a remote arrangement.",
    "reasoning": "The values and pattern specialists agree relocation conflicts with family priorities; the risk case against was stronger than the case for.",
    "confidence": 0.65,
    "dissent": "The pattern specialist notes your last bold move paid off despite similar doubts.",
    "open_questions": ["Whether the employer would accept a remote counter-offer."],
}


@dataclass
class CouncilOpinion:
    role: str
    position: str           # support | oppose | conditional
    argument: str
    confidence: float = 0.5
    cited_event_ids: list[str] = field(default_factory=list)


@dataclass
class CouncilResult:
    user_id: str
    app_id: str
    decision: str
    opinions: list[CouncilOpinion] = field(default_factory=list)
    recommendation: str = ""
    reasoning: str = ""
    confidence: float = 0.0
    dissent: str = ""
    cited_event_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    memories_considered: int = 0
    logged_event_id: str | None = None
    skipped: bool = False
    skip_reason: str = ""


class CouncilAgent:
    """
    Multi-agent deliberation: four specialists + a judge over shared memory.

    Usage:
        council = CouncilAgent(llm=llm, context_builder=builder, episodic=episodic)

        async with db_session() as pg, neo4j_session() as neo:
            result = await council.deliberate(pg, neo, user_id="u1",
                                              decision="Should I raise a seed round?")
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

    async def deliberate(
        self,
        pg_session: AsyncSession,
        neo_session: NeoSession,
        *,
        user_id: str,
        decision: str,
        options: list[str] | None = None,
        app_ids: list[str] | None = None,
    ) -> CouncilResult:
        app_id = app_ids[0] if app_ids else "default"
        result = CouncilResult(user_id=user_id, app_id=app_id, decision=decision)

        # ── 1. Assemble memory ONCE at the COMPLEX tier ────────────────────
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
                "No memory stored for this user — a deliberation would be "
                "generic, not personal. Add memories first."
            )
            metrics.AGENT_RUNS.labels(agent="council", outcome="skipped").inc()
            return result

        provided_ids = {str(sr.event.id) for sr in ctx.similar_events}
        provided_ids.update(str(e.id) for e in ctx.recent_events)

        # ── 2. Specialists deliberate concurrently over the shared memory ──
        memory_block = _render_memory_block(decision, options, ctx)
        raw_opinions = await asyncio.gather(
            *(
                self._specialist_opinion(role, charter, memory_block)
                for role, charter in COUNCIL_ROLES.items()
            ),
            return_exceptions=True,
        )
        for role, raw in zip(COUNCIL_ROLES, raw_opinions):
            if isinstance(raw, BaseException):
                logger.warning(
                    "Council specialist %s failed: %s", role, raw,
                    extra={"user_id": user_id},
                )
                continue
            raw.cited_event_ids = [c for c in raw.cited_event_ids if c in provided_ids]
            result.opinions.append(raw)

        if len(result.opinions) < MIN_OPINIONS_FOR_VERDICT:
            result.skipped = True
            result.skip_reason = (
                f"Only {len(result.opinions)} of {len(COUNCIL_ROLES)} specialists "
                "produced an opinion — not enough for a deliberation."
            )
            metrics.AGENT_RUNS.labels(agent="council", outcome="error").inc()
            return result

        # ── 3. Judge synthesises the deliberation ──────────────────────────
        try:
            verdict = await self.llm.extract_structured(
                prompt=_build_judge_prompt(decision, options, ctx, result.opinions),
                schema_description=_VERDICT_SCHEMA,
                example_output=_VERDICT_EXAMPLE,
            )
        except Exception as exc:
            logger.warning(
                "Council judge synthesis failed: %s", exc, extra={"user_id": user_id}
            )
            result.skipped = True
            result.skip_reason = f"Judge synthesis failed: {exc}"
            metrics.AGENT_RUNS.labels(agent="council", outcome="error").inc()
            return result

        result.recommendation = str(verdict.get("recommendation", "")).strip()
        result.reasoning = str(verdict.get("reasoning", "")).strip()
        result.dissent = str(verdict.get("dissent", "")).strip()
        try:
            result.confidence = max(0.0, min(1.0, float(verdict.get("confidence", 0.5))))
        except (TypeError, ValueError):
            result.confidence = 0.5
        result.open_questions = [
            str(q) for q in verdict.get("open_questions", []) if str(q).strip()
        ]
        # The judge weighs arguments; evidence stays with the specialists.
        cited: list[str] = []
        for op in result.opinions:
            cited.extend(c for c in op.cited_event_ids if c not in cited)
        result.cited_event_ids = cited

        # ── 4. Close the loop: verdict becomes memory + audit ──────────────
        if result.recommendation:
            try:
                summary_text = (
                    f"Council deliberation: {decision}\n"
                    f"Verdict: {result.recommendation}"
                    + (f"\nDissent: {result.dissent}" if result.dissent else "")
                )
                embedding = None
                try:
                    embedding = await self.llm.embed(summary_text)
                except Exception:
                    logger.warning("Council event embedding failed — storing without one")
                event = await self.episodic.store(
                    pg_session,
                    user_id=user_id,
                    app_id=app_id,
                    raw_text=summary_text,
                    embedding=embedding,
                    importance_score=0.8,
                    source_type=SourceType.AGENT_COUNCIL,
                    source_meta={
                        "agent": "council",
                        "confidence": result.confidence,
                        "positions": {op.role: op.position for op in result.opinions},
                        "cited_event_ids": result.cited_event_ids,
                    },
                )
                result.logged_event_id = str(event.id)
            except Exception:
                logger.exception("Failed to log council verdict as episodic event")

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.AGENT_COUNCIL,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "decision_preview": decision[:300],
                    "recommendation": result.recommendation,
                    "confidence": result.confidence,
                    "dissent": result.dissent,
                    "opinions": [
                        {
                            "role": op.role,
                            "position": op.position,
                            "confidence": op.confidence,
                            "argument": op.argument,
                            "cited_event_ids": op.cited_event_ids,
                        }
                        for op in result.opinions
                    ],
                    "cited_event_ids": result.cited_event_ids,
                    "memories_considered": result.memories_considered,
                    "logged_event_id": result.logged_event_id,
                },
            ))

        metrics.AGENT_RUNS.labels(agent="council", outcome="success").inc()
        logger.info(
            "Council deliberation complete",
            extra={
                "user_id": user_id,
                "opinions": len(result.opinions),
                "confidence": result.confidence,
                "citations": len(result.cited_event_ids),
            },
        )
        return result

    async def _specialist_opinion(
        self, role: str, charter: str, memory_block: str
    ) -> CouncilOpinion:
        """One specialist's structured opinion (raises on LLM failure)."""
        extracted = await self.llm.extract_structured(
            prompt=(
                f"{charter}\n\n"
                "Reason ONLY from this user's memory below — do not invent "
                "facts about them. Cite the events you rely on by their "
                "[event <id>] tags.\n\n"
                f"{memory_block}"
            ),
            schema_description=_OPINION_SCHEMA,
            example_output=_OPINION_EXAMPLE,
        )
        position = str(extracted.get("position", "conditional")).lower().strip()
        if position not in _VALID_POSITIONS:
            position = "conditional"
        try:
            confidence = max(0.0, min(1.0, float(extracted.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        return CouncilOpinion(
            role=role,
            position=position,
            argument=str(extracted.get("argument", "")).strip(),
            confidence=confidence,
            cited_event_ids=[str(c) for c in extracted.get("cited_event_ids", [])],
        )


# ── Prompt builders ───────────────────────────────────────────────────────────


def _render_memory_block(decision: str, options: list[str] | None, ctx) -> str:
    """The shared decision + memory block every specialist reasons over."""
    lines = [f"DECISION: {decision}"]
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


def _build_judge_prompt(
    decision: str, options: list[str] | None, ctx, opinions: list[CouncilOpinion]
) -> str:
    """The judge weighs the specialists' opinions — argument, not new evidence."""
    lines = [
        "You are the JUDGE of a deliberation council advising one user. Four "
        "specialists have argued over the user's own memories. Weigh their "
        "arguments against each other and produce a verdict. Take "
        "disagreement seriously: preserve the strongest surviving objection "
        "as dissent rather than papering over it.\n",
        f"DECISION: {decision}",
    ]
    if options:
        lines.append("OPTIONS:")
        lines.extend(f"  {i}. {opt}" for i, opt in enumerate(options, 1))
    lines.append("")

    if ctx.beliefs:
        lines.append("THE USER'S CORE BELIEFS (for reference):")
        for b in ctx.beliefs:
            lines.append(f"  - [{b.category}] {b.statement}")
        lines.append("")

    lines.append("THE COUNCIL'S OPINIONS:")
    for op in opinions:
        lines.append(
            f"  [{op.role} | {op.position} | confidence {op.confidence:.2f}]\n"
            f"    {op.argument}"
        )
    lines.append("")
    return "\n".join(lines)
