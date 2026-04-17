"""
IdentityBuilder — synthesizes a structured user identity from semantic facts.

Mirrors the brain's self-model: an integrated representation of who the user
is, derived from accumulated facts across all semantic categories.

Pipeline:
    SemanticMemory.get_user_profile()
        │
        └─► group facts by category → IdentityDimension[]
                │
                └─► LLMAdapter.extract_structured()  (narrative summary)
                        │
                        └─► UserIdentity  (structured + narrative)

UserIdentity.as_prompt_text() renders a concise identity block suitable
for prepending to any LLM system prompt alongside MemoryContext.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import UserBelief
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.semantic import FactRecord, SemanticMemory

logger = logging.getLogger(__name__)

_IDENTITY_SCHEMA = (
    "summary (string): a 1-2 sentence narrative describing who this user is — "
    "weave in their role, location, skills, interests, hobbies, health context, "
    "values, lifestyle, and any other dimensions present in the facts below. "
    "Be concise but holistic; omit dimensions with no data."
)
_IDENTITY_EXAMPLE = {
    "summary": (
        "This user is a vegetarian entrepreneur based in Mumbai building an AI memory "
        "startup called smritikosh, with deep expertise in LangGraph and RAG, "
        "who values family and meditates daily."
    )
}


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class IdentityDimension:
    """A single category of identity facts (e.g. all 'role' facts)."""

    category: str
    facts: list[FactRecord]
    dominant_value: str    # value of the highest-confidence fact
    confidence: float      # max confidence across facts in this dimension


@dataclass
class UserIdentity:
    """
    Synthesized user identity from all semantic facts.

    Contains structured dimensions (per fact category) and an LLM-generated
    narrative summary suitable for injection into LLM system prompts.
    """

    user_id: str
    app_id: str
    dimensions: list[IdentityDimension] = field(default_factory=list)
    beliefs: list[UserBelief] = field(default_factory=list)
    summary: str = ""
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_facts: int = 0

    def is_empty(self) -> bool:
        return not self.dimensions and not self.beliefs

    def as_prompt_text(self) -> str:
        """Render identity as a string for LLM system prompt injection."""
        if self.is_empty():
            return "## User Identity\n(no identity data available)"

        sections: list[str] = ["## User Identity\n"]

        if self.summary:
            sections.append(f"**Who they are:** {self.summary}\n")

        for dim in self.dimensions:
            values = ", ".join(
                f.value
                for f in sorted(dim.facts, key=lambda f: f.confidence, reverse=True)
            )
            sections.append(f"**{dim.category.title()}:** {values}")

        if self.beliefs:
            sections.append("\n**Core beliefs & values:**")
            for b in sorted(self.beliefs, key=lambda b: b.confidence, reverse=True):
                sections.append(
                    f"- {b.statement}  [confidence: {b.confidence:.2f}]"
                )

        return "\n".join(sections)


# ── IdentityBuilder ────────────────────────────────────────────────────────────


class IdentityBuilder:
    """
    Builds a UserIdentity from accumulated semantic facts in Neo4j.

    Usage:
        builder = IdentityBuilder(llm=llm, semantic=semantic)

        async with neo4j_session() as neo:
            identity = await builder.build(neo, user_id="u1")
            print(identity.as_prompt_text())
    """

    def __init__(
        self,
        llm: LLMAdapter,
        semantic: SemanticMemory,
        *,
        min_confidence: float = 0.5,
    ) -> None:
        self.llm = llm
        self.semantic = semantic
        self.min_confidence = min_confidence

    async def build(
        self,
        neo_session: NeoSession,
        *,
        user_id: str,
        app_id: str = "default",
        pg_session: AsyncSession | None = None,
    ) -> UserIdentity:
        """
        Build a UserIdentity for the given user.

        Steps:
            1. Fetch all semantic facts above min_confidence from Neo4j.
            2. Group into IdentityDimension objects by category.
            3. Generate a narrative summary via LLM (with fallback).
            4. If pg_session provided, load inferred beliefs from user_beliefs.
            5. Return UserIdentity.
        """
        profile = await self.semantic.get_user_profile(
            neo_session, user_id, app_id, min_confidence=self.min_confidence
        )
        facts = profile.facts if profile else []

        dimensions = _build_dimensions(facts)
        summary = await self._generate_summary(user_id, dimensions)

        beliefs: list[UserBelief] = []
        if pg_session is not None:
            try:
                from smritikosh.processing.belief_miner import BeliefMiner
                miner = BeliefMiner(llm=self.llm, semantic=self.semantic)
                beliefs = await miner.get_beliefs(
                    pg_session, user_id, app_id, min_confidence=self.min_confidence
                )
            except Exception as exc:
                logger.warning(
                    "Belief fetch failed during identity build",
                    extra={"user_id": user_id, "error": str(exc)},
                )

        return UserIdentity(
            user_id=user_id,
            app_id=app_id,
            dimensions=dimensions,
            beliefs=beliefs,
            summary=summary,
            total_facts=len(facts),
        )

    async def _generate_summary(
        self, user_id: str, dimensions: list[IdentityDimension]
    ) -> str:
        if not dimensions:
            return ""

        fact_lines = [
            f"{f.category}/{f.key}: {f.value} (confidence={f.confidence:.2f})"
            for dim in dimensions
            for f in dim.facts
        ]
        prompt = (
            "Synthesize a concise 1-2 sentence identity summary for a user "
            "based on these facts:\n" + "\n".join(fact_lines)
        )

        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_IDENTITY_SCHEMA,
                example_output=_IDENTITY_EXAMPLE,
            )
            return extracted.get("summary", "")
        except Exception as exc:
            logger.warning(
                "Identity summary generation failed — using fallback",
                extra={"user_id": user_id, "error": str(exc)},
            )
            return _fallback_summary(dimensions)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_dimensions(facts: list[FactRecord]) -> list[IdentityDimension]:
    """Group facts by category, highest-confidence first within each group."""
    by_category: dict[str, list[FactRecord]] = {}
    for fact in facts:
        by_category.setdefault(fact.category, []).append(fact)

    dimensions = []
    for category, cat_facts in sorted(by_category.items()):
        sorted_facts = sorted(cat_facts, key=lambda f: f.confidence, reverse=True)
        dimensions.append(
            IdentityDimension(
                category=category,
                facts=sorted_facts,
                dominant_value=str(sorted_facts[0].value),
                confidence=sorted_facts[0].confidence,
            )
        )
    return dimensions


def _fallback_summary(dimensions: list[IdentityDimension]) -> str:
    """Build a simple summary without LLM, used when the LLM call fails."""
    if not dimensions:
        return ""
    parts = [f"{dim.category}={dim.dominant_value}" for dim in dimensions]
    return "User profile: " + ", ".join(parts)
