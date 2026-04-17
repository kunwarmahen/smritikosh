"""
FactDecayer — applies time-based confidence decay to Neo4j semantic facts.

Mirrors biological memory: knowledge that isn't reinforced fades over time.
Facts a user no longer mentions become less confident; if they never resurface,
they're eventually discarded, keeping the knowledge graph up-to-date.

Decay formula (applied to every User→Fact relationship):
    new_confidence = confidence × exp(−ln(2) × age_days / half_life_days)

This halves confidence every `half_life_days` without reinforcement.
Any relationship that falls below `confidence_floor` is deleted. Fact nodes
left with no User relationships are also deleted (orphan cleanup).

Job cadence: weekly (run by MemoryScheduler).
"""

import logging
from dataclasses import dataclass, field

from smritikosh.config import settings
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class DecayResult:
    decayed_count: int = 0       # relationships whose confidence was reduced
    deleted_count: int = 0       # relationships deleted (fell below floor)
    orphans_deleted: int = 0     # orphaned Fact nodes removed
    skipped: bool = False
    skip_reason: str = ""


# ── FactDecayer ───────────────────────────────────────────────────────────────

class FactDecayer:
    """
    Applies exponential confidence decay to all semantic facts in Neo4j.

    Unlike the other processors, decay is global (all users / all facts),
    so there is no per-user loop — the Cypher queries operate graph-wide.

    Usage:
        decayer = FactDecayer(semantic=SemanticMemory())
        async with neo4j_session() as session:
            result = await decayer.run(session)
    """

    def __init__(
        self,
        semantic: SemanticMemory,
        half_life_days: float | None = None,
        confidence_floor: float | None = None,
    ) -> None:
        self.semantic = semantic
        self.half_life_days = half_life_days if half_life_days is not None \
            else settings.fact_decay_half_life_days
        self.confidence_floor = confidence_floor if confidence_floor is not None \
            else settings.fact_decay_floor

    async def run(self, session) -> DecayResult:
        """
        Execute one full decay cycle across all facts in the graph.

        Steps:
          1. Apply exponential decay to all User→Fact relationship confidences.
          2. Delete relationships below `confidence_floor`.
          3. Remove orphaned Fact nodes.

        Returns a DecayResult summarising what was changed.
        """
        try:
            decayed, deleted, orphans = await self.semantic.decay_stale_facts(
                session,
                decay_half_life_days=self.half_life_days,
                confidence_floor=self.confidence_floor,
            )
        except Exception as exc:
            logger.error("Fact decay failed: %s", exc, exc_info=True)
            result = DecayResult(skipped=True)
            result.skip_reason = str(exc)
            return result

        result = DecayResult(
            decayed_count=decayed,
            deleted_count=deleted,
            orphans_deleted=orphans,
        )
        logger.info(
            "Fact decay complete",
            extra={
                "decayed": decayed,
                "deleted": deleted,
                "orphans_deleted": orphans,
                "half_life_days": self.half_life_days,
                "confidence_floor": self.confidence_floor,
            },
        )
        return result
