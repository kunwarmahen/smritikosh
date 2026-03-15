"""
SemanticMemory — identity and knowledge graph backed by Neo4j.

Mirrors the semantic memory function in the human brain: durable, structured
facts about a person that survive individual conversations.

Graph schema:
    Nodes:
        (:User {user_id, app_id, created_at})
        (:Fact {category, key, value, updated_at})

    Relationships (type derived from fact category):
        (:User)-[:HAS_PREFERENCE  {confidence, frequency_count, first_seen_at, last_seen_at}]->(:Fact)
        (:User)-[:HAS_INTEREST    {…}]->(:Fact)
        (:User)-[:HAS_ROLE        {…}]->(:Fact)
        (:User)-[:WORKS_ON        {…}]->(:Fact)   ← projects
        (:User)-[:HAS_SKILL       {…}]->(:Fact)
        (:User)-[:HAS_GOAL        {…}]->(:Fact)
        (:User)-[:KNOWS           {…}]->(:Fact)   ← relationships
        (:Fact)-[:RELATED_TO      {strength}]->(:Fact)

Design notes:
  - Facts are upserted (MERGE) — duplicate extraction just strengthens confidence.
  - Relationship type is fixed (not parameterised) due to Cypher limitations;
    category → relationship type is mapped via _CATEGORY_TO_REL.
  - All write operations return structured FactRecord objects so callers are
    decoupled from raw Neo4j Record objects.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from neo4j import AsyncSession

from smritikosh.db.models import FactCategory


# ── Relationship type mapping ──────────────────────────────────────────────────
# Cypher does not allow parameterised relationship types, so we map explicitly.

_CATEGORY_TO_REL: dict[str, str] = {
    FactCategory.PREFERENCE: "HAS_PREFERENCE",
    FactCategory.INTEREST: "HAS_INTEREST",
    FactCategory.ROLE: "HAS_ROLE",
    FactCategory.PROJECT: "WORKS_ON",
    FactCategory.SKILL: "HAS_SKILL",
    FactCategory.GOAL: "HAS_GOAL",
    FactCategory.RELATIONSHIP: "KNOWS",
}


def _rel_type(category: str) -> str:
    """Return the Cypher relationship type for a given fact category."""
    rel = _CATEGORY_TO_REL.get(category)
    if rel is None:
        raise ValueError(
            f"Unknown fact category: {category!r}. "
            f"Valid categories: {list(_CATEGORY_TO_REL)}"
        )
    return rel


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Data objects ───────────────────────────────────────────────────────────────


@dataclass
class FactRecord:
    """A single fact about a user, as returned from the graph."""
    category: str
    key: str
    value: str
    confidence: float
    frequency_count: int
    first_seen_at: str
    last_seen_at: str


@dataclass
class UserProfile:
    """
    Full structured knowledge about a user, grouped by category.

    Example:
        profile.by_category()
        → {
            "preference": [FactRecord(key="ui_color", value="green"), ...],
            "interest":   [FactRecord(key="domain",   value="AI agents"), ...],
          }
    """
    user_id: str
    app_id: str
    facts: list[FactRecord] = field(default_factory=list)

    def by_category(self) -> dict[str, list[FactRecord]]:
        result: dict[str, list[FactRecord]] = {}
        for f in self.facts:
            result.setdefault(f.category, []).append(f)
        return result

    def as_text_summary(self) -> str:
        """
        Render the profile as a human-readable string for LLM prompt injection.

        Example output:
            Preferences: ui_color=green
            Interests: domain=AI agents, topic=LLM infrastructure
            Role: current=entrepreneur
        """
        lines = []
        for category, facts in self.by_category().items():
            items = ", ".join(f"{f.key}={f.value}" for f in facts)
            lines.append(f"{category.capitalize()}: {items}")
        return "\n".join(lines) if lines else "(no facts stored)"


# ── SemanticMemory ─────────────────────────────────────────────────────────────


class SemanticMemory:
    """
    Neo4j-backed identity graph for structured user knowledge.

    Each method accepts an AsyncSession (injected by callers) so transaction
    boundaries stay outside this class — consistent with EpisodicMemory.

    Usage:
        async with neo4j_session() as session:
            await semantic.upsert_fact(session, user_id="u1", ...)
            profile = await semantic.get_user_profile(session, "u1")
    """

    # ── Write ──────────────────────────────────────────────────────────────

    async def upsert_fact(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        category: str,
        key: str,
        value: str,
        app_id: str = "default",
        confidence: float = 1.0,
    ) -> FactRecord:
        """
        Insert or strengthen a fact about a user.

        Uses Cypher MERGE so:
          - First time: creates User node, Fact node, and relationship.
          - Subsequent times: increments frequency_count and updates confidence.

        Returns the final state of the fact as a FactRecord.
        """
        rel = _rel_type(category)
        now = _now_iso()

        # Note: relationship type must be string-interpolated (Cypher limitation).
        # The value is safe — it comes from _CATEGORY_TO_REL, not user input.
        result = await session.run(
            f"""
            MERGE (u:User {{user_id: $user_id, app_id: $app_id}})
            ON CREATE SET u.created_at = $now

            MERGE (f:Fact {{category: $category, key: $key, value: $value}})
            ON CREATE SET f.created_at = $now
            SET f.updated_at = $now

            MERGE (u)-[r:{rel}]->(f)
            ON CREATE SET
                r.confidence     = $confidence,
                r.frequency_count = 1,
                r.first_seen_at  = $now,
                r.last_seen_at   = $now
            ON MATCH SET
                r.confidence     = $confidence,
                r.frequency_count = r.frequency_count + 1,
                r.last_seen_at   = $now

            RETURN f.category AS category, f.key AS key, f.value AS value,
                   r.confidence AS confidence, r.frequency_count AS frequency_count,
                   r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at
            """,
            user_id=user_id,
            app_id=app_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            now=now,
        )
        record = await result.single()
        return _record_to_fact(record)

    async def relate_facts(
        self,
        session: AsyncSession,
        *,
        from_category: str,
        from_key: str,
        from_value: str,
        to_category: str,
        to_key: str,
        to_value: str,
        strength: float = 1.0,
    ) -> None:
        """
        Create a RELATED_TO link between two Fact nodes.
        Useful when the Consolidator discovers that two facts are connected.
        """
        await session.run(
            """
            MATCH (f1:Fact {category: $fc1, key: $fk1, value: $fv1})
            MATCH (f2:Fact {category: $fc2, key: $fk2, value: $fv2})
            MERGE (f1)-[r:RELATED_TO]->(f2)
            SET r.strength = $strength, r.updated_at = $now
            """,
            fc1=from_category, fk1=from_key, fv1=from_value,
            fc2=to_category,   fk2=to_key,   fv2=to_value,
            strength=strength,
            now=_now_iso(),
        )

    async def delete_fact(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        category: str,
        key: str,
        app_id: str = "default",
    ) -> bool:
        """
        Remove all relationships from this user to facts matching (category, key).
        Returns True if at least one relationship was deleted.
        """
        rel = _rel_type(category)
        result = await session.run(
            f"""
            MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                  -[r:{rel}]->(f:Fact {{category: $category, key: $key}})
            DELETE r
            RETURN count(r) AS deleted_count
            """,
            user_id=user_id,
            app_id=app_id,
            category=category,
            key=key,
        )
        record = await result.single()
        return bool(record and record["deleted_count"] > 0)

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_facts(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        category: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> list[FactRecord]:
        """
        Retrieve facts for a user, optionally filtered by category and confidence.
        Results are ordered by frequency_count (most reinforced first).
        """
        if category is not None:
            rel = _rel_type(category)
            cypher = f"""
                MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                      -[r:{rel}]->(f:Fact)
                WHERE r.confidence >= $min_confidence
                RETURN f.category AS category, f.key AS key, f.value AS value,
                       r.confidence AS confidence, r.frequency_count AS frequency_count,
                       r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at
                ORDER BY r.frequency_count DESC
            """
        else:
            cypher = """
                MATCH (u:User {user_id: $user_id, app_id: $app_id})
                      -[r]->(f:Fact)
                WHERE r.confidence >= $min_confidence
                RETURN f.category AS category, f.key AS key, f.value AS value,
                       r.confidence AS confidence, r.frequency_count AS frequency_count,
                       r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at
                ORDER BY r.frequency_count DESC
            """

        result = await session.run(
            cypher,
            user_id=user_id,
            app_id=app_id,
            min_confidence=min_confidence,
        )
        records = await result.data()
        return [_record_to_fact(r) for r in records]

    async def get_user_profile(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        min_confidence: float = 0.0,
    ) -> UserProfile:
        """
        Return the full structured profile for a user as a UserProfile object.
        This is what the ContextBuilder uses to inject identity context into prompts.
        """
        facts = await self.get_facts(
            session, user_id, app_id, min_confidence=min_confidence
        )
        return UserProfile(user_id=user_id, app_id=app_id, facts=facts)

    async def user_exists(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
    ) -> bool:
        """Check whether a User node exists in the graph."""
        result = await session.run(
            "MATCH (u:User {user_id: $user_id, app_id: $app_id}) RETURN count(u) AS n",
            user_id=user_id,
            app_id=app_id,
        )
        record = await result.single()
        return bool(record and record["n"] > 0)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _record_to_fact(record: dict) -> FactRecord:
    """Convert a raw Neo4j record dict to a FactRecord."""
    return FactRecord(
        category=record["category"],
        key=record["key"],
        value=record["value"],
        confidence=float(record["confidence"]),
        frequency_count=int(record["frequency_count"]),
        first_seen_at=str(record["first_seen_at"]),
        last_seen_at=str(record["last_seen_at"]),
    )
