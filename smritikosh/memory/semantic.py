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
Design notes:
  - Facts are upserted (MERGE) — duplicate extraction just strengthens confidence.
  - Relationship type is fixed (not parameterised) due to Cypher limitations;
    category → relationship type is mapped via _CATEGORY_TO_REL.
  - All write operations return structured FactRecord objects so callers are
    decoupled from raw Neo4j Record objects.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from neo4j import AsyncSession

from smritikosh.db.models import FactCategory, FactStatus, SourceType


# ── Relationship type mapping ──────────────────────────────────────────────────
# Cypher does not allow parameterised relationship types, so we map explicitly.

_CATEGORY_TO_REL: dict[str, str] = {
    # Identity & demographics
    FactCategory.IDENTITY:     "HAS_IDENTITY",
    FactCategory.LOCATION:     "LIVES_IN",
    # Work & professional
    FactCategory.ROLE:         "HAS_ROLE",
    FactCategory.SKILL:        "HAS_SKILL",
    FactCategory.EDUCATION:    "STUDIED_AT",
    FactCategory.PROJECT:      "WORKS_ON",
    FactCategory.GOAL:         "HAS_GOAL",
    # Personal interests & activities
    FactCategory.INTEREST:     "HAS_INTEREST",
    FactCategory.HOBBY:        "ENJOYS",
    FactCategory.HABIT:        "HAS_HABIT",
    FactCategory.PREFERENCE:   "HAS_PREFERENCE",
    FactCategory.PERSONALITY:  "HAS_TRAIT",
    # Relationships & social
    FactCategory.RELATIONSHIP: "KNOWS",
    FactCategory.PET:          "HAS_PET",
    # Health & wellness
    FactCategory.HEALTH:       "HAS_HEALTH_CONDITION",
    FactCategory.DIET:         "FOLLOWS_DIET",
    # Beliefs & values
    FactCategory.BELIEF:       "BELIEVES",
    FactCategory.VALUE:        "VALUES",
    FactCategory.RELIGION:     "PRACTICES",
    # Lifestyle & context
    FactCategory.FINANCE:      "HAS_FINANCE",
    FactCategory.LIFESTYLE:    "HAS_LIFESTYLE",
    FactCategory.EVENT:        "EXPERIENCED",
    FactCategory.TOOL:         "USES",
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


_WORD = re.compile(r"[a-z0-9]+")

#: Facts the weekly decay never touches: said by the user on the
#: dashboard, or filed from a labelled memory (memory/labels.py).
NEVER_DECAYED = ("ui_manual", "agent_labelled")
#: Kinds of fact that change rarely: a slow half-life.
STABLE_CATEGORIES = (
    FactCategory.IDENTITY, FactCategory.LOCATION, FactCategory.RELATIONSHIP,
    FactCategory.EDUCATION, FactCategory.HEALTH, FactCategory.DIET,
    FactCategory.RELIGION, FactCategory.PET,
)
#: Kinds of fact that are over soon: the base half-life.
TRANSIENT_CATEGORIES = (FactCategory.PROJECT, FactCategory.GOAL, FactCategory.EVENT)


def _same_words(a: str, b: str) -> bool:
    """Whether two values are one fact reworded: two thirds of all their
    words shared ("MBA from Fuqua" / "MBA from Fuqua at Duke"). Half is
    too few -- "Lives in Cary" and "Lives in Austin" share half."""
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= 0.65


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
    source_event_ids: list[str] = field(default_factory=list)
    source_type: str = SourceType.API_EXPLICIT
    source_meta: dict = field(default_factory=dict)
    status: str = FactStatus.ACTIVE


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
        source_event_ids: list[str] | None = None,
        source_type: str = SourceType.API_EXPLICIT,
        source_meta: dict | None = None,
        status: str = FactStatus.ACTIVE,
    ) -> FactRecord:
        """
        Insert or strengthen a fact about a user.

        Uses Cypher MERGE so:
          - First time: creates User node, Fact node, and relationship.
          - Subsequent times: increments frequency_count and updates confidence.

        source_event_ids: episodic event IDs that contributed to this fact.
          New IDs are appended (deduplicated) to the existing list, capped at 50.

        source_type / source_meta: provenance of this fact (which ingestion path).
          On CREATE the values are stored; on MATCH, source_type is preserved from
          the first write unless explicitly overridden by a higher-confidence source.

        Returns the final state of the fact as a FactRecord.
        """
        rel = _rel_type(category)
        now = _now_iso()
        new_ids = source_event_ids or []
        meta = json.dumps(source_meta or {})

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
                r.confidence       = $confidence,
                r.frequency_count  = 1,
                r.first_seen_at    = $now,
                r.last_seen_at     = $now,
                r.source_event_ids = $new_ids,
                r.source_type      = $source_type,
                r.source_meta      = $source_meta,
                r.status           = $status
            ON MATCH SET
                r.confidence       = $confidence,
                r.frequency_count  = r.frequency_count + 1,
                r.last_seen_at     = $now,
                r.source_event_ids = (
                    coalesce(r.source_event_ids, []) +
                    [x IN $new_ids WHERE NOT x IN coalesce(r.source_event_ids, [])]
                )[0..50],
                r.source_meta      = $source_meta

            RETURN f.category AS category, f.key AS key, f.value AS value,
                   r.confidence AS confidence, r.frequency_count AS frequency_count,
                   r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at,
                   r.source_event_ids AS source_event_ids,
                   r.source_type AS source_type, r.source_meta AS source_meta,
                   r.status AS status
            """,
            user_id=user_id,
            app_id=app_id,
            category=category,
            key=key,
            value=value,
            confidence=confidence,
            now=now,
            new_ids=new_ids,
            source_type=source_type,
            source_meta=meta,
            status=status,
        )
        record = await result.single()
        return _record_to_fact(record)

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

    async def set_fact(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        category: str,
        key: str,
        value: str,
        app_id: str = "default",
        confidence: float = 1.0,
        source_event_ids: list[str] | None = None,
        source_type: str = SourceType.API_EXPLICIT,
        source_meta: dict | None = None,
        status: str = FactStatus.ACTIVE,
    ) -> tuple[FactRecord, list[str]]:
        """
        Write a fact as the ONE value of (category, key) for this user.

        ``upsert_fact`` merges on the exact value, so "MBA from Fuqua" and
        "MBA from Fuqua at Duke" became two facts, and a new city sat beside
        the old one. Here any other value under the same key is taken off
        first. A rewording (most words shared) hands its source ids on, so
        forgetting either event still finds the fact; a real change does
        not -- the old value's events no longer back the new one.

        Returns the written fact and the values it replaced.
        """
        rel = _rel_type(category)
        result = await session.run(
            f"""
            MATCH (u:User {{user_id: $user_id, app_id: $app_id}})-[r:{rel}]->(f:Fact)
            WHERE f.category = $category AND f.key = $key AND f.value <> $value
            RETURN f.value AS value, coalesce(r.source_event_ids, []) AS ids
            """,
            user_id=user_id, app_id=app_id, category=category, key=key, value=value,
        )
        others = await result.data()
        carried: list[str] = []
        for other in others:
            if _same_words(other["value"], value):
                carried += [i for i in other["ids"] if i not in carried]
        if others:
            await session.run(
                f"""
                MATCH (u:User {{user_id: $user_id, app_id: $app_id}})-[r:{rel}]->(f:Fact)
                WHERE f.category = $category AND f.key = $key AND f.value <> $value
                DELETE r
                WITH DISTINCT f
                OPTIONAL MATCH (f)--(other)
                WITH f, count(other) AS links
                WHERE links = 0
                DELETE f
                """,
                user_id=user_id, app_id=app_id, category=category, key=key, value=value,
            )
        ids = list(source_event_ids or [])
        ids += [i for i in carried if i not in ids]
        fact = await self.upsert_fact(
            session, user_id=user_id, app_id=app_id, category=category, key=key,
            value=value, confidence=confidence, source_event_ids=ids,
            source_type=source_type, source_meta=source_meta, status=status,
        )
        return fact, [o["value"] for o in others]

    async def unlink_event(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        event_id: str,
        app_id: str = "default",
    ) -> int:
        """
        Take a forgotten event out of the facts drawn from it.

        The event id leaves every source_event_ids list of this user's facts.
        A fact left with no source at all goes too — the only thing it rested
        on is gone, so keeping it would remember what the user asked to forget.
        A fact another event still backs stays. A Fact node no one links to
        any more is removed with its last relationship.

        Facts written with no source ids (a direct POST /memory/fact, or a
        distilled fact whose sources could not be verified) are never touched:
        nothing says which event they came from.

        Returns how many of the user's facts were removed.
        """
        return await self.unlink_events(
            session, user_id=user_id, event_ids=[event_id], app_id=app_id)

    async def unlink_events(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        event_ids: list[str],
        app_id: str = "default",
    ) -> int:
        """``unlink_event`` for many events at once (deleting all of a user's
        memories). Returns how many of the user's facts were removed."""
        if not event_ids:
            return 0
        result = await session.run(
            """
            MATCH (u:User {user_id: $user_id, app_id: $app_id})-[r]->(f:Fact)
            WHERE any(x IN coalesce(r.source_event_ids, []) WHERE x IN $event_ids)
            SET r.source_event_ids = [x IN r.source_event_ids WHERE NOT x IN $event_ids]
            WITH r, f
            WHERE size(r.source_event_ids) = 0
            DELETE r
            WITH DISTINCT f
            OPTIONAL MATCH (f)--(other)
            WITH f, count(other) AS links
            WHERE links = 0
            DELETE f
            """,
            user_id=user_id,
            app_id=app_id,
            event_ids=[str(i) for i in event_ids],
        )
        summary = await result.consume()
        return summary.counters.relationships_deleted

    async def touch_facts(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        keys: list[tuple[str, str]],
        app_id: str = "default",
    ) -> list[str]:
        """
        A client's word that an answer leaned on these facts, given as
        (category, key): last_seen_at = now, so the weekly decay counts
        from today. Returns the source event ids of the facts touched --
        the memories they rest on, which the caller marks used too.
        """
        if not keys:
            return []
        result = await session.run(
            """
            UNWIND $keys AS k
            MATCH (u:User {user_id: $user_id, app_id: $app_id})-[r]->(f:Fact)
            WHERE f.category = k[0] AND f.key = k[1]
            SET r.last_seen_at = $now
            RETURN coalesce(r.source_event_ids, []) AS ids
            """,
            user_id=user_id,
            app_id=app_id,
            keys=[[c, k] for c, k in keys],
            now=_now_iso(),
        )
        ids: list[str] = []
        for record in await result.data():
            ids.extend(str(i) for i in record["ids"])
        return list(dict.fromkeys(ids))

    # ── Read ───────────────────────────────────────────────────────────────

    async def get_facts(
        self,
        session: AsyncSession,
        user_id: str,
        app_id: str = "default",
        category: Optional[str] = None,
        min_confidence: float = 0.0,
        active_only: bool = True,
    ) -> list[FactRecord]:
        """
        Retrieve facts for a user, optionally filtered by category and confidence.

        active_only (default True): exclude pending/rejected facts so they never
        leak into context assembly. Pass False to list all facts for review UIs.
        Results are ordered by frequency_count (most reinforced first).
        """
        status_clause = "AND r.status = 'active'" if active_only else ""
        if category is not None:
            rel = _rel_type(category)
            cypher = f"""
                MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                      -[r:{rel}]->(f:Fact)
                WHERE r.confidence >= $min_confidence {status_clause}
                RETURN f.category AS category, f.key AS key, f.value AS value,
                       r.confidence AS confidence, r.frequency_count AS frequency_count,
                       r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at,
                       r.source_event_ids AS source_event_ids,
                       r.source_type AS source_type, r.source_meta AS source_meta,
                       r.status AS status
                ORDER BY r.frequency_count DESC
            """
        else:
            cypher = f"""
                MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                      -[r]->(f:Fact)
                WHERE r.confidence >= $min_confidence {status_clause}
                RETURN f.category AS category, f.key AS key, f.value AS value,
                       r.confidence AS confidence, r.frequency_count AS frequency_count,
                       r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at,
                       r.source_event_ids AS source_event_ids,
                       r.source_type AS source_type, r.source_meta AS source_meta,
                       r.status AS status
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
        Only active facts are included — pending/rejected facts are excluded.
        """
        facts = await self.get_facts(
            session, user_id, app_id, min_confidence=min_confidence, active_only=True
        )
        return UserProfile(user_id=user_id, app_id=app_id, facts=facts)

    async def check_fact_conflict(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str,
        category: str,
        key: str,
        candidate_value: str,
    ) -> dict | None:
        """
        Check whether a fact with the same (user, app, category, key) already exists
        but with a DIFFERENT value.

        Returns a dict with existing_value and existing_confidence if a conflict is
        found, or None if the key is new or has the same value (no conflict).
        """
        rel = _rel_type(category)
        result = await session.run(
            f"""
            MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                  -[r:{rel}]->(f:Fact)
            WHERE f.category = $category AND f.key = $key AND f.value <> $candidate_value
            RETURN f.value AS existing_value, r.confidence AS existing_confidence
            LIMIT 1
            """,
            user_id=user_id,
            app_id=app_id,
            category=category,
            key=key,
            candidate_value=candidate_value,
        )
        record = await result.single()
        if record is None:
            return None
        return {
            "existing_value": record["existing_value"],
            "existing_confidence": float(record["existing_confidence"] or 0.0),
        }

    async def set_fact_status(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str,
        category: str,
        key: str,
        status: str,
    ) -> "FactRecord | None":
        """
        Set the status on an existing (user, app, category, key) fact relationship.
        Returns the updated FactRecord, or None if no such fact exists.
        """
        rel = _rel_type(category)
        result = await session.run(
            f"""
            MATCH (u:User {{user_id: $user_id, app_id: $app_id}})
                  -[r:{rel}]->(f:Fact)
            WHERE f.category = $category AND f.key = $key
            SET r.status = $status
            RETURN f.category AS category, f.key AS key, f.value AS value,
                   r.confidence AS confidence, r.frequency_count AS frequency_count,
                   r.first_seen_at AS first_seen_at, r.last_seen_at AS last_seen_at,
                   r.source_event_ids AS source_event_ids,
                   r.source_type AS source_type, r.source_meta AS source_meta,
                   r.status AS status
            """,
            user_id=user_id,
            app_id=app_id,
            category=category,
            key=key,
            status=status,
        )
        record = await result.single()
        if record is None:
            return None
        return _record_to_fact(record)

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

    async def decay_stale_facts(
        self,
        session: AsyncSession,
        *,
        decay_half_life_days: float = 60.0,
        confidence_floor: float = 0.1,
        staleness_pending_threshold: float = 0.20,
    ) -> tuple[int, int, int, int]:
        """
        Apply exponential confidence decay to all user→fact relationships.

        Formula: new_confidence = confidence × exp(−ln(2) × age_days / decay_half_life_days)
        This halves confidence every `decay_half_life_days` days without reinforcement.

        age_days is the time since this relationship was last DECAYED or last
        seen, whichever is later -- not since it was last seen. Each pass
        stamps ``decayed_at``. Counting from last_seen_at on every weekly
        pass multiplied the same weeks in again and again. Days are counted
        with ``duration.inDays``: ``duration.between(...).days`` is only the
        days left over after whole months, so 56 days read as 25.

        Source-type rules applied:
          - ui_manual facts are NEVER decayed (user explicitly stated them).
          - agent_labelled facts are NEVER decayed: they live exactly as long
            as the memory they were filed from (memory/labels.py).
          - cross_system facts decay at 2× the normal rate (behavioral patterns shift fast).

        The half-life depends on what the fact is about. Who someone is,
        where they live, their family, schooling, health, diet, faith and
        pets change rarely and are corrected when they do: 6× the base
        (360 days by default). Projects, goals and events are over soon:
        the base (60 days). Everything else: 2× (120 days).

        Four-pass execution:
          1. Decay all eligible relationships (excluding ui_manual).
          2. Move relationships below staleness_pending_threshold to pending status.
          3. Delete relationships below confidence_floor.
          4. DETACH DELETE orphaned Fact nodes no longer connected to any User.

        Returns:
            (decayed_count, pending_promoted_count, deleted_count, orphans_deleted)
        """
        LN2 = 0.6931471805599453

        # Pass 1: decay — skip ui_manual; apply 2× rate for cross_system
        r1 = await session.run(
            """
            MATCH (u:User)-[r]->(f:Fact)
            WHERE r.last_seen_at IS NOT NULL
              AND NOT coalesce(r.source_type, '') IN $never
            WITH r, f,
                 CASE WHEN r.decayed_at IS NOT NULL
                           AND datetime(r.decayed_at) > datetime(r.last_seen_at)
                      THEN r.decayed_at ELSE r.last_seen_at END AS since,
                 CASE WHEN f.category IN $stable THEN $decay_days * 6.0
                      WHEN f.category IN $transient THEN $decay_days
                      ELSE $decay_days * 2.0 END AS by_kind
            WITH r, since,
                 CASE WHEN r.source_type = 'cross_system' THEN by_kind / 2.0
                      ELSE by_kind END AS effective_half_life
            WITH r, effective_half_life,
                 duration.inDays(datetime(since), datetime()).days AS age_days
            SET r.confidence = r.confidence * exp(-$ln2 * toFloat(age_days) / effective_half_life),
                r.decayed_at = toString(datetime())
            RETURN count(r) AS decayed_count
            """,
            ln2=LN2,
            decay_days=float(decay_half_life_days),
            never=list(NEVER_DECAYED),
            stable=[str(c) for c in STABLE_CATEGORIES],
            transient=[str(c) for c in TRANSIENT_CATEGORIES],
        )
        rec1 = await r1.single()
        decayed_count = int(rec1["decayed_count"]) if rec1 else 0

        # Pass 2: move stale active facts to pending (they need review before deletion)
        r2 = await session.run(
            """
            MATCH (u:User)-[r]->(f:Fact)
            WHERE r.confidence < $staleness_threshold
              AND r.status = 'active'
              AND NOT coalesce(r.source_type, '') IN $never
            SET r.status = 'pending'
            RETURN count(r) AS pending_count
            """,
            staleness_threshold=float(staleness_pending_threshold),
            never=list(NEVER_DECAYED),
        )
        rec2 = await r2.single()
        pending_promoted_count = int(rec2["pending_count"]) if rec2 else 0

        # Pass 3: delete relationships below the hard floor
        r3 = await session.run(
            """
            MATCH (u:User)-[r]->(f:Fact)
            WHERE r.confidence < $confidence_floor
              AND NOT coalesce(r.source_type, '') IN $never
            DELETE r
            RETURN count(*) AS deleted_count
            """,
            confidence_floor=float(confidence_floor),
            never=list(NEVER_DECAYED),
        )
        rec3 = await r3.single()
        deleted_count = int(rec3["deleted_count"]) if rec3 else 0

        # Pass 4: remove Fact nodes no longer connected to any User
        r4 = await session.run(
            """
            MATCH (f:Fact)
            WHERE NOT (:User)-[]->(f)
            DETACH DELETE f
            RETURN count(*) AS orphans_deleted
            """
        )
        rec4 = await r4.single()
        orphans_deleted = int(rec4["orphans_deleted"]) if rec4 else 0

        return decayed_count, pending_promoted_count, deleted_count, orphans_deleted

    async def purge_unseen_facts(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        not_seen_since_days: int,
    ) -> int:
        """
        Delete Neo4j facts for a user that have not been reinforced in
        `not_seen_since_days` days.

        Called by SynapticPruner after deleting old episodic events: facts
        that were only reinforced by those events will no longer be
        re-confirmed by future consolidation runs, so we remove them eagerly
        rather than waiting for the weekly confidence-decay cycle.

        Returns the number of relationships deleted (orphaned Fact nodes are
        also cleaned up in a second pass).
        """
        r1 = await session.run(
            """
            MATCH (u:User {id: $user_id})-[r]->(f:Fact)
            WHERE r.last_seen_at IS NOT NULL
              AND duration.inDays(datetime(r.last_seen_at), datetime()).days > $cutoff_days
            DELETE r
            RETURN count(*) AS deleted_count
            """,
            user_id=user_id,
            cutoff_days=int(not_seen_since_days),
        )
        rec = await r1.single()
        deleted = int(rec["deleted_count"]) if rec else 0

        # Clean up any Fact nodes that are now orphaned
        await session.run(
            """
            MATCH (f:Fact)
            WHERE NOT (:User)-[]->(f)
            DETACH DELETE f
            """
        )

        return deleted


# ── Helpers ────────────────────────────────────────────────────────────────────


def _parse_source_meta(raw: object) -> dict:
    """Deserialize source_meta stored as a JSON string in Neo4j."""
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _record_to_fact(record: dict) -> FactRecord:
    """Convert a raw Neo4j record dict to a FactRecord."""
    return FactRecord(
        category=record["category"],
        key=record["key"],
        value=str(record["value"]),
        confidence=float(record["confidence"]),
        frequency_count=int(record["frequency_count"]),
        first_seen_at=str(record["first_seen_at"]),
        last_seen_at=str(record["last_seen_at"]),
        source_event_ids=list(record.get("source_event_ids") or []),
        source_type=str(record.get("source_type") or SourceType.API_EXPLICIT),
        source_meta=_parse_source_meta(record.get("source_meta")),
        status=str(record.get("status") or FactStatus.ACTIVE),
    )
