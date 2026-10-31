"""
Profile — what a client's prompt is given about a user, in one call.

A client that keeps its own statements (an agent that decides what to
remember) still wants what Smritikosh made of them: facts, one current
value per thing, and beliefs, the patterns across many statements.
``build`` hands over all three, as the user's memory view allows
(memory/views.py):

    facts     CORE ones always -- who they are, where they live, what
              they prefer -- and the ones RELEVANT to the question: a fact
              whose memories the question's search found. An account
              number or a street address rides only when it is relevant.
    beliefs   the user's active beliefs, most confident first, each one
              a guess the client should say is one
    memories  the statements no fact covers, relevant first, then the
              most recently used or written -- once a fact says "lives in
              Cary", the memory it came from would only say it twice

A FACT'S ID NAMES ITS PLACE: ``fact:<category>/<key>``, and a belief's
is ``belief:<id>``. A client that leans on one hands the id back to
POST /memory/used like a memory's, and the fact's own memories stop
ageing with it.

SHOWN IS COUNTED, NOT TRUSTED. Every memory a profile puts in a prompt
(and every memory behind a shown fact) gets ``recall_count + 1``: how
often it was shown, for the dashboard. Whether that keeps it is the
view's ``keep_by``: under "shown" a profile marks what it shows as used;
under "used" (the default) only the client's word does.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import BeliefStatus, FactCategory, UserBelief
from smritikosh.memory import views

logger = logging.getLogger(__name__)

#: Categories always handed over: who they are and what they prefer.
CORE_CATEGORIES = (FactCategory.IDENTITY, FactCategory.PREFERENCE)
#: Location keys that say where they live (a street address is not one:
#: it rides when the question needs it).
CORE_LOCATION_KEYS = ("city", "home", "timezone", "time_zone", "country", "region",
                      "airport", "neighbourhood", "neighborhood")
#: How many memories the question's search reads to find relevant facts.
SEARCHED = 20
FACT_LIMIT = 40
BELIEF_LIMIT = 8
MEMORY_LIMIT = 20

FACT_PREFIX = "fact:"
BELIEF_PREFIX = "belief:"


def fact_id(category: str, key: str) -> str:
    return f"{FACT_PREFIX}{category}/{key}"


def parse_ids(ids: list[str]) -> tuple[list[uuid.UUID], list[tuple[str, str]], list[uuid.UUID]]:
    """(memory ids, (category, key) of facts, belief ids) from what a
    client handed back. Anything else is skipped."""
    events, facts, beliefs = [], [], []
    for raw in ids:
        raw = str(raw).strip()
        if raw.startswith(FACT_PREFIX):
            category, _, key = raw[len(FACT_PREFIX):].partition("/")
            if category and key:
                facts.append((category, key))
            continue
        target = beliefs if raw.startswith(BELIEF_PREFIX) else events
        try:
            target.append(uuid.UUID(raw.removeprefix(BELIEF_PREFIX)))
        except ValueError:
            continue
    return events, facts, beliefs


def is_core(category: str, key: str) -> bool:
    if "." in key:              # someone else's fact (naina.email), never core
        return False
    if category in CORE_CATEGORIES:
        return True
    return category == FactCategory.LOCATION and key in CORE_LOCATION_KEYS


def _memory(event) -> dict:
    return {"id": str(event.id), "statement": event.raw_text,
            "created_at": event.created_at.isoformat() if event.created_at else "",
            "last_used_at": event.last_used_at.isoformat() if event.last_used_at else None,
            "metadata": event.event_metadata or {}}


async def build(pg: AsyncSession, neo, *, episodic, semantic, user_id: str,
                app_id: str = "default", app_ids: list[str] | None = None,
                query: str = "", embedding: list[float] | None = None) -> dict:
    """The profile for a prompt. Neo4j down or the query not embedded
    costs the facts or the relevance, never the call."""
    view, _ = await views.effective(pg, user_id, app_id)
    found = []
    if embedding is not None:
        found = [r.event for r in await episodic.hybrid_search(
            pg, user_id, embedding, app_ids=app_ids, top_k=SEARCHED, own_only=True)]
    found_ids = [str(e.id) for e in found]

    all_facts = []
    if neo is not None:
        try:
            all_facts = await semantic.get_facts(neo, user_id, app_id)
        except Exception as exc:
            logger.warning("Profile without facts (graph unavailable): %s", exc)
    covered = {i for f in all_facts for i in f.source_event_ids}

    facts: list[dict] = []
    if view.show_facts:
        rank = {i: n for n, i in enumerate(found_ids)}

        def relevance(fact) -> int:
            return min((rank[i] for i in fact.source_event_ids if i in rank), default=-1)

        core = [f for f in all_facts if is_core(f.category, f.key)]
        relevant = sorted((f for f in all_facts if relevance(f) >= 0 and f not in core),
                          key=relevance)
        for f in (core + relevant)[:FACT_LIMIT]:
            facts.append({"id": fact_id(f.category, f.key), "category": f.category,
                          "key": f.key, "value": f.value,
                          "core": is_core(f.category, f.key),
                          "sources": list(f.source_event_ids)})

    beliefs: list[dict] = []
    if view.show_beliefs:
        rows = await pg.execute(
            select(UserBelief).where(UserBelief.user_id == user_id,
                                     UserBelief.app_id == app_id,
                                     UserBelief.status == BeliefStatus.ACTIVE)
            .order_by(UserBelief.confidence.desc()).limit(BELIEF_LIMIT))
        beliefs = [{"id": f"{BELIEF_PREFIX}{b.id}", "statement": b.statement,
                    "category": b.category, "confidence": round(b.confidence or 0.0, 2)}
                   for b in rows.scalars().all()]

    memories: list[dict] = []
    if view.show_memories:
        # With no facts shown, nothing covers a memory: all of them may show.
        hidden = covered if view.show_facts else set()
        recent = await episodic.get_recent(pg, user_id, app_ids, limit=MEMORY_LIMIT * 3,
                                           order="active", own_only=True)
        seen: set[str] = set()
        for event in [*found, *recent]:
            key = str(event.id)
            if key in seen or key in hidden:
                continue
            seen.add(key)
            memories.append(_memory(event))
            if len(memories) >= MEMORY_LIMIT:
                break

    shown = list(dict.fromkeys([m["id"] for m in memories]
                               + [i for f in facts for i in f["sources"]]))
    await _count_shown(pg, neo, episodic, semantic, user_id, app_id, view, shown,
                       facts, beliefs)
    return {"user_id": user_id, "view": view.as_dict(), "facts": facts,
            "beliefs": beliefs, "memories": memories}


async def _count_shown(pg, neo, episodic, semantic, user_id, app_id, view, shown,
                       facts, beliefs) -> None:
    ids = []
    for raw in shown:
        try:
            ids.append(uuid.UUID(raw))
        except ValueError:
            continue
    if ids:
        await episodic.increment_recall(pg, ids)
    if view.keep_by != "shown":
        return
    await mark_used(pg, neo, episodic=episodic, semantic=semantic, user_id=user_id,
                    app_id=app_id, ids=[*shown, *(f["id"] for f in facts),
                                        *(b["id"] for b in beliefs)])


async def mark_used(pg: AsyncSession, neo, *, episodic, semantic, user_id: str,
                    app_id: str = "default", ids: list[str]) -> int:
    """Memories, facts and beliefs an answer leaned on (or, under
    keep_by "shown", was shown). A fact is touched in the graph and its
    memories marked; a belief records the time. Returns how many of the
    user's memories, facts and beliefs were marked."""
    events, fact_keys, belief_ids = parse_ids(ids)
    touched = 0
    if fact_keys and neo is not None:
        try:
            behind = await semantic.touch_facts(neo, user_id=user_id, keys=fact_keys,
                                                app_id=app_id)
            touched = len(fact_keys)
            more, _, _ = parse_ids(behind)
            events = list(dict.fromkeys([*events, *more]))
        except Exception as exc:
            logger.warning("Facts not marked used (graph unavailable): %s", exc)
    marked = await episodic.mark_used(pg, user_id, events) if events else 0
    believed = 0
    if belief_ids:
        result = await pg.execute(
            update(UserBelief)
            .where(UserBelief.id.in_(belief_ids), UserBelief.user_id == user_id)
            .values(last_used_at=datetime.now(timezone.utc)))
        believed = int(result.rowcount or 0)
    return marked + touched + believed
