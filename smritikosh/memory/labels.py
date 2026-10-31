"""
Labels — a statement that arrives already filed.

A client that decides for itself what to remember (an agent with its own
look back) can say what each statement is about when it stores it, in
the event metadata:

    topic          one of TOPIC_CATEGORY's keys ("home", "account", …)
    about          whose fact it is, when not the user's ("Naina")
    key            the one thing it names ("email", "brokerage")
    lifetime_days  how long it lasts unused; absent = until deleted

Two things follow here.

STRAIGHT TO THE GRAPH. A labelled statement is already a fact, worded by
the model that decided to keep it. Running the consolidation LLM over it
again only rewords it (worse, on a small local model) and invents a key
that drifts from the last one. So it is written to the knowledge graph
as it stands -- category from the topic, key from key/topic prefixed by
``about``, value the statement, source the event -- and the event is
marked consolidated. Its fact lives exactly as long as its event: a
forget, a replace (the client forgets the old event) or a lapsed
lifetime takes both. ``source_type = agent_labelled`` keeps the weekly
decay off it, since time is the client's call (``lifetime_days``).

A LIFETIME IS THE CLIENT'S. The pruner deletes a labelled event once
``lifetime_days`` have passed since it was written or last used (POST
/memory/used), whichever is later.
"""

from __future__ import annotations

import re

from smritikosh.db.models import FactCategory, SourceType

#: The client's topic → the graph's category. "other" has none: an
#: unfiled statement is left for consolidation to read.
TOPIC_CATEGORY: dict[str, str] = {
    "home":       FactCategory.LOCATION,
    "work":       FactCategory.ROLE,
    "family":     FactCategory.RELATIONSHIP,
    "contact":    FactCategory.IDENTITY,
    "health":     FactCategory.HEALTH,
    "diet":       FactCategory.DIET,
    "account":    FactCategory.FINANCE,
    "tool":       FactCategory.TOOL,
    "preference": FactCategory.PREFERENCE,
    "habit":      FactCategory.HABIT,
    "project":    FactCategory.PROJECT,
}

#: The source type of a fact written from a label.
AGENT_LABELLED = SourceType.AGENT_LABELLED

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(text: str | None) -> str:
    return _SLUG.sub("_", (text or "").lower()).strip("_")


def fact_key(key: str | None, about: str | None = None, fallback: str = "") -> str:
    """``email`` for the user's own, ``naina.email`` for someone else's.

    Someone else's fact must never share a key with the user's: Akansha's
    date of birth filed as ``identity/dob`` would replace the user's own.
    """
    base = slug(key) or slug(fallback)
    who = slug(about)
    return f"{who}.{base}" if who else base


def labelled_fact(metadata: dict | None, statement: str) -> dict | None:
    """The fact a labelled statement files as, or None when it has no
    topic the graph knows."""
    meta = metadata or {}
    category = TOPIC_CATEGORY.get(str(meta.get("topic") or ""))
    if category is None:
        return None
    return {
        "category": str(category),
        "key": fact_key(meta.get("key"), meta.get("about"), fallback=str(meta["topic"])),
        "value": statement.strip(),
        "confidence": 1.0,
    }


def lifetime_days(metadata: dict | None) -> int | None:
    """The client's lifetime in whole days, or None for none / nonsense."""
    raw = (metadata or {}).get("lifetime_days")
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None
    return days if days > 0 else None
