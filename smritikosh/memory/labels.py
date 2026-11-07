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

THE MEMORY IS THE CLIENT'S WORDS; THE FACT IS SMRITIKOSH'S. A labelled
statement is stored exactly as sent -- that is the memory the person
approved and sees. Its fact is made from it at the next consolidation
cycle (processing/consolidator.py), worded short: the thing and its value
and at most a short detail in brackets, nothing else. Copying the
statement in whole made facts like "Has Acme Mutual as their auto insurer
(policy ending 4K, visible in their mail inbox), so a later 'who handles
my car insurance' question should default to Acme Mutual rather than asking
which carrier" -- a memory's words, where it was seen, and advice to the
agent, all carried into every prompt. The fact keeps the client's slot
(category from the topic, key from key/topic prefixed by ``about``), so
the cycle only words the value and never invents a key that drifts from
the last one. Until the cycle runs, the memory has no fact, and the
profile hands it over as a memory instead: nothing is missing meanwhile.
Its fact lives exactly as long as its event: a forget, a replace (the
client forgets the old event) or a lapsed lifetime takes both.
``source_type = agent_labelled`` keeps the weekly decay off it, since
time is the client's call (``lifetime_days``).

TRIMMED BY RULE WHEN THE MODEL CANNOT WORD IT. ``trim_statement`` cuts a
trailing "so a later ..." clause and a "visible in their inbox" aside
with no model at all; a fact whose model call failed is filed with that,
and every worded fact goes through it too.

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
        "value": trim_statement(statement),
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


#: Longer than this and a fact is a paragraph: kept to its first sentence.
MAX_FACT_CHARS = 140

#: A trailing clause that tells the agent what to do with the fact.
_ADVICE = re.compile(
    r"(?:[,;:]|\s[-–—])\s*(?:so|which means|meaning|that way|in case|"
    r"hence|therefore)\b.*$"
    r"|\s+so (?:that )?(?:a|any|the|future|later)\b.*$",
    re.IGNORECASE | re.DOTALL)
#: An aside saying where the fact was seen, not what it is.
_SOURCE = re.compile(
    r"\b(?:visible|seen|found|shown|noted|spotted|according)\b"
    r"|\b(?:from|in) (?:their|the) [\w ]*(?:inbox|email|e-mail|mail|account page|"
    r"screenshot|chat|conversation)\b",
    re.IGNORECASE)
_PARENS = re.compile(r"\s*\(([^()]*)\)")


def trim_statement(statement: str) -> str:
    """A statement cut down to its fact, by rule: no advice clause, no
    "where it was seen" aside, one sentence, ending in a full stop.

    >>> trim_statement("Has Acme Mutual as their auto insurer (policy ending 4K, "
    ...                "visible in their mail inbox), so a later question "
    ...                "should default to Acme Mutual.")
    'Has Acme Mutual as their auto insurer (policy ending 4K).'
    """
    text = " ".join((statement or "").split())

    def keep_detail(match: re.Match) -> str:
        parts = [p.strip() for p in match.group(1).split(",")]
        kept = [p for p in parts if p and not _SOURCE.search(p)]
        return f" ({', '.join(kept)})" if kept else ""

    text = _PARENS.sub(keep_detail, text)
    text = _ADVICE.sub("", text).strip()
    if len(text) > MAX_FACT_CHARS:
        head = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
        text = head if len(head) < len(text) else text
    text = text.rstrip(" ,;:-")
    if text and text[-1] not in ".!?":
        text += "."
    return text
