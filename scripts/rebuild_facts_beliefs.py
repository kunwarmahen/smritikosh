#!/usr/bin/env python3
"""
rebuild_facts_beliefs.py — make one user's facts and beliefs again, from
their memories, with the current consolidation and belief rules.

Facts and beliefs are derived: every one comes from the user's memories
(events). When the rules that derive them change, the old ones carry the
old mistakes -- duplicate keys, someone else's facts filed as the user's,
hundreds of beliefs at confidence 1.0 -- and new runs would only pile on
top. So this clears them and derives them again. Memories are never
deleted or reworded.

DRY RUN BY DEFAULT. Without --apply nothing is written: the memories are
read, the consolidation LLM is asked as the scheduler would ask it (each
batch seeing the facts the earlier batches made), and the old and new
facts and beliefs are printed side by side.

--label first files the user's memories that have no topic: the LLM
picks one from the fixed list (home, work, family, …), with whose fact
it is and which thing it names. A labelled memory files its own fact
with no LLM rewording, and lasts as its topic says (a project 60 days
unused, an account 365; home, family and the like until replaced or
forgotten). Shown in the dry run; written only with --apply.

--apply, in order:
  1. back up the user's current facts and beliefs to a JSON file;
  2. (with --label) write the labels into the memories' metadata;
  3. delete the user's fact links in the graph (and facts left orphaned)
     and their beliefs -- except retracted ones, which are what stops a
     wrong belief coming back;
  4. file each labelled memory's fact directly;
  5. mark the rest unconsolidated and run consolidation until none are left;
  6. mine beliefs once, then fold any near-duplicates.

Nothing else needs rebuilding: /context is assembled on each request from
memories, facts and beliefs.

Usage (from the smritikosh folder, with the stack running):
    uv run python scripts/rebuild_facts_beliefs.py --user alice
    uv run python scripts/rebuild_facts_beliefs.py --user alice --label
    uv run python scripts/rebuild_facts_beliefs.py --user alice --label --apply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select, update  # noqa: E402

from smritikosh.api.deps import get_episodic, get_llm, get_narrative, get_semantic  # noqa: E402
from smritikosh.db.models import SYSTEM_SOURCE_TYPES, BeliefStatus, Event, UserBelief  # noqa: E402
from smritikosh.db.neo4j import neo4j_session  # noqa: E402
from smritikosh.db.postgres import db_session  # noqa: E402
from smritikosh.memory.labels import AGENT_LABELLED, labelled_fact  # noqa: E402
from smritikosh.memory.semantic import FactRecord, UserProfile  # noqa: E402
from smritikosh.processing.belief_miner import (  # noqa: E402
    _BELIEF_EXAMPLE,
    _BELIEF_SCHEMA,
    MIN_EVIDENCE,
    BeliefMiner,
    _build_belief_prompt,
    _cited,
    evidence_confidence,
    tidy_beliefs,
)
from smritikosh.processing.consolidator import BATCH_SIZE, Consolidator  # noqa: E402

#: The topics a client may file under, with how many days each lasts
#: unused -- the same list Yantra sends (its memory/__init__.py TOPICS).
TOPIC_LIFETIME: dict[str, int | None] = {
    "home": None, "work": None, "family": None, "contact": None, "health": None,
    "diet": None, "account": 365, "tool": 365, "preference": 365, "habit": 180,
    "project": 60, "other": None,
}
_SLUG = re.compile(r"[^a-z0-9]+")

_LABEL_SCHEMA = (
    "labels: list of objects, one per numbered statement, with: "
    "n (int, the statement's number), "
    f"topic (one of: {', '.join(TOPIC_LIFETIME)}), "
    "about (string: whose fact it is when it is NOT the user's own -- a first "
    "name like 'Naina'; empty for the user's own), "
    "key (string: the one thing it names, e.g. 'city' for where they live, "
    "'email', 'brokerage'; a later fact with the same topic, about and key "
    "replaces this one, so two different things never share a key)."
)
_LABEL_EXAMPLE = {"labels": [
    {"n": 1, "topic": "home", "about": "", "key": ""},
    {"n": 2, "topic": "contact", "about": "Meera", "key": "email"},
    {"n": 3, "topic": "tool", "about": "", "key": "editor"},
]}


def _slug(text: str | None) -> str:
    return _SLUG.sub("_", (text or "").lower()).strip("_")


def slot(topic: str, about: str, key: str) -> str | None:
    """Yantra's slot rule: a topic and a key, or no slot (nothing replaces it)."""
    if topic not in TOPIC_LIFETIME or not _slug(key):
        return None
    return f"{topic}|{_slug(about)}|{_slug(key)}"


# ── Reading what is there ──────────────────────────────────────────────────────


async def load(user: str, app: str):
    async with db_session() as pg:
        events = list((await pg.execute(
            select(Event).where(Event.user_id == user, Event.app_id == app)
            .order_by(Event.created_at.asc())
        )).scalars().all())
        beliefs = list((await pg.execute(
            select(UserBelief).where(UserBelief.user_id == user, UserBelief.app_id == app)
        )).scalars().all())
    async with neo4j_session() as neo:
        facts = await get_semantic().get_facts(neo, user, app, min_confidence=0.0,
                                               active_only=False)
    return events, beliefs, facts


# ── Labelling ──────────────────────────────────────────────────────────────────


async def propose_labels(events: list[Event]) -> dict[str, dict]:
    """event id → {topic, about, key} for memories with no topic yet."""
    llm = get_llm()
    todo = [e for e in events if not (e.event_metadata or {}).get("topic")
            and e.source_type not in SYSTEM_SOURCE_TYPES]
    out: dict[str, dict] = {}
    for start in range(0, len(todo), 20):
        chunk = todo[start:start + 20]
        prompt = "\n".join(
            ["File each statement about a user under one topic.", ""]
            + [f"{i}. {e.raw_text[:300]}" for i, e in enumerate(chunk, 1)])
        reply = await llm.extract_structured(
            prompt=prompt, schema_description=_LABEL_SCHEMA, example_output=_LABEL_EXAMPLE)
        for label in reply.get("labels", []) or []:
            try:
                event = chunk[int(label["n"]) - 1]
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            topic = str(label.get("topic") or "").lower()
            if topic in TOPIC_LIFETIME:
                out[str(event.id)] = {"topic": topic,
                                      "about": str(label.get("about") or "").strip(),
                                      "key": str(label.get("key") or "").strip()}
    return out


def labelled_metadata(meta: dict, label: dict) -> dict:
    new = dict(meta or {})
    new.update({"topic": label["topic"], "slot": slot(label["topic"], label["about"], label["key"])})
    if label["about"]:
        new["about"] = label["about"]
    if label["key"]:
        new["key"] = label["key"]
    if TOPIC_LIFETIME[label["topic"]] is not None:
        new["lifetime_days"] = TOPIC_LIFETIME[label["topic"]]
    return new


# ── Dry run: ask the LLM, write nothing ────────────────────────────────────────


class _Recorder:
    """Stands in for SemanticMemory: facts kept in a dict, so each batch
    sees the facts the earlier ones made, as it would for real."""

    def __init__(self) -> None:
        self.facts: dict[tuple[str, str], str] = {}

    async def get_user_profile(self, _neo, user_id, app_id, min_confidence=0.0):
        return UserProfile(user_id=user_id, app_id=app_id, facts=[
            FactRecord(category=c, key=k, value=v, confidence=1.0, frequency_count=1,
                       first_seen_at="", last_seen_at="")
            for (c, k), v in self.facts.items()])

    async def set_fact(self, _neo, *, category, key, value, **_):
        self.facts[(category, key)] = value
        return None, []


class _NoWrites:
    def __getattr__(self, _name):
        async def nothing(*_a, **_k):
            return None
        return nothing


async def dry_facts(events: list[Event], labels: dict[str, dict], user: str, app: str):
    recorder = _Recorder()
    labelled, rest = [], []
    for e in events:
        if e.source_type in SYSTEM_SOURCE_TYPES:
            continue                      # Smritikosh's own findings: never facts
        meta = (labelled_metadata(e.event_metadata, labels[str(e.id)])
                if str(e.id) in labels else e.event_metadata)
        fact = labelled_fact(meta, e.raw_text)
        (labelled if fact else rest).append((e, fact))
    for _e, fact in labelled:
        recorder.facts[(fact["category"], fact["key"])] = fact["value"]
    consolidator = Consolidator(llm=get_llm(), episodic=_NoWrites(), semantic=recorder,
                                narrative=None)
    plain = [e for e, _ in rest]
    for start in range(0, len(plain), BATCH_SIZE):
        batch = plain[start:start + BATCH_SIZE]
        for e in batch:
            e.summary = None              # read as written, as after a reset
        await consolidator._consolidate_batch(None, None, user, app, batch)
        print(f"  consolidated {min(start + BATCH_SIZE, len(plain))}/{len(plain)}",
              file=sys.stderr)
    return recorder.facts, len(labelled)


async def dry_beliefs(facts: dict, events: list[Event]):
    recent = [e for e in reversed(events) if e.source_type not in SYSTEM_SOURCE_TYPES][:20]
    records = [FactRecord(category=c, key=k, value=v, confidence=1.0, frequency_count=1,
                          first_seen_at="", last_seen_at="") for (c, k), v in facts.items()][:15]
    reply = await get_llm().extract_structured(
        prompt=_build_belief_prompt(records, recent, [], []),
        schema_description=_BELIEF_SCHEMA, example_output=_BELIEF_EXAMPLE)
    kept = []
    for bd in reply.get("beliefs", []) or []:
        cited = _cited(bd.get("evidence"), recent)
        if len(cited) >= MIN_EVIDENCE and str(bd.get("statement", "")).strip():
            kept.append((bd["statement"], evidence_confidence(len(cited)), len(cited)))
    return kept


# ── Apply ──────────────────────────────────────────────────────────────────────


async def apply(user: str, app: str, events, beliefs, facts, labels: dict[str, dict]) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = Path(f"rebuild-backup-{user}-{stamp}.json")
    backup.write_text(json.dumps({
        "user": user, "app": app,
        "facts": [{"category": f.category, "key": f.key, "value": f.value,
                   "confidence": f.confidence, "source_event_ids": f.source_event_ids,
                   "source_type": f.source_type, "status": f.status} for f in facts],
        "beliefs": [{"statement": b.statement, "category": b.category,
                     "confidence": b.confidence, "evidence_count": b.evidence_count,
                     "status": b.status} for b in beliefs],
        "labels": labels,
    }, indent=1, default=str))
    print(f"backed up {len(facts)} facts and {len(beliefs)} beliefs to {backup}")

    semantic, episodic = get_semantic(), get_episodic()
    async with db_session() as pg:
        for event_id, label in labels.items():
            event = await pg.get(Event, uuid.UUID(event_id))
            if event is not None:
                event.event_metadata = labelled_metadata(event.event_metadata, label)
        await pg.execute(delete(UserBelief).where(
            UserBelief.user_id == user, UserBelief.app_id == app,
            UserBelief.status != BeliefStatus.REJECTED))
    async with neo4j_session() as neo:
        await neo.run(
            "MATCH (u:User {user_id: $u, app_id: $a})-[r]->(f:Fact) DELETE r "
            "WITH DISTINCT f WHERE NOT (f)--() DELETE f", u=user, a=app)
    print("cleared the old facts and beliefs (retracted beliefs kept)")

    filed = 0
    async with db_session() as pg, neo4j_session() as neo:
        rows = (await pg.execute(select(Event).where(
            Event.user_id == user, Event.app_id == app))).scalars().all()
        plain = []
        for e in rows:
            fact = labelled_fact(e.event_metadata, e.raw_text)
            if fact is None:
                plain.append(e.id)
                continue
            await semantic.set_fact(neo, user_id=user, app_id=app, category=fact["category"],
                                    key=fact["key"], value=fact["value"],
                                    confidence=fact["confidence"],
                                    source_event_ids=[str(e.id)], source_type=AGENT_LABELLED)
            await episodic.mark_consolidated(pg, [e.id], user_id=user, app_id=app,
                                             anchor_event_id=e.id)
            filed += 1
        if plain:
            await pg.execute(update(Event).where(Event.id.in_(plain)).values(
                consolidated=False, consolidation_anchor=False, summary=None))
    print(f"filed {filed} labelled memories as facts; {len(plain)} left for consolidation")

    consolidator = Consolidator(llm=get_llm(), episodic=episodic, semantic=semantic,
                                narrative=get_narrative(), min_events=1)
    while True:
        async with db_session() as pg, neo4j_session() as neo:
            result = await consolidator.run(pg, neo, user_id=user, app_id=app)
        print(f"  consolidation: {result.events_consolidated} memories, "
              f"{result.facts_distilled} facts", file=sys.stderr)
        if result.skipped or not result.events_consolidated:
            break

    miner = BeliefMiner(llm=get_llm(), semantic=semantic)
    async with db_session() as pg, neo4j_session() as neo:
        mined = await miner.mine(pg, neo, user_id=user, app_id=app)
    async with db_session() as pg:
        folded = await tidy_beliefs(pg, get_llm(), user_id=user, app_id=app)
    print(f"beliefs: {mined.beliefs_upserted} new, {mined.beliefs_unsupported} dropped for "
          f"too little evidence, {folded} folded as duplicates")


# ── Main ───────────────────────────────────────────────────────────────────────


def show_facts(title: str, rows: list[tuple[str, str, str]]) -> None:
    print(f"\n{title} ({len(rows)})")
    for c, k, v in sorted(rows):
        print(f"  {c}/{k}: {v[:90]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--user", required=True)
    parser.add_argument("--app", default="default")
    parser.add_argument("--label", action="store_true",
                        help="file memories that have no topic first")
    parser.add_argument("--apply", action="store_true", help="write (default: dry run)")
    args = parser.parse_args()

    events, beliefs, facts = await load(args.user, args.app)
    active = [b for b in beliefs if b.status != BeliefStatus.REJECTED]
    print(f"{args.user}: {len(events)} memories, {len(facts)} facts, "
          f"{len(active)} beliefs ({len(beliefs) - len(active)} retracted)")

    labels = await propose_labels(events) if args.label else {}
    if labels:
        print(f"\nlabels for {len(labels)} memories:")
        by_id = {str(e.id): e for e in events}
        for event_id, label in labels.items():
            who = f" about {label['about']}" if label["about"] else ""
            key = f" key {label['key']}" if label["key"] else ""
            print(f"  [{label['topic']}{who}{key}] {by_id[event_id].raw_text[:80]}")

    if args.apply:
        await apply(args.user, args.app, events, beliefs, facts, labels)
        events, beliefs, facts = await load(args.user, args.app)
        active = [b for b in beliefs if b.status != BeliefStatus.REJECTED]
        show_facts("facts now", [(f.category, f.key, f.value) for f in facts])
        print(f"\nbeliefs now ({len(active)})")
        for b in sorted(active, key=lambda b: -b.confidence):
            print(f"  {b.confidence:.2f}  {b.statement}")
        return

    show_facts("facts now", [(f.category, f.key, f.value) for f in facts])
    new_facts, filed = await dry_facts(events, labels, args.user, args.app)
    show_facts(f"facts after a rebuild ({filed} filed straight from labels)",
               [(c, k, v) for (c, k), v in new_facts.items()])
    print(f"\nbeliefs now: {len(active)}; after a rebuild, one mining run would keep:")
    for statement, confidence, n in await dry_beliefs(new_facts, events):
        print(f"  {confidence:.2f}  {statement}  ({n} memories)")
    print("\nDry run: nothing was written. Add --apply to rebuild.")


if __name__ == "__main__":
    asyncio.run(main())
