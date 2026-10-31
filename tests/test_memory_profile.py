"""
Tests for the memory view (memory/views.py) and the profile a client's
prompt is given (memory/profile.py).

The bias these tests encode: WHAT THE ADMIN TURNED OFF NEVER REACHES A
PROMPT, AND NOTHING IS SAID TWICE. A switch set for one user overrides
only that field; a memory a fact covers is not handed over beside the
fact; an account number rides only when the question found it; and
being shown keeps a memory only when the view says "shown".
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.memory import profile, views
from smritikosh.memory.semantic import FactRecord


def fact(category, key, value, *sources):
    return FactRecord(category=category, key=key, value=value, confidence=0.9,
                      frequency_count=1, first_seen_at="", last_seen_at="",
                      source_event_ids=list(sources))


def event(text):
    return SimpleNamespace(id=uuid.uuid4(), raw_text=text,
                           created_at=datetime(2026, 10, 1, tzinfo=timezone.utc),
                           last_used_at=None, event_metadata={})


def row(**fields):
    return SimpleNamespace(**{name: fields.get(name) for name in views.FIELDS})


class TestTheView:
    async def test_a_users_row_overrides_only_what_it_sets(self):
        rows = {"asha": row(show_beliefs=False), "*": row(keep_by="shown")}
        with patch.object(views, "_row", AsyncMock(side_effect=lambda pg, u, a: rows.get(u))):
            view, source = await views.effective(None, "asha")
        assert view.show_beliefs is False and source["show_beliefs"] == "user"
        assert view.keep_by == "shown" and source["keep_by"] == "everyone"
        assert view.show_facts is True and source["show_facts"] == "default"

    async def test_nobody_said_anything_means_all_on_and_kept_by_use(self):
        with patch.object(views, "_row", AsyncMock(return_value=None)):
            view, _ = await views.effective(None, "asha")
        assert view == views.View(True, True, True, "used")

    async def test_keep_by_is_used_or_shown_and_nothing_else(self):
        with pytest.raises(ValueError, match="keep_by"):
            await views.set_view(AsyncMock(), "asha", keep_by="never")


class TestIds:
    def test_a_fact_and_a_belief_come_back_as_what_they_name(self):
        b = uuid.uuid4()
        m = uuid.uuid4()
        events, facts, beliefs = profile.parse_ids(
            [str(m), "fact:location/city", f"belief:{b}", "nonsense", "fact:broken"])
        assert events == [m] and facts == [("location", "city")] and beliefs == [b]

    def test_core_is_who_they_are_where_they_live_and_what_they_prefer(self):
        assert profile.is_core("identity", "email")
        assert profile.is_core("location", "city")
        assert not profile.is_core("location", "address")      # rides when relevant
        assert not profile.is_core("finance", "robinhood_ira_account")
        assert not profile.is_core("identity", "naina.email")  # someone else's


def deps(facts, found=(), recent=(), beliefs=()):
    episodic = MagicMock()
    episodic.hybrid_search = AsyncMock(return_value=[SimpleNamespace(event=e) for e in found])
    episodic.get_recent = AsyncMock(return_value=list(recent))
    episodic.increment_recall = AsyncMock()
    episodic.mark_used = AsyncMock(return_value=0)
    semantic = MagicMock()
    semantic.get_facts = AsyncMock(return_value=facts)
    semantic.touch_facts = AsyncMock(return_value=[])
    pg = MagicMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = list(beliefs)
    pg.execute = AsyncMock(return_value=result)
    return pg, episodic, semantic


async def build(view, facts, **kw):
    pg, episodic, semantic = deps(facts, **kw)
    with patch.object(views, "effective", AsyncMock(return_value=(view, {}))):
        got = await profile.build(pg, object(), episodic=episodic, semantic=semantic,
                                  user_id="mahen", query="q", embedding=[0.1])
    return got, episodic, semantic


class TestTheProfile:
    async def test_core_facts_always_and_an_account_only_when_the_question_found_it(self):
        home, ira = event("Lives in Cary"), event("IRA is 888")
        facts = [fact("location", "city", "Lives in Cary", str(home.id)),
                 fact("finance", "ira", "IRA is 888", str(ira.id))]
        got, _, _ = await build(views.View(), facts, found=[])
        assert [f["key"] for f in got["facts"]] == ["city"]
        got, _, _ = await build(views.View(), facts, found=[ira])
        assert [f["id"] for f in got["facts"]] == ["fact:location/city", "fact:finance/ira"]

    async def test_a_memory_a_fact_covers_is_not_said_twice(self):
        home, new = event("Lives in Cary"), event("Started a pottery class")
        facts = [fact("location", "city", "Lives in Cary", str(home.id))]
        got, _, _ = await build(views.View(), facts, recent=[home, new])
        assert [m["statement"] for m in got["memories"]] == ["Started a pottery class"]

    async def test_with_facts_off_every_memory_may_show(self):
        home = event("Lives in Cary")
        facts = [fact("location", "city", "Lives in Cary", str(home.id))]
        got, _, _ = await build(views.View(show_facts=False), facts, recent=[home])
        assert got["facts"] == [] and [m["statement"] for m in got["memories"]] == [
            "Lives in Cary"]

    async def test_what_the_admin_turned_off_is_never_handed_over(self):
        b = SimpleNamespace(id=uuid.uuid4(), statement="prefers charts", category="attitude",
                            confidence=0.7)
        got, _, _ = await build(views.View(show_beliefs=False, show_memories=False),
                                [], recent=[event("x")], beliefs=[b])
        assert got["beliefs"] == [] and got["memories"] == []
        got, _, _ = await build(views.View(), [], beliefs=[b])
        assert got["beliefs"] == [{"id": f"belief:{b.id}", "statement": "prefers charts",
                                   "category": "attitude", "confidence": 0.7}]

    async def test_shown_is_counted_but_keeps_nothing_under_used(self):
        new = event("Started a pottery class")
        _, episodic, _ = await build(views.View(), [], recent=[new])
        episodic.increment_recall.assert_awaited_once()
        episodic.mark_used.assert_not_awaited()

    async def test_under_shown_what_is_shown_is_marked_used(self):
        new = event("Started a pottery class")
        _, episodic, _ = await build(views.View(keep_by="shown"), [], recent=[new])
        episodic.mark_used.assert_awaited_once()
        assert episodic.mark_used.await_args.args[2] == [new.id]


class TestUsingAFact:
    async def test_leaning_on_a_fact_marks_the_memories_it_rests_on(self):
        behind = uuid.uuid4()
        pg, episodic, semantic = deps([])
        semantic.touch_facts = AsyncMock(return_value=[str(behind)])
        episodic.mark_used = AsyncMock(return_value=1)
        marked = await profile.mark_used(pg, object(), episodic=episodic, semantic=semantic,
                                         user_id="mahen", ids=["fact:location/city"])
        semantic.touch_facts.assert_awaited_once()
        assert episodic.mark_used.await_args.args[2] == [behind]
        assert marked == 2                      # the fact, and the memory behind it
