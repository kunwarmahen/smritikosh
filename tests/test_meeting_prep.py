"""
Tests for the Meeting Prep agent (E4, FUTURE.md #3):
    MeetingPrepAgent.prepare — per-attendee retrieval pools, dedup, citation
    validation, brief logged as memory, graceful pool failures.
    MeetingPrepAgent.debrief — notes re-enter the full encoding pipeline.

All tests run offline with mocked sessions and LLMs.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from smritikosh.cognition.meeting_prep import (
    DebriefResult,
    MeetingPrepAgent,
    MeetingPrepResult,
)
from smritikosh.db.models import Event, SourceType
from smritikosh.memory.episodic import SearchResult
from smritikosh.memory.hippocampus import EncodedMemory
from smritikosh.memory.semantic import FactRecord, UserProfile


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(text: str = "some memory") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id="u1",
        app_id="default",
        raw_text=text,
        importance_score=0.7,
        source_type=SourceType.API_EXPLICIT,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_profile(with_facts: bool = True) -> UserProfile:
    facts = [FactRecord(
        category="goal", key="q3", value="close the Acme pilot", confidence=0.9,
        frequency_count=1, first_seen_at="2026-01-01", last_seen_at="2026-06-01",
    )] if with_facts else []
    return UserProfile(user_id="u1", app_id="default", facts=facts)


def prep_response(name: str = "Priya", cited: list[str] | None = None) -> dict:
    return {
        "attendee_briefs": [
            {
                "name": name,
                "known_facts": ["CTO at Acme."],
                "history": ["2026-05-12: demo call."],
                "open_commitments": ["Send the security whitepaper."],
            }
        ],
        "talking_points": ["Lead with SOC2 progress."],
        "questions_to_ask": ["Pilot budget approved?"],
        "watch_outs": ["Pushed back on pricing last time."],
        "cited_event_ids": cited or [],
    }


def make_agent(
    search_events: list[Event],
    llm_response: dict | Exception | None = None,
    profile: UserProfile | None = None,
    recent: list[Event] | None = None,
) -> MeetingPrepAgent:
    llm = AsyncMock()
    if isinstance(llm_response, Exception):
        llm.extract_structured = AsyncMock(side_effect=llm_response)
    else:
        llm.extract_structured = AsyncMock(return_value=llm_response or prep_response())
    llm.embed = AsyncMock(return_value=[0.1] * 8)

    episodic = AsyncMock()
    episodic.hybrid_search = AsyncMock(
        return_value=[SearchResult(event=e, hybrid_score=0.9) for e in search_events]
    )
    episodic.get_recent = AsyncMock(return_value=recent or [])
    episodic.store = AsyncMock(return_value=make_event("brief log"))

    semantic = AsyncMock()
    semantic.get_user_profile = AsyncMock(return_value=profile or make_profile())

    hippocampus = AsyncMock()
    hippocampus.encode = AsyncMock(
        return_value=EncodedMemory(
            event=make_event("debrief"), facts=[], importance_score=0.7
        )
    )

    return MeetingPrepAgent(
        llm=llm, episodic=episodic, semantic=semantic, hippocampus=hippocampus
    )


# ── prepare ───────────────────────────────────────────────────────────────────


class TestPrepare:
    @pytest.mark.asyncio
    async def test_no_attendees_skips(self):
        agent = make_agent([])

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["  ", ""]
        )

        assert result.skipped is True
        assert "No attendees" in result.skip_reason
        agent.llm.extract_structured.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_memory_skips(self):
        agent = make_agent([], profile=make_profile(with_facts=False))

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        assert result.skipped is True
        assert "No memory" in result.skip_reason

    @pytest.mark.asyncio
    async def test_brief_parsed_and_logged(self):
        event = make_event("Priya asked about SOC2")
        agent = make_agent([event], prep_response(cited=[str(event.id)]))

        result = await agent.prepare(
            AsyncMock(), AsyncMock(),
            user_id="u1", attendees=["Priya"], topic="pilot renewal",
        )

        assert result.skipped is False
        assert result.attendee_briefs[0].name == "Priya"
        assert result.attendee_briefs[0].open_commitments == ["Send the security whitepaper."]
        assert result.talking_points == ["Lead with SOC2 progress."]
        assert result.cited_event_ids == [str(event.id)]
        # brief logged as memory with the right source type
        agent.episodic.store.assert_awaited_once()
        kwargs = agent.episodic.store.await_args.kwargs
        assert kwargs["source_type"] == SourceType.AGENT_MEETING_PREP
        assert kwargs["source_meta"]["attendees"] == ["Priya"]
        assert result.logged_event_id is not None

    @pytest.mark.asyncio
    async def test_one_search_pool_per_attendee_plus_topic(self):
        agent = make_agent([make_event()])

        await agent.prepare(
            AsyncMock(), AsyncMock(),
            user_id="u1", attendees=["Priya", "Rohan"], topic="renewal",
        )

        # 2 attendees + 1 topic = 3 embeds for pools (+1 for logging the brief)
        assert agent.episodic.hybrid_search.await_count == 3

    @pytest.mark.asyncio
    async def test_hallucinated_citations_filtered(self):
        event = make_event()
        agent = make_agent(
            [event], prep_response(cited=[str(event.id), str(uuid.uuid4())])
        )

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        assert result.cited_event_ids == [str(event.id)]

    @pytest.mark.asyncio
    async def test_events_deduped_across_pools(self):
        event = make_event()   # same event returned by both attendee pools
        agent = make_agent([event])

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya", "Rohan"]
        )

        assert result.memories_considered == 1

    @pytest.mark.asyncio
    async def test_embed_failure_degrades_to_empty_pool(self):
        agent = make_agent([make_event()])
        agent.llm.embed = AsyncMock(side_effect=RuntimeError("embed down"))

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        # profile facts still exist → synthesis proceeds without event pools
        assert result.skipped is False
        agent.episodic.hybrid_search.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_skipped(self):
        agent = make_agent([make_event()], RuntimeError("provider down"))

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        assert result.skipped is True
        assert "provider down" in result.skip_reason
        agent.episodic.store.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nested_meeting_lists_hoisted(self):
        # Some local models put talking_points etc. inside each attendee
        # object instead of at the top level — they must be hoisted, and
        # merged without duplicating top-level entries.
        response = {
            "attendee_briefs": [
                {
                    "name": "Priya",
                    "known_facts": ["CTO."],
                    "talking_points": ["Nested point.", "Shared point."],
                    "questions_to_ask": ["Nested question?"],
                    "watch_outs": ["Nested risk."],
                }
            ],
            "talking_points": ["Shared point."],
        }
        agent = make_agent([make_event()], response)

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        assert result.talking_points == ["Shared point.", "Nested point."]
        assert result.questions_to_ask == ["Nested question?"]
        assert result.watch_outs == ["Nested risk."]

    @pytest.mark.asyncio
    async def test_malformed_attendee_brief_skipped(self):
        response = prep_response()
        response["attendee_briefs"].append("not-an-object")
        agent = make_agent([make_event()], response)

        result = await agent.prepare(
            AsyncMock(), AsyncMock(), user_id="u1", attendees=["Priya"]
        )

        assert len(result.attendee_briefs) == 1


# ── debrief ───────────────────────────────────────────────────────────────────


class TestDebrief:
    @pytest.mark.asyncio
    async def test_notes_enter_encoding_pipeline(self):
        agent = make_agent([])

        result = await agent.debrief(
            AsyncMock(), AsyncMock(),
            user_id="u1", notes="Priya confirmed the pilot budget.",
            attendees=["Priya"],
        )

        agent.hippocampus.encode.assert_awaited_once()
        kwargs = agent.hippocampus.encode.await_args.kwargs
        assert kwargs["source_type"] == SourceType.MEETING_DEBRIEF
        assert kwargs["source_meta"]["attendees"] == ["Priya"]
        assert result.event_id is not None
        assert result.extraction_failed is False

    @pytest.mark.asyncio
    async def test_extraction_failure_surfaced(self):
        agent = make_agent([])
        agent.hippocampus.encode = AsyncMock(
            return_value=EncodedMemory(
                event=make_event(), facts=[], importance_score=0.5,
                extraction_failed=True,
            )
        )

        result = await agent.debrief(
            AsyncMock(), AsyncMock(), user_id="u1", notes="Some meeting notes here."
        )

        assert result.extraction_failed is True


class TestResults:
    def test_prep_defaults(self):
        r = MeetingPrepResult(user_id="u1", app_id="default", attendees=["A"])
        assert r.attendee_briefs == []
        assert r.skipped is False

    def test_debrief_defaults(self):
        r = DebriefResult(user_id="u1", app_id="default")
        assert r.facts_extracted == 0
