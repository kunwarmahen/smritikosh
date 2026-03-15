"""
Tests for SemanticMemory.

How to run:
    # Unit tests (no Neo4j — session is mocked):
    pytest tests/test_semantic_memory.py -v

    # DB integration tests (requires running Neo4j):
    #   docker compose up -d neo4j
    pytest tests/test_semantic_memory.py -v -m db

Test strategy:
    - Unit tests mock the Neo4j AsyncSession to verify Cypher is called correctly
      and that FactRecord / UserProfile objects are built from returned data.
    - DB integration tests run against a live Neo4j instance.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smritikosh.db.models import FactCategory
from smritikosh.memory.semantic import (
    FactRecord,
    SemanticMemory,
    UserProfile,
    _rel_type,
    _record_to_fact,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_semantic() -> SemanticMemory:
    return SemanticMemory()


def make_mock_session(return_data: list[dict] | dict | None = None) -> AsyncMock:
    """
    Build a mock Neo4j AsyncSession.

    Neo4j result chain: session.run() → AsyncResult → .single() or .data()
    """
    session = AsyncMock()

    # Build the result mock
    result_mock = AsyncMock()

    if isinstance(return_data, dict):
        # single() call returns one record
        result_mock.single = AsyncMock(return_value=return_data)
        result_mock.data = AsyncMock(return_value=[return_data])
    elif isinstance(return_data, list):
        result_mock.single = AsyncMock(return_value=return_data[0] if return_data else None)
        result_mock.data = AsyncMock(return_value=return_data)
    else:
        result_mock.single = AsyncMock(return_value=None)
        result_mock.data = AsyncMock(return_value=[])

    session.run = AsyncMock(return_value=result_mock)
    return session


def make_fact_record(**kwargs) -> dict:
    """Returns a raw Neo4j-style record dict."""
    return {
        "category": kwargs.get("category", "interest"),
        "key": kwargs.get("key", "domain"),
        "value": kwargs.get("value", "AI agents"),
        "confidence": kwargs.get("confidence", 0.9),
        "frequency_count": kwargs.get("frequency_count", 3),
        "first_seen_at": kwargs.get("first_seen_at", "2026-03-10T00:00:00+00:00"),
        "last_seen_at": kwargs.get("last_seen_at", "2026-03-15T00:00:00+00:00"),
    }


# ── _rel_type ─────────────────────────────────────────────────────────────────


class TestRelType:
    def test_preference(self):
        assert _rel_type("preference") == "HAS_PREFERENCE"

    def test_interest(self):
        assert _rel_type("interest") == "HAS_INTEREST"

    def test_role(self):
        assert _rel_type("role") == "HAS_ROLE"

    def test_project(self):
        assert _rel_type("project") == "WORKS_ON"

    def test_skill(self):
        assert _rel_type("skill") == "HAS_SKILL"

    def test_goal(self):
        assert _rel_type("goal") == "HAS_GOAL"

    def test_relationship(self):
        assert _rel_type("relationship") == "KNOWS"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown fact category"):
            _rel_type("unknown_category")


# ── _record_to_fact ───────────────────────────────────────────────────────────


class TestRecordToFact:
    def test_converts_dict_to_fact_record(self):
        raw = make_fact_record(category="preference", key="color", value="green")
        fact = _record_to_fact(raw)
        assert isinstance(fact, FactRecord)
        assert fact.category == "preference"
        assert fact.key == "color"
        assert fact.value == "green"
        assert fact.confidence == 0.9
        assert fact.frequency_count == 3

    def test_types_are_correct(self):
        raw = make_fact_record(confidence="0.75", frequency_count="5")
        fact = _record_to_fact(raw)
        assert isinstance(fact.confidence, float)
        assert isinstance(fact.frequency_count, int)


# ── UserProfile ───────────────────────────────────────────────────────────────


class TestUserProfile:
    def _make_profile(self) -> UserProfile:
        return UserProfile(
            user_id="u1",
            app_id="default",
            facts=[
                FactRecord("preference", "ui_color", "green", 0.9, 3, "2026-03-01", "2026-03-15"),
                FactRecord("interest", "domain", "AI agents", 0.95, 5, "2026-03-01", "2026-03-15"),
                FactRecord("interest", "topic", "LLM infra", 0.8, 2, "2026-03-01", "2026-03-15"),
                FactRecord("role", "current", "entrepreneur", 1.0, 1, "2026-03-01", "2026-03-15"),
            ],
        )

    def test_by_category_groups_correctly(self):
        profile = self._make_profile()
        grouped = profile.by_category()
        assert set(grouped.keys()) == {"preference", "interest", "role"}
        assert len(grouped["interest"]) == 2

    def test_as_text_summary_includes_all_categories(self):
        profile = self._make_profile()
        summary = profile.as_text_summary()
        assert "Preference" in summary
        assert "Interest" in summary
        assert "Role" in summary
        assert "green" in summary
        assert "AI agents" in summary

    def test_empty_profile_returns_placeholder(self):
        profile = UserProfile(user_id="u1", app_id="default", facts=[])
        assert "(no facts stored)" in profile.as_text_summary()

    def test_by_category_returns_empty_dict_for_no_facts(self):
        profile = UserProfile(user_id="u1", app_id="default")
        assert profile.by_category() == {}


# ── upsert_fact() ─────────────────────────────────────────────────────────────


class TestUpsertFact:
    @pytest.mark.asyncio
    async def test_calls_session_run(self):
        semantic = make_semantic()
        raw = make_fact_record(category="interest", key="domain", value="AI agents")
        session = make_mock_session(return_data=raw)

        fact = await semantic.upsert_fact(
            session,
            user_id="u1",
            category="interest",
            key="domain",
            value="AI agents",
        )

        session.run.assert_called_once()
        assert isinstance(fact, FactRecord)
        assert fact.category == "interest"
        assert fact.value == "AI agents"

    @pytest.mark.asyncio
    async def test_cypher_contains_rel_type(self):
        """The correct relationship type must appear in the Cypher query."""
        semantic = make_semantic()
        raw = make_fact_record(category="preference")
        session = make_mock_session(return_data=raw)

        await semantic.upsert_fact(
            session, user_id="u1", category="preference", key="color", value="green"
        )

        cypher_call = session.run.call_args[0][0]
        assert "HAS_PREFERENCE" in cypher_call

    @pytest.mark.asyncio
    async def test_invalid_category_raises(self):
        semantic = make_semantic()
        session = make_mock_session()

        with pytest.raises(ValueError, match="Unknown fact category"):
            await semantic.upsert_fact(
                session, user_id="u1", category="bogus", key="k", value="v"
            )

    @pytest.mark.asyncio
    async def test_default_confidence_is_one(self):
        semantic = make_semantic()
        raw = make_fact_record()
        session = make_mock_session(return_data=raw)

        await semantic.upsert_fact(
            session, user_id="u1", category="interest", key="k", value="v"
        )

        call_kwargs = session.run.call_args[1]
        assert call_kwargs["confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_custom_app_id_passed(self):
        semantic = make_semantic()
        raw = make_fact_record()
        session = make_mock_session(return_data=raw)

        await semantic.upsert_fact(
            session, user_id="u1", category="interest", key="k", value="v", app_id="my_app"
        )

        call_kwargs = session.run.call_args[1]
        assert call_kwargs["app_id"] == "my_app"


# ── get_facts() ───────────────────────────────────────────────────────────────


class TestGetFacts:
    @pytest.mark.asyncio
    async def test_returns_list_of_fact_records(self):
        semantic = make_semantic()
        raws = [
            make_fact_record(key="domain", value="AI agents"),
            make_fact_record(key="topic", value="RAG"),
        ]
        session = make_mock_session(return_data=raws)

        facts = await semantic.get_facts(session, "u1")

        assert len(facts) == 2
        assert all(isinstance(f, FactRecord) for f in facts)

    @pytest.mark.asyncio
    async def test_filtered_by_category_uses_specific_rel(self):
        semantic = make_semantic()
        session = make_mock_session(return_data=[make_fact_record(category="skill")])

        await semantic.get_facts(session, "u1", category="skill")

        cypher = session.run.call_args[0][0]
        assert "HAS_SKILL" in cypher

    @pytest.mark.asyncio
    async def test_without_category_filter_uses_generic_match(self):
        semantic = make_semantic()
        session = make_mock_session(return_data=[])

        await semantic.get_facts(session, "u1")

        cypher = session.run.call_args[0][0]
        # No specific rel type — matches all outgoing relationships
        assert "-[r]->" in cypher

    @pytest.mark.asyncio
    async def test_min_confidence_passed_as_param(self):
        semantic = make_semantic()
        session = make_mock_session(return_data=[])

        await semantic.get_facts(session, "u1", min_confidence=0.7)

        call_kwargs = session.run.call_args[1]
        assert call_kwargs["min_confidence"] == 0.7


# ── get_user_profile() ────────────────────────────────────────────────────────


class TestGetUserProfile:
    @pytest.mark.asyncio
    async def test_returns_user_profile(self):
        semantic = make_semantic()
        raws = [
            make_fact_record(category="role", key="current", value="entrepreneur"),
        ]
        session = make_mock_session(return_data=raws)

        profile = await semantic.get_user_profile(session, "u1")

        assert isinstance(profile, UserProfile)
        assert profile.user_id == "u1"
        assert len(profile.facts) == 1
        assert profile.facts[0].category == "role"

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_profile(self):
        semantic = make_semantic()
        session = make_mock_session(return_data=[])

        profile = await semantic.get_user_profile(session, "u1")

        assert profile.facts == []
        assert "(no facts stored)" in profile.as_text_summary()


# ── delete_fact() ─────────────────────────────────────────────────────────────


class TestDeleteFact:
    @pytest.mark.asyncio
    async def test_returns_true_when_deleted(self):
        semantic = make_semantic()
        session = make_mock_session(return_data={"deleted_count": 1})

        result = await semantic.delete_fact(
            session, user_id="u1", category="interest", key="domain"
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_nothing_deleted(self):
        semantic = make_semantic()
        session = make_mock_session(return_data={"deleted_count": 0})

        result = await semantic.delete_fact(
            session, user_id="u1", category="interest", key="nonexistent"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_cypher_uses_correct_rel_type(self):
        semantic = make_semantic()
        session = make_mock_session(return_data={"deleted_count": 1})

        await semantic.delete_fact(session, user_id="u1", category="project", key="name")

        cypher = session.run.call_args[0][0]
        assert "WORKS_ON" in cypher


# ── user_exists() ─────────────────────────────────────────────────────────────


class TestUserExists:
    @pytest.mark.asyncio
    async def test_returns_true_when_found(self):
        semantic = make_semantic()
        session = make_mock_session(return_data={"n": 1})

        result = await semantic.user_exists(session, "u1")

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self):
        semantic = make_semantic()
        session = make_mock_session(return_data={"n": 0})

        result = await semantic.user_exists(session, "u1")

        assert result is False


# ── DB integration tests ───────────────────────────────────────────────────────


@pytest.mark.db
class TestSemanticMemoryDB:
    """
    How to run:
        docker compose up -d neo4j
        pytest tests/test_semantic_memory.py -v -m db

    Neo4j browser available at: http://localhost:7474
    """

    @pytest.fixture(autouse=True)
    async def setup_neo4j(self):
        from smritikosh.db.neo4j import get_driver, init_neo4j

        await init_neo4j()
        self.driver = get_driver()
        self.semantic = SemanticMemory()

        yield

        # Clean up test data
        async with self.driver.session() as session:
            await session.run("MATCH (n) WHERE n.user_id STARTS WITH 'test_' DETACH DELETE n")

    async def test_upsert_and_retrieve_fact(self):
        async with self.driver.session() as session:
            fact = await self.semantic.upsert_fact(
                session,
                user_id="test_u1",
                category="interest",
                key="domain",
                value="AI agents",
            )
        assert fact.category == "interest"
        assert fact.value == "AI agents"
        assert fact.frequency_count == 1
        assert fact.confidence == 1.0

    async def test_upsert_increments_frequency(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="test_u2", category="interest", key="domain", value="AI"
            )
            fact = await self.semantic.upsert_fact(
                session, user_id="test_u2", category="interest", key="domain", value="AI"
            )
        assert fact.frequency_count == 2

    async def test_get_facts_returns_correct_user(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="test_u3", category="role", key="current", value="entrepreneur"
            )
            await self.semantic.upsert_fact(
                session, user_id="test_u4", category="role", key="current", value="engineer"
            )
            facts_u3 = await self.semantic.get_facts(session, "test_u3")

        assert len(facts_u3) == 1
        assert facts_u3[0].value == "entrepreneur"

    async def test_get_user_profile_groups_by_category(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="test_u5", category="interest", key="ai", value="AI agents"
            )
            await self.semantic.upsert_fact(
                session, user_id="test_u5", category="preference", key="color", value="green"
            )
            profile = await self.semantic.get_user_profile(session, "test_u5")

        assert len(profile.facts) == 2
        grouped = profile.by_category()
        assert "interest" in grouped
        assert "preference" in grouped

    async def test_as_text_summary_format(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="test_u6", category="role", key="current", value="founder"
            )
            profile = await self.semantic.get_user_profile(session, "test_u6")

        summary = profile.as_text_summary()
        assert "Role" in summary
        assert "founder" in summary

    async def test_delete_fact(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="test_u7", category="goal", key="launch", value="Q2 2026"
            )
            deleted = await self.semantic.delete_fact(
                session, user_id="test_u7", category="goal", key="launch"
            )
            facts = await self.semantic.get_facts(session, "test_u7")

        assert deleted is True
        assert len(facts) == 0

    async def test_user_exists(self):
        async with self.driver.session() as session:
            before = await self.semantic.user_exists(session, "test_u8")
            await self.semantic.upsert_fact(
                session, user_id="test_u8", category="skill", key="rag", value="expert"
            )
            after = await self.semantic.user_exists(session, "test_u8")

        assert before is False
        assert after is True
