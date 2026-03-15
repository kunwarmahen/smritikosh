"""
Tests for IdentityBuilder and UserIdentity.

Unit tests mock SemanticMemory and LLMAdapter.
DB integration tests (gated behind @pytest.mark.db) verify the full pipeline
against a live Neo4j database.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.memory.identity import (
    IdentityBuilder,
    IdentityDimension,
    UserIdentity,
    _build_dimensions,
    _fallback_summary,
)
from smritikosh.memory.semantic import FactRecord, UserProfile


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_fact(category: str, key: str, value: str, confidence: float = 0.8) -> FactRecord:
    return FactRecord(
        category=category,
        key=key,
        value=value,
        confidence=confidence,
        frequency_count=1,
        first_seen_at="2026-01-01",
        last_seen_at="2026-03-15",
    )


def make_profile(facts: list[FactRecord] | None = None) -> UserProfile:
    return UserProfile(
        user_id="u1",
        app_id="default",
        facts=facts or [],
    )


def make_mock_llm(summary: str = "Test summary.") -> AsyncMock:
    llm = AsyncMock()
    llm.extract_structured = AsyncMock(return_value={"summary": summary})
    return llm


def make_mock_semantic(facts: list[FactRecord] | None = None) -> AsyncMock:
    from smritikosh.memory.semantic import SemanticMemory
    semantic = AsyncMock(spec=SemanticMemory)
    semantic.get_user_profile = AsyncMock(return_value=make_profile(facts or []))
    return semantic


# ── _build_dimensions ─────────────────────────────────────────────────────────


class TestBuildDimensions:
    def test_empty_facts_returns_empty(self):
        assert _build_dimensions([]) == []

    def test_groups_by_category(self):
        facts = [
            make_fact("role", "current", "engineer"),
            make_fact("interest", "domain", "AI"),
            make_fact("role", "past", "student", confidence=0.6),
        ]
        dims = _build_dimensions(facts)
        categories = {d.category for d in dims}
        assert categories == {"role", "interest"}

    def test_categories_sorted_alphabetically(self):
        facts = [
            make_fact("role", "current", "engineer"),
            make_fact("interest", "domain", "AI"),
            make_fact("preference", "ui_color", "green"),
        ]
        dims = _build_dimensions(facts)
        assert [d.category for d in dims] == ["interest", "preference", "role"]

    def test_dominant_value_is_highest_confidence(self):
        facts = [
            make_fact("role", "past", "student", confidence=0.5),
            make_fact("role", "current", "entrepreneur", confidence=0.95),
        ]
        dims = _build_dimensions(facts)
        role_dim = dims[0]
        assert role_dim.dominant_value == "entrepreneur"
        assert role_dim.confidence == 0.95

    def test_facts_within_dimension_sorted_by_confidence(self):
        facts = [
            make_fact("interest", "a", "X", confidence=0.5),
            make_fact("interest", "b", "Y", confidence=0.9),
            make_fact("interest", "c", "Z", confidence=0.7),
        ]
        dims = _build_dimensions(facts)
        confidences = [f.confidence for f in dims[0].facts]
        assert confidences == sorted(confidences, reverse=True)

    def test_single_fact_creates_one_dimension(self):
        dims = _build_dimensions([make_fact("skill", "langraph", "expert")])
        assert len(dims) == 1
        assert dims[0].category == "skill"
        assert len(dims[0].facts) == 1

    def test_dimension_fact_count(self):
        facts = [
            make_fact("interest", "a", "X"),
            make_fact("interest", "b", "Y"),
            make_fact("interest", "c", "Z"),
        ]
        dims = _build_dimensions(facts)
        assert len(dims[0].facts) == 3


# ── _fallback_summary ─────────────────────────────────────────────────────────


class TestFallbackSummary:
    def test_empty_returns_empty_string(self):
        assert _fallback_summary([]) == ""

    def test_includes_category_and_dominant_value(self):
        dim = IdentityDimension(
            category="role",
            facts=[make_fact("role", "current", "entrepreneur")],
            dominant_value="entrepreneur",
            confidence=0.9,
        )
        result = _fallback_summary([dim])
        assert "role" in result
        assert "entrepreneur" in result

    def test_multiple_dimensions_joined(self):
        dims = [
            IdentityDimension("role", [], "engineer", 0.9),
            IdentityDimension("interest", [], "AI", 0.8),
        ]
        result = _fallback_summary(dims)
        assert "role=engineer" in result
        assert "interest=AI" in result


# ── UserIdentity ──────────────────────────────────────────────────────────────


class TestUserIdentity:
    def test_is_empty_with_no_dimensions(self):
        identity = UserIdentity(user_id="u1", app_id="default")
        assert identity.is_empty()

    def test_is_empty_false_with_dimensions(self):
        dim = IdentityDimension("role", [], "engineer", 0.9)
        identity = UserIdentity(user_id="u1", app_id="default", dimensions=[dim])
        assert not identity.is_empty()

    def test_computed_at_defaults_to_now(self):
        before = datetime.now(timezone.utc)
        identity = UserIdentity(user_id="u1", app_id="default")
        assert identity.computed_at >= before

    def test_as_prompt_text_empty(self):
        identity = UserIdentity(user_id="u1", app_id="default")
        text = identity.as_prompt_text()
        assert "## User Identity" in text
        assert "no identity data" in text

    def test_as_prompt_text_contains_header(self):
        dim = IdentityDimension(
            "role",
            [make_fact("role", "current", "entrepreneur")],
            "entrepreneur",
            0.9,
        )
        identity = UserIdentity(
            user_id="u1", app_id="default",
            dimensions=[dim],
            summary="An entrepreneur building AI tools.",
        )
        text = identity.as_prompt_text()
        assert "## User Identity" in text
        assert "Who they are" in text
        assert "entrepreneur" in text
        assert "An entrepreneur building AI tools." in text

    def test_as_prompt_text_shows_all_categories(self):
        dims = [
            IdentityDimension("role", [make_fact("role", "k", "engineer")], "engineer", 0.9),
            IdentityDimension("interest", [make_fact("interest", "k", "AI")], "AI", 0.8),
        ]
        identity = UserIdentity(user_id="u1", app_id="default", dimensions=dims)
        text = identity.as_prompt_text()
        assert "Role" in text
        assert "Interest" in text
        assert "engineer" in text
        assert "AI" in text

    def test_as_prompt_text_no_summary_section_when_empty(self):
        dim = IdentityDimension("role", [make_fact("role", "k", "v")], "v", 0.9)
        identity = UserIdentity(user_id="u1", app_id="default", dimensions=[dim], summary="")
        text = identity.as_prompt_text()
        assert "Who they are" not in text


# ── IdentityBuilder ───────────────────────────────────────────────────────────


class TestIdentityBuilder:
    @pytest.mark.asyncio
    async def test_returns_user_identity(self):
        facts = [make_fact("role", "current", "entrepreneur", confidence=0.95)]
        semantic = make_mock_semantic(facts)
        llm = make_mock_llm("An entrepreneur.")
        builder = IdentityBuilder(llm=llm, semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert isinstance(identity, UserIdentity)
        assert identity.user_id == "u1"

    @pytest.mark.asyncio
    async def test_empty_profile_returns_empty_identity(self):
        semantic = make_mock_semantic([])
        llm = make_mock_llm()
        builder = IdentityBuilder(llm=llm, semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert identity.is_empty()
        assert identity.total_facts == 0
        llm.extract_structured.assert_not_called()

    @pytest.mark.asyncio
    async def test_total_facts_matches_input(self):
        facts = [
            make_fact("role", "current", "entrepreneur"),
            make_fact("interest", "domain", "AI"),
            make_fact("preference", "ui_color", "green"),
        ]
        semantic = make_mock_semantic(facts)
        builder = IdentityBuilder(llm=make_mock_llm(), semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert identity.total_facts == 3

    @pytest.mark.asyncio
    async def test_dimensions_grouped_correctly(self):
        facts = [
            make_fact("role", "current", "entrepreneur"),
            make_fact("interest", "domain", "AI"),
            make_fact("interest", "topic", "LangGraph", confidence=0.7),
        ]
        semantic = make_mock_semantic(facts)
        builder = IdentityBuilder(llm=make_mock_llm(), semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert len(identity.dimensions) == 2
        interest_dim = next(d for d in identity.dimensions if d.category == "interest")
        assert len(interest_dim.facts) == 2

    @pytest.mark.asyncio
    async def test_llm_summary_used(self):
        facts = [make_fact("role", "current", "engineer")]
        semantic = make_mock_semantic(facts)
        llm = make_mock_llm("A seasoned software engineer.")
        builder = IdentityBuilder(llm=llm, semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert identity.summary == "A seasoned software engineer."

    @pytest.mark.asyncio
    async def test_llm_failure_uses_fallback(self):
        facts = [make_fact("role", "current", "entrepreneur")]
        semantic = make_mock_semantic(facts)
        llm = AsyncMock()
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("LLM down"))
        builder = IdentityBuilder(llm=llm, semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        # Fallback summary contains role=entrepreneur
        assert "role" in identity.summary
        assert "entrepreneur" in identity.summary

    @pytest.mark.asyncio
    async def test_app_id_forwarded_to_semantic(self):
        semantic = make_mock_semantic()
        builder = IdentityBuilder(llm=make_mock_llm(), semantic=semantic)
        neo = AsyncMock()

        await builder.build(neo, user_id="u1", app_id="my_app")

        call_kwargs = semantic.get_user_profile.call_args
        assert call_kwargs[0][1] == "u1"   # positional: user_id
        assert call_kwargs[0][2] == "my_app"  # positional: app_id

    @pytest.mark.asyncio
    async def test_none_profile_handled_gracefully(self):
        """SemanticMemory returns None → empty identity, no error."""
        from smritikosh.memory.semantic import SemanticMemory
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=None)
        builder = IdentityBuilder(llm=make_mock_llm(), semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")

        assert identity.is_empty()
        assert identity.total_facts == 0

    @pytest.mark.asyncio
    async def test_identity_prompt_text_renderable(self):
        facts = [
            make_fact("role", "current", "entrepreneur"),
            make_fact("interest", "domain", "AI agents"),
        ]
        semantic = make_mock_semantic(facts)
        builder = IdentityBuilder(llm=make_mock_llm("Entrepreneur building AI."), semantic=semantic)

        identity = await builder.build(AsyncMock(), user_id="u1")
        text = identity.as_prompt_text()

        assert "## User Identity" in text
        assert "entrepreneur" in text
        assert "AI agents" in text


# ── DB integration tests ──────────────────────────────────────────────────────


@pytest.mark.db
class TestIdentityBuilderDB:
    """
    How to run:
        docker compose up -d neo4j
        pytest tests/test_identity.py -v -m db
    """

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        from neo4j import AsyncGraphDatabase

        from smritikosh.config import settings
        from smritikosh.llm.adapter import LLMAdapter
        from smritikosh.memory.semantic import SemanticMemory

        self.driver = AsyncGraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        self.semantic = SemanticMemory()
        self.llm = LLMAdapter()
        self.builder = IdentityBuilder(llm=self.llm, semantic=self.semantic)

        yield

        async with self.driver.session() as session:
            await session.run("MATCH (u:User {user_id: $uid}) DETACH DELETE u", uid="u_id_test")
        await self.driver.close()

    async def test_identity_with_facts(self):
        async with self.driver.session() as session:
            await self.semantic.upsert_fact(
                session, user_id="u_id_test", app_id="default",
                category="role", key="current", value="entrepreneur", confidence=0.95,
            )
            await self.semantic.upsert_fact(
                session, user_id="u_id_test", app_id="default",
                category="interest", key="domain", value="AI agents", confidence=0.9,
            )

        async with self.driver.session() as session:
            identity = await self.builder.build(session, user_id="u_id_test")

        assert not identity.is_empty()
        assert identity.total_facts == 2
        categories = {d.category for d in identity.dimensions}
        assert "role" in categories
        assert "interest" in categories

    async def test_identity_empty_user(self):
        async with self.driver.session() as session:
            identity = await self.builder.build(session, user_id="nonexistent_user_xyz")

        assert identity.is_empty()
        assert identity.total_facts == 0
