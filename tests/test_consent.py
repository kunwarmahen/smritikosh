"""
Tests for the cross-app memory consent layer (item S4).

Three surfaces:
  1. ConsentService — grant/revoke/upsert semantics, category validation,
     consented_facts (filtering, provenance tagging, audit, fault isolation)
  2. ContextBuilder merge — shared facts appended, own-app facts win collisions
  3. /consents routes — auth (self-or-admin, source-app access), status codes

All I/O mocked; no live databases.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api import deps
from smritikosh.api.main import app
from smritikosh.auth.deps import get_current_user
from smritikosh.db.models import MemoryConsent
from smritikosh.db.postgres import get_session
from smritikosh.memory.consent import ConsentError, ConsentService
from smritikosh.memory.semantic import FactRecord, UserProfile

# ── Fixtures ──────────────────────────────────────────────────────────────────


def make_fact(category="diet", key="restriction", value="vegetarian", confidence=0.9):
    return FactRecord(category, key, value, confidence, 1, "2026-07-01", "2026-07-01")


def make_consent(
    user_id="u1",
    source_app_id="app-a",
    target_app_id="app-b",
    categories=None,
    revoked_at=None,
) -> MemoryConsent:
    return MemoryConsent(
        id=uuid.uuid4(),
        user_id=user_id,
        source_app_id=source_app_id,
        target_app_id=target_app_id,
        categories=categories or [],
        granted_at=datetime.now(timezone.utc),
        revoked_at=revoked_at,
        created_by="u1",
    )


def pg_returning(scalar_one=None, scalars_all=None):
    """AsyncSession mock whose execute() resolves to the given rows."""
    result = MagicMock()
    result.scalar_one_or_none.return_value = scalar_one
    result.scalars.return_value.all.return_value = scalars_all or []
    pg = AsyncMock()
    pg.execute = AsyncMock(return_value=result)
    pg.add = MagicMock()
    return pg


# ── ConsentService: grants ────────────────────────────────────────────────────


class TestConsentGrant:
    async def test_grant_creates_row(self):
        service = ConsentService()
        pg = pg_returning(scalar_one=None)

        consent = await service.grant(
            pg, user_id="u1", source_app_id="app-a", target_app_id="app-b",
            categories=["diet", "health"], created_by="u1",
        )

        pg.add.assert_called_once()
        assert consent.categories == ["diet", "health"]
        assert consent.is_active

    async def test_grant_rejects_self_grant(self):
        service = ConsentService()
        with pytest.raises(ConsentError, match="must differ"):
            await service.grant(
                pg_returning(), user_id="u1", source_app_id="app-a",
                target_app_id="app-a", created_by="u1",
            )

    async def test_grant_rejects_unknown_category(self):
        service = ConsentService()
        with pytest.raises(ConsentError, match="Unknown fact categories"):
            await service.grant(
                pg_returning(), user_id="u1", source_app_id="app-a",
                target_app_id="app-b", categories=["diet", "nonsense"],
                created_by="u1",
            )

    async def test_regrant_reactivates_revoked_row(self):
        from datetime import datetime, timezone

        service = ConsentService()
        existing = make_consent(
            categories=["diet"], revoked_at=datetime.now(timezone.utc)
        )
        pg = pg_returning(scalar_one=existing)

        consent = await service.grant(
            pg, user_id="u1", source_app_id="app-a", target_app_id="app-b",
            categories=["preference"], created_by="admin",
        )

        assert consent is existing          # upsert, not a new row
        pg.add.assert_not_called()
        assert consent.is_active
        assert consent.categories == ["preference"]
        assert consent.created_by == "admin"

    async def test_grant_emits_audit_event(self):
        audit = AsyncMock()
        service = ConsentService(audit=audit)

        await service.grant(
            pg_returning(scalar_one=None), user_id="u1", source_app_id="app-a",
            target_app_id="app-b", created_by="u1",
        )

        event = audit.emit.await_args.args[0]
        assert event.event_type == "consent.granted"
        assert event.payload["source_app_id"] == "app-a"


class TestConsentRevoke:
    async def test_revoke_sets_revoked_at(self):
        service = ConsentService()
        existing = make_consent()
        pg = pg_returning(scalar_one=existing)

        assert await service.revoke(
            pg, user_id="u1", source_app_id="app-a", target_app_id="app-b"
        ) is True
        assert existing.revoked_at is not None

    async def test_revoke_absent_returns_false(self):
        service = ConsentService()
        assert await service.revoke(
            pg_returning(scalar_one=None), user_id="u1",
            source_app_id="app-a", target_app_id="app-b",
        ) is False

    async def test_revoke_already_revoked_returns_false(self):
        from datetime import datetime, timezone

        service = ConsentService()
        existing = make_consent(revoked_at=datetime.now(timezone.utc))
        assert await service.revoke(
            pg_returning(scalar_one=existing), user_id="u1",
            source_app_id="app-a", target_app_id="app-b",
        ) is False


# ── ConsentService: read-path enforcement ─────────────────────────────────────


class TestConsentedFacts:
    def _service(self, profile_by_app: dict, audit=None) -> ConsentService:
        semantic = AsyncMock()

        async def get_user_profile(neo, user_id, app_id, min_confidence=0.0):
            value = profile_by_app.get(app_id)
            if isinstance(value, Exception):
                raise value
            return value

        semantic.get_user_profile = AsyncMock(side_effect=get_user_profile)
        return ConsentService(semantic=semantic, audit=audit)

    async def test_filters_to_consented_categories_and_tags_provenance(self):
        profile = UserProfile(
            user_id="u1", app_id="app-a",
            facts=[make_fact(category="diet"), make_fact(category="finance", key="budget")],
        )
        service = self._service({"app-a": profile})
        pg = pg_returning(scalars_all=[make_consent(categories=["diet"])])

        shared = await service.consented_facts(
            pg, AsyncMock(), user_id="u1", target_app_id="app-b"
        )

        assert [f.category for f in shared] == ["diet"]
        assert shared[0].source_meta["shared_from_app"] == "app-a"

    async def test_empty_categories_means_all(self):
        profile = UserProfile(
            user_id="u1", app_id="app-a",
            facts=[make_fact(category="diet"), make_fact(category="finance", key="budget")],
        )
        service = self._service({"app-a": profile})
        pg = pg_returning(scalars_all=[make_consent(categories=[])])

        shared = await service.consented_facts(
            pg, AsyncMock(), user_id="u1", target_app_id="app-b"
        )
        assert len(shared) == 2

    async def test_audits_every_cross_app_read(self):
        audit = AsyncMock()
        profile = UserProfile(user_id="u1", app_id="app-a", facts=[make_fact()])
        service = self._service({"app-a": profile}, audit=audit)
        pg = pg_returning(scalars_all=[make_consent(categories=["diet"])])

        await service.consented_facts(
            pg, AsyncMock(), user_id="u1", target_app_id="app-b"
        )

        event = audit.emit.await_args.args[0]
        assert event.event_type == "consent.cross_app_read"
        assert event.payload == {
            "source_app_id": "app-a",
            "target_app_id": "app-b",
            "categories": ["diet"],
            "facts_returned": 1,
        }

    async def test_source_failure_does_not_break_other_sources(self):
        profile_c = UserProfile(user_id="u1", app_id="app-c", facts=[make_fact()])
        service = self._service({"app-a": RuntimeError("neo down"), "app-c": profile_c})
        pg = pg_returning(scalars_all=[
            make_consent(source_app_id="app-a"),
            make_consent(source_app_id="app-c"),
        ])

        shared = await service.consented_facts(
            pg, AsyncMock(), user_id="u1", target_app_id="app-b"
        )
        assert len(shared) == 1
        assert shared[0].source_meta["shared_from_app"] == "app-c"

    async def test_no_semantic_returns_empty(self):
        service = ConsentService(semantic=None)
        assert await service.consented_facts(
            pg_returning(), AsyncMock(), user_id="u1", target_app_id="app-b"
        ) == []


# ── ContextBuilder merge ──────────────────────────────────────────────────────


class TestContextBuilderConsentMerge:
    def _build_args(self):
        return dict(user_id="u1", query="what do I eat?", app_ids=["app-b"])

    def _builder(self, own_profile, shared_facts):
        from smritikosh.memory.episodic import EpisodicMemory
        from smritikosh.memory.semantic import SemanticMemory
        from smritikosh.retrieval.context_builder import ContextBuilder

        llm = AsyncMock()
        llm.embed = AsyncMock(return_value=[0.1] * 8)
        episodic = AsyncMock(spec=EpisodicMemory)
        episodic.hybrid_search = AsyncMock(return_value=[])
        episodic.get_recent = AsyncMock(return_value=[])
        semantic = AsyncMock(spec=SemanticMemory)
        semantic.get_user_profile = AsyncMock(return_value=own_profile)
        consent = AsyncMock(spec=ConsentService)
        consent.consented_facts = AsyncMock(return_value=shared_facts)
        return ContextBuilder(
            llm=llm, episodic=episodic, semantic=semantic, consent=consent
        ), consent

    async def test_shared_facts_merged_into_profile(self):
        own = UserProfile(user_id="u1", app_id="app-b", facts=[make_fact(category="role", key="current", value="founder")])
        shared = [make_fact(category="diet")]
        builder, consent = self._builder(own, shared)

        ctx = await builder.build(AsyncMock(), AsyncMock(), **self._build_args())

        categories = {f.category for f in ctx.user_profile.facts}
        assert categories == {"role", "diet"}
        consent.consented_facts.assert_awaited_once()
        assert consent.consented_facts.await_args.kwargs["target_app_id"] == "app-b"

    async def test_own_app_fact_wins_collision(self):
        own = UserProfile(user_id="u1", app_id="app-b", facts=[make_fact(value="vegan")])
        shared = [make_fact(value="vegetarian")]  # same (category, key)
        builder, _ = self._builder(own, shared)

        ctx = await builder.build(AsyncMock(), AsyncMock(), **self._build_args())

        diet_facts = [f for f in ctx.user_profile.facts if f.category == "diet"]
        assert len(diet_facts) == 1
        assert diet_facts[0].value == "vegan"

    async def test_profile_created_when_target_app_has_none(self):
        builder, _ = self._builder(None, [make_fact()])

        ctx = await builder.build(AsyncMock(), AsyncMock(), **self._build_args())

        assert ctx.user_profile is not None
        assert ctx.user_profile.facts[0].category == "diet"

    async def test_consent_failure_degrades_gracefully(self):
        own = UserProfile(user_id="u1", app_id="app-b", facts=[make_fact()])
        builder, consent = self._builder(own, [])
        consent.consented_facts = AsyncMock(side_effect=RuntimeError("pg down"))

        ctx = await builder.build(AsyncMock(), AsyncMock(), **self._build_args())

        assert len(ctx.user_profile.facts) == 1  # own profile untouched


# ── /consents routes ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_consent_service():
    return AsyncMock(spec=ConsentService)


@pytest.fixture
def current_user():
    return {"sub": "u1", "role": "user", "app_ids": ["app-a", "app-b"]}


@pytest.fixture
def client(mock_consent_service, current_user):
    app.dependency_overrides[get_session] = lambda: AsyncMock()
    app.dependency_overrides[deps.get_consent_service] = lambda: mock_consent_service
    app.dependency_overrides[get_current_user] = lambda: current_user
    transport = ASGITransport(app=app)
    yield AsyncClient(transport=transport, base_url="http://test")
    app.dependency_overrides.clear()


GRANT_BODY = {
    "user_id": "u1",
    "source_app_id": "app-a",
    "target_app_id": "app-b",
    "categories": ["diet"],
}


class TestConsentRoutes:
    async def test_grant_returns_201(self, client, mock_consent_service):
        mock_consent_service.grant = AsyncMock(
            return_value=make_consent(categories=["diet"])
        )
        async with client as ac:
            response = await ac.post("/consents", json=GRANT_BODY)

        assert response.status_code == 201
        body = response.json()
        assert body["source_app_id"] == "app-a"
        assert body["active"] is True
        assert mock_consent_service.grant.await_args.kwargs["created_by"] == "u1"

    async def test_grant_forbidden_for_other_user(self, client):
        async with client as ac:
            response = await ac.post("/consents", json={**GRANT_BODY, "user_id": "someone-else"})
        assert response.status_code == 403

    async def test_grant_forbidden_without_source_app_access(self, client, current_user):
        current_user["app_ids"] = ["app-b"]  # no access to source app-a
        async with client as ac:
            response = await ac.post("/consents", json=GRANT_BODY)
        assert response.status_code == 403

    async def test_grant_maps_consent_error_to_422(self, client, mock_consent_service):
        mock_consent_service.grant = AsyncMock(side_effect=ConsentError("bad category"))
        async with client as ac:
            response = await ac.post("/consents", json=GRANT_BODY)
        assert response.status_code == 422

    async def test_revoke_returns_404_when_absent(self, client, mock_consent_service):
        mock_consent_service.revoke = AsyncMock(return_value=False)
        async with client as ac:
            response = await ac.request("DELETE", "/consents", json={
                "user_id": "u1", "source_app_id": "app-a", "target_app_id": "app-b",
            })
        assert response.status_code == 404

    async def test_revoke_success(self, client, mock_consent_service):
        mock_consent_service.revoke = AsyncMock(return_value=True)
        async with client as ac:
            response = await ac.request("DELETE", "/consents", json={
                "user_id": "u1", "source_app_id": "app-a", "target_app_id": "app-b",
            })
        assert response.status_code == 200
        assert response.json()["revoked"] is True

    async def test_list_consents(self, client, mock_consent_service):
        mock_consent_service.list_for_user = AsyncMock(
            return_value=[make_consent(categories=["diet"])]
        )
        async with client as ac:
            response = await ac.get("/consents/u1")

        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "u1"
        assert body["consents"][0]["target_app_id"] == "app-b"

    async def test_list_forbidden_for_other_user(self, client):
        async with client as ac:
            response = await ac.get("/consents/someone-else")
        assert response.status_code == 403
