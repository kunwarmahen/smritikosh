"""
API route tests for admin and ingest endpoints.

POST /admin/consolidate
POST /admin/prune
POST /admin/cluster
POST /admin/mine-beliefs
POST /admin/reconsolidate
POST /ingest/push
POST /ingest/file
POST /ingest/slack/events
POST /ingest/email/sync
POST /ingest/calendar
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.auth.deps import get_current_user, require_admin
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.hippocampus import EncodedMemory, Hippocampus
from smritikosh.memory.semantic import FactRecord
from smritikosh.db.models import Event
from smritikosh.processing.reconsolidation import ReconsolidationEngine, ReconsolidationResult
from smritikosh.processing.scheduler import MemoryScheduler


# ── Scheduler result helpers ──────────────────────────────────────────────────


def make_consolidation_result(user_id="u1", skipped=False):
    r = MagicMock()
    r.user_id = user_id
    r.app_id = "default"
    r.skipped = skipped
    r.skip_reason = "no events" if skipped else ""
    r.events_consolidated = 3
    r.facts_distilled = 5
    return r


def make_pruning_result(user_id="u1", skipped=False):
    r = MagicMock()
    r.user_id = user_id
    r.app_id = "default"
    r.skipped = skipped
    r.skip_reason = "" if not skipped else "no events"
    r.events_evaluated = 10
    r.events_pruned = 2
    return r


def make_clustering_result(user_id="u1", skipped=False):
    r = MagicMock()
    r.user_id = user_id
    r.app_id = "default"
    r.skipped = skipped
    r.skip_reason = ""
    r.clusters_found = 3
    r.events_clustered = 15
    return r


def make_belief_result(user_id="u1", skipped=False):
    r = MagicMock()
    r.user_id = user_id
    r.app_id = "default"
    r.skipped = skipped
    r.skip_reason = ""
    r.beliefs_upserted = 4
    return r


def make_event_obj(user_id="u1") -> Event:
    return Event(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        raw_text="Some interaction text",
        importance_score=0.8,
        consolidated=False,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )


def make_encoded_memory(user_id="u1") -> EncodedMemory:
    return EncodedMemory(
        event=make_event_obj(user_id),
        facts=[],
        importance_score=0.8,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pg():
    return AsyncMock()


@pytest.fixture
def mock_neo():
    return AsyncMock()


@pytest.fixture
def mock_scheduler():
    s = AsyncMock(spec=MemoryScheduler)
    s.run_consolidation_now = AsyncMock(return_value=make_consolidation_result())
    s.run_consolidation_for_all_users = AsyncMock(return_value=[make_consolidation_result()])
    s.run_pruning_now = AsyncMock(return_value=make_pruning_result())
    s.run_pruning_for_all_users = AsyncMock(return_value=[make_pruning_result()])
    s.run_clustering_now = AsyncMock(return_value=make_clustering_result())
    s.run_clustering_for_all_users = AsyncMock(return_value=[make_clustering_result()])
    s.run_belief_mining_now = AsyncMock(return_value=make_belief_result())
    s.run_belief_mining_for_all_users = AsyncMock(return_value=[make_belief_result()])
    return s


@pytest.fixture
def mock_reconsolidation_engine():
    return AsyncMock(spec=ReconsolidationEngine)


@pytest.fixture
def mock_hippocampus():
    return AsyncMock(spec=Hippocampus)


_ADMIN_PAYLOAD = {"sub": "admin", "role": "admin", "app_ids": ["default"]}


@pytest.fixture(autouse=True)
def override_deps(mock_pg, mock_neo, mock_scheduler,
                  mock_reconsolidation_engine, mock_hippocampus):
    app.dependency_overrides[get_session] = lambda: mock_pg
    app.dependency_overrides[get_neo4j_session] = lambda: mock_neo
    app.dependency_overrides[deps.get_reconsolidation_engine] = lambda: mock_reconsolidation_engine
    app.dependency_overrides[deps.get_hippocampus] = lambda: mock_hippocampus
    app.dependency_overrides[require_admin] = lambda: _ADMIN_PAYLOAD
    app.dependency_overrides[get_current_user] = lambda: _ADMIN_PAYLOAD
    # Inject scheduler onto app state
    app.state.scheduler = mock_scheduler
    yield
    app.dependency_overrides.clear()
    if hasattr(app.state, "scheduler"):
        del app.state.scheduler


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


VALID_UUID = str(uuid.uuid4())


# ── POST /admin/consolidate ───────────────────────────────────────────────────


class TestAdminConsolidate:
    @pytest.mark.asyncio
    async def test_returns_200_for_specific_user(self, client, mock_scheduler):
        response = await client.post("/admin/consolidate", json={"user_id": "u1"})
        assert response.status_code == 200
        body = response.json()
        assert body["job"] == "consolidation"
        assert body["users_processed"] == 1

    @pytest.mark.asyncio
    async def test_returns_200_for_all_users(self, client, mock_scheduler):
        mock_scheduler.run_consolidation_for_all_users = AsyncMock(
            return_value=[make_consolidation_result("u1"), make_consolidation_result("u2")]
        )
        response = await client.post("/admin/consolidate", json={})
        body = response.json()
        assert body["users_processed"] == 2

    @pytest.mark.asyncio
    async def test_result_shape(self, client, mock_scheduler):
        response = await client.post("/admin/consolidate", json={"user_id": "u1"})
        result = response.json()["results"][0]
        assert "user_id" in result
        assert "skipped" in result
        assert "detail" in result

    @pytest.mark.asyncio
    async def test_skipped_result_shows_reason(self, client, mock_scheduler):
        mock_scheduler.run_consolidation_now = AsyncMock(
            return_value=make_consolidation_result(skipped=True)
        )
        response = await client.post("/admin/consolidate", json={"user_id": "u1"})
        result = response.json()["results"][0]
        assert result["skipped"] is True
        assert "no events" in result["detail"]

    @pytest.mark.asyncio
    async def test_no_scheduler_returns_503(self, client):
        del app.state.scheduler
        response = await client.post("/admin/consolidate", json={})
        assert response.status_code == 503


# ── POST /admin/prune ─────────────────────────────────────────────────────────


class TestAdminPrune:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        response = await client.post("/admin/prune", json={"user_id": "u1"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_job_name_is_pruning(self, client):
        response = await client.post("/admin/prune", json={})
        assert response.json()["job"] == "pruning"

    @pytest.mark.asyncio
    async def test_detail_contains_pruned_count(self, client, mock_scheduler):
        mock_scheduler.run_pruning_now = AsyncMock(
            return_value=make_pruning_result()
        )
        response = await client.post("/admin/prune", json={"user_id": "u1"})
        detail = response.json()["results"][0]["detail"]
        assert "pruned=" in detail


# ── POST /admin/cluster ───────────────────────────────────────────────────────


class TestAdminCluster:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        response = await client.post("/admin/cluster", json={"user_id": "u1"})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_job_name_is_clustering(self, client):
        response = await client.post("/admin/cluster", json={})
        assert response.json()["job"] == "clustering"


# ── POST /admin/mine-beliefs ──────────────────────────────────────────────────


class TestAdminMineBeliefs:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        response = await client.post("/admin/mine-beliefs", json={})
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_job_name_is_belief_mining(self, client):
        response = await client.post("/admin/mine-beliefs", json={})
        assert response.json()["job"] == "belief_mining"

    @pytest.mark.asyncio
    async def test_detail_contains_beliefs_count(self, client, mock_scheduler):
        mock_scheduler.run_belief_mining_now = AsyncMock(
            return_value=make_belief_result()
        )
        response = await client.post("/admin/mine-beliefs", json={"user_id": "u1"})
        detail = response.json()["results"][0]["detail"]
        assert "beliefs_upserted=" in detail


# ── POST /admin/reconsolidate ─────────────────────────────────────────────────


class TestAdminReconsolidate:
    @pytest.mark.asyncio
    async def test_returns_200_on_success(self, client, mock_reconsolidation_engine):
        mock_reconsolidation_engine.reconsolidate_event = AsyncMock(
            return_value=ReconsolidationResult(
                event_id=VALID_UUID, user_id="u1",
                updated=True, new_summary="better summary", old_summary="old",
            )
        )
        response = await client.post("/admin/reconsolidate", json={
            "event_id": VALID_UUID,
            "query": "what am I building?",
            "user_id": "u1",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_reconsolidation_engine):
        mock_reconsolidation_engine.reconsolidate_event = AsyncMock(
            return_value=ReconsolidationResult(
                event_id=VALID_UUID, user_id="u1",
                updated=True, new_summary="refined", old_summary="original",
            )
        )
        response = await client.post("/admin/reconsolidate", json={
            "event_id": VALID_UUID,
            "query": "query",
            "user_id": "u1",
        })
        body = response.json()
        assert body["event_id"] == VALID_UUID
        assert body["updated"] is True
        assert body["new_summary"] == "refined"

    @pytest.mark.asyncio
    async def test_skipped_result_returned(self, client, mock_reconsolidation_engine):
        mock_reconsolidation_engine.reconsolidate_event = AsyncMock(
            return_value=ReconsolidationResult(
                event_id=VALID_UUID, user_id="u1",
                skipped=True, skip_reason="recall_count too low",
            )
        )
        response = await client.post("/admin/reconsolidate", json={
            "event_id": VALID_UUID,
            "query": "test",
            "user_id": "u1",
        })
        body = response.json()
        assert body["skipped"] is True
        assert "recall_count" in body["skip_reason"]

    @pytest.mark.asyncio
    async def test_missing_event_id_returns_422(self, client):
        response = await client.post("/admin/reconsolidate", json={
            "query": "test",
            "user_id": "u1",
        })
        assert response.status_code == 422


# ── POST /ingest/push ─────────────────────────────────────────────────────────


class TestIngestPush:
    @pytest.mark.asyncio
    async def test_returns_201_on_success(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        response = await client.post("/ingest/push", json={
            "user_id": "u1",
            "content": "user prefers dark mode",
        })
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        response = await client.post("/ingest/push", json={
            "user_id": "u1",
            "content": "user is building an AI startup",
            "source": "zapier",
        })
        body = response.json()
        assert body["source"] == "zapier"
        assert body["events_ingested"] == 1
        assert body["events_failed"] == 0
        assert len(body["event_ids"]) == 1

    @pytest.mark.asyncio
    async def test_empty_content_returns_422(self, client):
        response = await client.post("/ingest/push", json={
            "user_id": "u1",
            "content": "   ",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_content_returns_422(self, client):
        response = await client.post("/ingest/push", json={"user_id": "u1"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_hippocampus_encode_called(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        await client.post("/ingest/push", json={
            "user_id": "u1",
            "content": "I built a RAG pipeline",
        })
        mock_hippocampus.encode.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_hippocampus_failure_counts_as_failed(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(side_effect=RuntimeError("db down"))
        response = await client.post("/ingest/push", json={
            "user_id": "u1",
            "content": "some content",
        })
        # The route catches per-event exceptions — response still 201
        assert response.status_code == 201
        body = response.json()
        assert body["events_failed"] == 1
        assert body["events_ingested"] == 0


# ── POST /ingest/file ─────────────────────────────────────────────────────────


class TestIngestFile:
    @pytest.mark.asyncio
    async def test_txt_file_ingested(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        content = b"First paragraph with enough content to be worth storing.\n\nSecond paragraph also worth saving here."
        response = await client.post(
            "/ingest/file",
            data={"user_id": "u1"},
            files={"file": ("notes.txt", content, "text/plain")},
        )
        assert response.status_code == 201
        assert response.json()["events_ingested"] >= 1

    @pytest.mark.asyncio
    async def test_source_is_file(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        content = b"Meaningful paragraph with content worth encoding into memory."
        response = await client.post(
            "/ingest/file",
            data={"user_id": "u1"},
            files={"file": ("notes.txt", content, "text/plain")},
        )
        assert response.json()["source"] == "file"

    @pytest.mark.asyncio
    async def test_empty_file_returns_422(self, client):
        response = await client.post(
            "/ingest/file",
            data={"user_id": "u1"},
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_user_id_returns_422(self, client):
        response = await client.post(
            "/ingest/file",
            files={"file": ("notes.txt", b"content", "text/plain")},
        )
        assert response.status_code == 422


# ── POST /ingest/slack/events ─────────────────────────────────────────────────


class TestIngestSlack:
    @pytest.mark.asyncio
    async def test_url_verification_challenge_returned(self, client):
        """No signing secret configured → 501; but with mock we bypass verification."""
        # Without signing secret, endpoint returns 501
        response = await client.post(
            "/ingest/slack/events",
            content=json.dumps({"type": "url_verification", "challenge": "abc123"}),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 501  # SLACK_SIGNING_SECRET not set

    @pytest.mark.asyncio
    async def test_missing_signature_headers_returns_400(self, client):
        from smritikosh.config import settings
        original = settings.slack_signing_secret
        settings.slack_signing_secret = "testsecret"
        try:
            response = await client.post(
                "/ingest/slack/events",
                content=json.dumps({"type": "event_callback"}),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 400
        finally:
            settings.slack_signing_secret = original


# ── POST /ingest/calendar ─────────────────────────────────────────────────────


ICS_CONTENT = b"""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:test-001
SUMMARY:Strategy Meeting
DESCRIPTION:Discuss product roadmap
DTSTART:20240315T100000Z
DTEND:20240315T110000Z
END:VEVENT
END:VCALENDAR"""


class TestIngestCalendar:
    @pytest.mark.asyncio
    async def test_ics_file_ingested(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        response = await client.post(
            "/ingest/calendar",
            data={"user_id": "u1"},
            files={"file": ("calendar.ics", ICS_CONTENT, "text/calendar")},
        )
        assert response.status_code == 201
        assert response.json()["events_ingested"] == 1

    @pytest.mark.asyncio
    async def test_source_is_calendar(self, client, mock_hippocampus):
        mock_hippocampus.encode = AsyncMock(return_value=make_encoded_memory())
        response = await client.post(
            "/ingest/calendar",
            data={"user_id": "u1"},
            files={"file": ("work.ics", ICS_CONTENT, "text/calendar")},
        )
        assert response.json()["source"] == "calendar"

    @pytest.mark.asyncio
    async def test_empty_ics_returns_422(self, client):
        response = await client.post(
            "/ingest/calendar",
            data={"user_id": "u1"},
            files={"file": ("empty.ics", b"", "text/calendar")},
        )
        assert response.status_code == 422
