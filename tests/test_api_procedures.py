"""
API route tests for procedural memory endpoints.

POST   /procedures
GET    /procedures/{user_id}
PATCH  /procedures/{procedure_id}
DELETE /procedures/{procedure_id}
DELETE /procedures/user/{user_id}
DELETE /memory/event/{event_id}
DELETE /memory/user/{user_id}
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.db.models import UserProcedure
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.procedural import ProceduralMemory


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_procedure(
    trigger: str = "LLM deployment",
    instruction: str = "mention GPU optimization",
    user_id: str = "u1",
    priority: int = 5,
    is_active: bool = True,
) -> UserProcedure:
    p = UserProcedure(
        id=uuid.uuid4(),
        user_id=user_id,
        app_id="default",
        trigger=trigger,
        instruction=instruction,
        category="topic_response",
        priority=priority,
        is_active=is_active,
        hit_count=0,
        confidence=1.0,
        source="manual",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    return p


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pg():
    return AsyncMock()


@pytest.fixture
def mock_neo():
    return AsyncMock()


@pytest.fixture
def mock_procedural():
    return AsyncMock(spec=ProceduralMemory)


@pytest.fixture
def mock_episodic():
    return AsyncMock(spec=EpisodicMemory)


@pytest.fixture(autouse=True)
def override_deps(mock_pg, mock_neo, mock_procedural, mock_episodic):
    app.dependency_overrides[get_session] = lambda: mock_pg
    app.dependency_overrides[get_neo4j_session] = lambda: mock_neo
    app.dependency_overrides[deps.get_procedural] = lambda: mock_procedural
    app.dependency_overrides[deps.get_episodic] = lambda: mock_episodic
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


VALID_UUID = str(uuid.uuid4())


# ── POST /procedures ──────────────────────────────────────────────────────────


class TestCreateProcedure:
    @pytest.mark.asyncio
    async def test_returns_201_on_success(self, client, mock_procedural):
        mock_procedural.store = AsyncMock(return_value=make_procedure())
        response = await client.post("/procedures", json={
            "user_id": "u1",
            "trigger": "LLM deployment",
            "instruction": "mention GPU optimization",
        })
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_response_shape(self, client, mock_procedural):
        proc = make_procedure()
        mock_procedural.store = AsyncMock(return_value=proc)
        response = await client.post("/procedures", json={
            "user_id": "u1",
            "trigger": "LLM deployment",
            "instruction": "mention GPU optimization",
        })
        body = response.json()
        assert "procedure_id" in body
        assert body["user_id"] == "u1"
        assert body["trigger"] == "LLM deployment"
        assert body["is_active"] is True
        assert isinstance(body["priority"], int)

    @pytest.mark.asyncio
    async def test_missing_trigger_returns_422(self, client):
        response = await client.post("/procedures", json={
            "user_id": "u1",
            "instruction": "do something",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_instruction_returns_422(self, client):
        response = await client.post("/procedures", json={
            "user_id": "u1",
            "trigger": "startup",
        })
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_custom_priority_forwarded(self, client, mock_procedural):
        mock_procedural.store = AsyncMock(return_value=make_procedure(priority=9))
        response = await client.post("/procedures", json={
            "user_id": "u1",
            "trigger": "t",
            "instruction": "i",
            "priority": 9,
        })
        assert response.json()["priority"] == 9


# ── GET /procedures/{user_id} ─────────────────────────────────────────────────


class TestListProcedures:
    @pytest.mark.asyncio
    async def test_returns_200_with_list(self, client, mock_procedural):
        mock_procedural.get_all = AsyncMock(return_value=[make_procedure()])
        response = await client.get("/procedures/u1")
        assert response.status_code == 200
        body = response.json()
        assert body["user_id"] == "u1"
        assert len(body["procedures"]) == 1

    @pytest.mark.asyncio
    async def test_procedure_item_shape(self, client, mock_procedural):
        mock_procedural.get_all = AsyncMock(return_value=[make_procedure()])
        response = await client.get("/procedures/u1")
        item = response.json()["procedures"][0]
        assert "procedure_id" in item
        assert "trigger" in item
        assert "instruction" in item
        assert "priority" in item
        assert "is_active" in item

    @pytest.mark.asyncio
    async def test_empty_list(self, client, mock_procedural):
        mock_procedural.get_all = AsyncMock(return_value=[])
        response = await client.get("/procedures/u1")
        assert response.json()["procedures"] == []

    @pytest.mark.asyncio
    async def test_active_only_param_forwarded(self, client, mock_procedural):
        mock_procedural.get_all = AsyncMock(return_value=[])
        await client.get("/procedures/u1?active_only=false")
        call_kwargs = mock_procedural.get_all.call_args.kwargs
        assert call_kwargs["active_only"] is False

    @pytest.mark.asyncio
    async def test_category_filter_forwarded(self, client, mock_procedural):
        mock_procedural.get_all = AsyncMock(return_value=[])
        await client.get("/procedures/u1?category=communication")
        call_kwargs = mock_procedural.get_all.call_args.kwargs
        assert call_kwargs["category"] == "communication"


# ── PATCH /procedures/{id} ────────────────────────────────────────────────────


class TestUpdateProcedure:
    @pytest.mark.asyncio
    async def test_returns_200_on_success(self, client, mock_procedural):
        mock_procedural.update = AsyncMock(return_value=make_procedure())
        response = await client.patch(f"/procedures/{VALID_UUID}", json={
            "trigger": "new trigger",
        })
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_404_when_not_found(self, client, mock_procedural):
        mock_procedural.update = AsyncMock(return_value=None)
        response = await client.patch(f"/procedures/{VALID_UUID}", json={"trigger": "x"})
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_422(self, client):
        response = await client.patch("/procedures/not-a-uuid", json={"trigger": "x"})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_deactivate_via_patch(self, client, mock_procedural):
        proc = make_procedure(is_active=False)
        mock_procedural.update = AsyncMock(return_value=proc)
        response = await client.patch(f"/procedures/{VALID_UUID}", json={"is_active": False})
        assert response.json()["is_active"] is False


# ── DELETE /procedures/{id} ───────────────────────────────────────────────────


class TestDeleteProcedure:
    @pytest.mark.asyncio
    async def test_returns_deleted_true(self, client, mock_procedural):
        mock_procedural.delete = AsyncMock(return_value=True)
        response = await client.delete(f"/procedures/{VALID_UUID}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_returns_deleted_false_when_not_found(self, client, mock_procedural):
        mock_procedural.delete = AsyncMock(return_value=False)
        response = await client.delete(f"/procedures/{VALID_UUID}")
        assert response.json()["deleted"] is False

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_422(self, client):
        response = await client.delete("/procedures/not-a-uuid")
        assert response.status_code == 422


# ── DELETE /procedures/user/{user_id} ─────────────────────────────────────────


class TestDeleteUserProcedures:
    @pytest.mark.asyncio
    async def test_returns_count(self, client, mock_procedural):
        mock_procedural.delete_all_for_user = AsyncMock(return_value=3)
        response = await client.delete("/procedures/user/u1")
        assert response.status_code == 200
        body = response.json()
        assert body["procedures_deleted"] == 3
        assert body["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_zero_when_none_exist(self, client, mock_procedural):
        mock_procedural.delete_all_for_user = AsyncMock(return_value=0)
        response = await client.delete("/procedures/user/u1")
        assert response.json()["procedures_deleted"] == 0


# ── DELETE /memory/event/{event_id} ───────────────────────────────────────────


class TestDeleteEvent:
    @pytest.mark.asyncio
    async def test_returns_deleted_true(self, client, mock_episodic):
        mock_episodic.delete = AsyncMock(return_value=True)
        response = await client.delete(f"/memory/event/{VALID_UUID}")
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    @pytest.mark.asyncio
    async def test_returns_deleted_false_when_not_found(self, client, mock_episodic):
        mock_episodic.delete = AsyncMock(return_value=False)
        response = await client.delete(f"/memory/event/{VALID_UUID}")
        assert response.json()["deleted"] is False

    @pytest.mark.asyncio
    async def test_invalid_uuid_returns_422(self, client):
        response = await client.delete("/memory/event/not-a-uuid")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_event_id_in_response(self, client, mock_episodic):
        mock_episodic.delete = AsyncMock(return_value=True)
        response = await client.delete(f"/memory/event/{VALID_UUID}")
        assert response.json()["event_id"] == VALID_UUID


# ── DELETE /memory/user/{user_id} ─────────────────────────────────────────────


class TestDeleteUserMemory:
    @pytest.mark.asyncio
    async def test_returns_count(self, client, mock_episodic):
        mock_episodic.delete_all_for_user = AsyncMock(return_value=5)
        response = await client.delete("/memory/user/u1")
        assert response.status_code == 200
        body = response.json()
        assert body["events_deleted"] == 5
        assert body["user_id"] == "u1"

    @pytest.mark.asyncio
    async def test_app_id_forwarded(self, client, mock_episodic):
        mock_episodic.delete_all_for_user = AsyncMock(return_value=0)
        await client.delete("/memory/user/u1?app_id=myapp")
        call_args = mock_episodic.delete_all_for_user.call_args
        assert call_args.args[2] == "myapp" or call_args.kwargs.get("app_id") == "myapp"
