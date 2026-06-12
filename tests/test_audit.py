"""Tests for the audit trail (item B2 / Gap 5).

Covers:
- AuditEvent.to_document — serialisation shape
- AuditLogger — fire-and-forget emit, error swallowing, query helpers
- get_audit_collection — disabled when MONGODB_URL is unset
- /audit routes — 503 guard without MongoDB, happy path with a mock logger
"""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.audit.logger import AuditEvent, AuditLogger, EventType


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_event(**overrides) -> AuditEvent:
    base = dict(
        event_type=EventType.MEMORY_ENCODED,
        user_id="u1",
        app_id="default",
        payload={"importance_score": 0.7},
    )
    base.update(overrides)
    return AuditEvent(**base)


def make_find_collection(docs: list[dict]) -> MagicMock:
    """Mock motor collection whose find() chain resolves to `docs`."""
    col = MagicMock()
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=docs)
    col.find.return_value = cursor
    col.insert_one = AsyncMock()
    return col


class AsyncIter:
    """Minimal async iterator, stands in for a motor aggregate cursor."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


# ── AuditEvent ────────────────────────────────────────────────────────────────


class TestAuditEvent:
    def test_to_document_shape(self):
        e = make_event(event_id="abc", session_id="s1")
        doc = e.to_document()
        assert doc["_id"] == e.id
        assert doc["event_type"] == EventType.MEMORY_ENCODED
        assert doc["user_id"] == "u1"
        assert doc["app_id"] == "default"
        assert doc["event_id"] == "abc"
        assert doc["session_id"] == "s1"
        assert doc["payload"] == {"importance_score": 0.7}

    def test_default_id_is_uuid(self):
        e = make_event()
        uuid.UUID(e.id)  # raises if not a valid UUID

    def test_default_timestamp_is_utc(self):
        e = make_event()
        assert e.timestamp.tzinfo is not None
        assert e.timestamp.utcoffset().total_seconds() == 0


# ── AuditLogger writes ────────────────────────────────────────────────────────


class TestAuditLoggerWrite:
    @pytest.mark.asyncio
    async def test_emit_sync_inserts_document(self):
        col = make_find_collection([])
        audit = AuditLogger(col)
        event = make_event()

        await audit.emit_sync(event)

        col.insert_one.assert_awaited_once()
        doc = col.insert_one.call_args.args[0]
        assert doc["_id"] == event.id

    @pytest.mark.asyncio
    async def test_emit_is_fire_and_forget(self):
        """emit() returns immediately; the write happens on a background task."""
        col = make_find_collection([])
        audit = AuditLogger(col)

        await audit.emit(make_event())
        # Yield to the event loop so the scheduled task runs.
        await asyncio.sleep(0)

        col.insert_one.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_failure_is_swallowed(self, caplog):
        """Mongo being down must never propagate into the pipeline."""
        col = make_find_collection([])
        col.insert_one = AsyncMock(side_effect=RuntimeError("mongo down"))
        audit = AuditLogger(col)

        await audit.emit_sync(make_event())  # must not raise

        assert any("Audit write failed" in r.message for r in caplog.records)


# ── AuditLogger queries ───────────────────────────────────────────────────────


def _stored_doc(**overrides) -> dict:
    base = dict(
        _id=str(uuid.uuid4()),
        event_type=EventType.MEMORY_ENCODED,
        user_id="u1",
        app_id="default",
        event_id=None,
        session_id=None,
        timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        payload={},
    )
    base.update(overrides)
    return base


class TestAuditLoggerQueries:
    @pytest.mark.asyncio
    async def test_get_timeline_filters_user_and_app(self):
        col = make_find_collection([_stored_doc()])
        audit = AuditLogger(col)

        records = await audit.get_timeline("u1", app_id="myapp")

        filt = col.find.call_args.args[0]
        assert filt == {"user_id": "u1", "app_id": "myapp"}
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_timeline_optional_filters(self):
        col = make_find_collection([])
        audit = AuditLogger(col)
        from_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
        to_ts = datetime(2026, 5, 31, tzinfo=timezone.utc)

        await audit.get_timeline(
            "u1",
            event_type=EventType.SEARCH_PERFORMED,
            event_id="e1",
            from_ts=from_ts,
            to_ts=to_ts,
        )

        filt = col.find.call_args.args[0]
        assert filt["event_type"] == EventType.SEARCH_PERFORMED
        assert filt["event_id"] == "e1"
        assert filt["timestamp"] == {"$gte": from_ts, "$lte": to_ts}

    @pytest.mark.asyncio
    async def test_get_timeline_renames_id_and_normalises_timestamp(self):
        doc = _stored_doc()
        original_id = doc["_id"]
        col = make_find_collection([doc])
        audit = AuditLogger(col)

        records = await audit.get_timeline("u1")

        assert records[0]["id"] == original_id
        assert "_id" not in records[0]
        assert records[0]["timestamp"] == "2026-05-01T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_get_event_lineage_filters_event_id(self):
        col = make_find_collection([_stored_doc(event_id="e1")])
        audit = AuditLogger(col)

        records = await audit.get_event_lineage("e1")

        filt = col.find.call_args.args[0]
        assert filt == {"event_id": "e1"}
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_stats_counts_per_event_type(self):
        col = make_find_collection([])
        col.aggregate.return_value = AsyncIter([
            {"_id": EventType.MEMORY_ENCODED, "count": 5},
            {"_id": EventType.SEARCH_PERFORMED, "count": 2},
        ])
        audit = AuditLogger(col)

        stats = await audit.get_stats("u1")

        assert stats == {
            EventType.MEMORY_ENCODED: 5,
            EventType.SEARCH_PERFORMED: 2,
        }


# ── get_audit_collection ──────────────────────────────────────────────────────


class TestGetAuditCollection:
    def test_returns_none_when_mongodb_unset(self, monkeypatch):
        from smritikosh.config import settings

        monkeypatch.setattr(settings, "mongodb_url", None)
        from smritikosh.audit.mongodb import get_audit_collection

        assert get_audit_collection() is None

    def test_deps_logger_none_when_mongodb_unset(self, monkeypatch):
        from smritikosh.config import settings

        monkeypatch.setattr(settings, "mongodb_url", None)
        from smritikosh.api.deps import get_audit_logger

        assert get_audit_logger() is None


# ── /audit routes ─────────────────────────────────────────────────────────────


_ADMIN_PAYLOAD = {"sub": "admin", "role": "admin", "app_ids": ["default"]}


@pytest.fixture
def audit_app():
    from smritikosh.api.main import app
    from smritikosh.auth.deps import get_current_user
    from smritikosh.db.postgres import get_session

    mock_pg = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_pg.execute = AsyncMock(return_value=result)

    app.dependency_overrides[get_session] = lambda: mock_pg
    app.dependency_overrides[get_current_user] = lambda: _ADMIN_PAYLOAD
    yield app
    app.dependency_overrides.clear()


class TestAuditRoutes:
    @pytest.mark.asyncio
    async def test_timeline_503_when_not_configured(self, audit_app):
        from smritikosh.api.deps import get_audit_logger

        audit_app.dependency_overrides[get_audit_logger] = lambda: None
        async with AsyncClient(
            transport=ASGITransport(app=audit_app), base_url="http://test"
        ) as client:
            resp = await client.get("/audit/u1")

        assert resp.status_code == 503
        assert "MONGODB_URL" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_timeline_returns_records(self, audit_app):
        from smritikosh.api.deps import get_audit_logger

        mock_audit = AsyncMock(spec=AuditLogger)
        mock_audit.get_timeline = AsyncMock(return_value=[{"id": "r1"}])
        audit_app.dependency_overrides[get_audit_logger] = lambda: mock_audit

        async with AsyncClient(
            transport=ASGITransport(app=audit_app), base_url="http://test"
        ) as client:
            resp = await client.get("/audit/u1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["records"] == [{"id": "r1"}]

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, audit_app):
        from smritikosh.api.deps import get_audit_logger

        mock_audit = AsyncMock(spec=AuditLogger)
        mock_audit.get_stats = AsyncMock(
            return_value={EventType.MEMORY_ENCODED: 3}
        )
        audit_app.dependency_overrides[get_audit_logger] = lambda: mock_audit

        async with AsyncClient(
            transport=ASGITransport(app=audit_app), base_url="http://test"
        ) as client:
            resp = await client.get("/audit/stats/u1")

        assert resp.status_code == 200
        assert resp.json()["total"] == 3
