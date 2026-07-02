"""Tests for the durable task queue (item A3).

Covers:
- queue_enabled / redis_settings / enqueue — Redis gating and graceful fallback
- _process_media_record — not-found and already-processed paths
- ARQ task wrappers + WorkerSettings wiring
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.tasks import jobs, queue


def _acm(value):
    """Build an async context manager yielding `value` (mocks db_session())."""

    @asynccontextmanager
    async def _cm(*args, **kwargs):
        yield value

    return _cm


class TestQueueGating:
    def test_queue_disabled_without_redis_url(self, monkeypatch):
        monkeypatch.setattr(queue.settings, "redis_url", None)
        assert queue.queue_enabled() is False

    def test_queue_enabled_with_redis_url(self, monkeypatch):
        monkeypatch.setattr(queue.settings, "redis_url", "redis://localhost:6379/0")
        assert queue.queue_enabled() is True

    def test_redis_settings_raises_without_url(self, monkeypatch):
        monkeypatch.setattr(queue.settings, "redis_url", None)
        with pytest.raises(RuntimeError, match="REDIS_URL is not configured"):
            queue.redis_settings()

    def test_redis_settings_built_from_url(self, monkeypatch):
        monkeypatch.setattr(queue.settings, "redis_url", "redis://localhost:6379/2")
        rs = queue.redis_settings()
        assert rs.host == "localhost"
        assert rs.port == 6379

    async def test_enqueue_returns_none_when_queue_disabled(self, monkeypatch):
        # No Redis → enqueue returns None so the caller runs the work in-process.
        monkeypatch.setattr(queue.settings, "redis_url", None)
        assert await queue.enqueue("process_media", "abc") is None

    async def test_enqueue_returns_none_on_pool_failure(self, monkeypatch):
        monkeypatch.setattr(queue.settings, "redis_url", "redis://localhost:6379/0")
        monkeypatch.setattr(queue, "get_pool", AsyncMock(side_effect=OSError("redis down")))
        # A failed enqueue must degrade to None, not raise.
        assert await queue.enqueue("process_media", "abc") is None


class TestProcessMediaRecord:
    async def test_returns_not_found_for_missing_record(self):
        pg = AsyncMock()
        pg.get = AsyncMock(return_value=None)
        with patch("smritikosh.db.postgres.db_session", _acm(pg)):
            result = await jobs._process_media_record("00000000-0000-0000-0000-000000000000")
        assert result == "not_found"

    async def test_returns_skipped_when_bytes_already_cleared(self):
        # raw_file is None → already processed (or never stored); nothing to do.
        ingest = MagicMock(raw_file=None, user_id="u", app_id="default", content_type="document")
        pg = AsyncMock()
        pg.get = AsyncMock(return_value=ingest)
        with patch("smritikosh.db.postgres.db_session", _acm(pg)):
            result = await jobs._process_media_record("11111111-1111-1111-1111-111111111111")
        assert result == "skipped"


class TestArqWrappers:
    async def test_process_media_delegates_to_helper(self):
        with patch.object(jobs, "_process_media_record", AsyncMock(return_value="complete")) as h:
            result = await jobs.process_media(ctx={}, media_id="abc")
        assert result == "complete"
        h.assert_awaited_once_with("abc")

    async def test_re_embed_events_delegates_to_helper(self):
        payload = {"success": 3, "errors": 0, "total": 3}
        with patch.object(jobs, "_re_embed_stale_events", AsyncMock(return_value=payload)) as h:
            result = await jobs.re_embed_events(ctx={})
        assert result == payload
        h.assert_awaited_once()

    async def test_reconsolidate_recalled_delegates_to_helper(self):
        payload = {"evaluated": 1, "updated": 1, "skipped": 0}
        with patch.object(jobs, "_reconsolidate_recalled", AsyncMock(return_value=payload)) as h:
            result = await jobs.reconsolidate_recalled(
                ctx={}, event_ids=["e1"], query="q", user_id="u1", app_id="a1"
            )
        assert result == payload
        h.assert_awaited_once_with(["e1"], "q", "u1", "a1")


class TestReconsolidateRecalledHelper:
    async def test_runs_engine_by_ids(self):
        from types import SimpleNamespace

        engine = AsyncMock()
        engine.reconsolidate_after_recall_by_ids = AsyncMock(
            return_value=SimpleNamespace(
                events_evaluated=2, events_updated=1, events_skipped=1
            )
        )
        with patch("smritikosh.api.deps.get_reconsolidation_engine", return_value=engine):
            result = await jobs._reconsolidate_recalled(["e1", "e2"], "q", "u1", "default")

        assert result == {"evaluated": 2, "updated": 1, "skipped": 1}
        engine.reconsolidate_after_recall_by_ids.assert_awaited_once_with(
            ["e1", "e2"], "q", "u1", "default"
        )


class TestWorkerSettings:
    def test_worker_registers_all_tasks(self):
        names = {f.__name__ for f in jobs.WorkerSettings.functions}
        assert names == {
            "process_media",
            "re_embed_events",
            "reconsolidate_recalled",
            "rotate_connector_tokens",
        }

    def test_worker_has_retry_and_timeout(self):
        assert jobs.WorkerSettings.max_tries >= 1
        assert jobs.WorkerSettings.job_timeout >= 300
