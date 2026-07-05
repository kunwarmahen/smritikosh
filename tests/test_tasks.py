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

    async def test_re_embed_events_creates_migration_and_runs_chunk(self):
        # No migration_id → create-or-resume, then process one chunk (H1).
        with (
            patch.object(
                jobs, "_create_or_resume_migration",
                AsyncMock(return_value=("mig-1", 3, False)),
            ) as create,
            patch.object(
                jobs, "_run_embedding_migration_chunk",
                AsyncMock(return_value="complete"),
            ) as chunk,
        ):
            result = await jobs.re_embed_events(ctx={})
        assert result == {"migration_id": "mig-1", "outcome": "complete"}
        create.assert_awaited_once()
        chunk.assert_awaited_once_with("mig-1")

    async def test_re_embed_events_reenqueues_next_chunk(self):
        # A non-terminal chunk re-enqueues itself with the same migration id.
        with (
            patch.object(
                jobs, "_run_embedding_migration_chunk",
                AsyncMock(return_value="continue"),
            ),
            patch("smritikosh.tasks.queue.enqueue", AsyncMock(return_value=object())) as enq,
        ):
            result = await jobs.re_embed_events(ctx={}, migration_id="mig-2")
        assert result == {"migration_id": "mig-2", "outcome": "continue"}
        enq.assert_awaited_once_with("re_embed_events", "mig-2")

    async def test_re_embed_events_falls_back_inline_when_queue_gone(self):
        # If re-enqueueing fails (Redis outage), the migration finishes inline.
        with (
            patch.object(
                jobs, "_run_embedding_migration_chunk",
                AsyncMock(return_value="continue"),
            ),
            patch("smritikosh.tasks.queue.enqueue", AsyncMock(return_value=None)),
            patch.object(
                jobs, "_run_embedding_migration_inline",
                AsyncMock(return_value={"migration_id": "mig-3", "outcome": "complete", "chunks": 2}),
            ) as inline,
        ):
            result = await jobs.re_embed_events(ctx={}, migration_id="mig-3")
        assert result["outcome"] == "complete"
        inline.assert_awaited_once_with("mig-3")

    async def test_inline_runner_loops_until_terminal(self):
        outcomes = iter(["continue", "continue", "complete"])
        with patch.object(
            jobs, "_run_embedding_migration_chunk",
            AsyncMock(side_effect=lambda _mid: next(outcomes)),
        ) as chunk:
            result = await jobs._run_embedding_migration_inline("mig-4")
        assert result == {"migration_id": "mig-4", "outcome": "complete", "chunks": 2}
        assert chunk.await_count == 3

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
            "record_prediction_outcome",
        }

    def test_worker_has_retry_and_timeout(self):
        assert jobs.WorkerSettings.max_tries >= 1
        assert jobs.WorkerSettings.job_timeout >= 300


# ── H1: resumable embedding migration chunks ──────────────────────────────────


class TestEmbeddingMigrationChunk:
    async def test_missing_migration_stops(self):
        pg = AsyncMock()
        pg.get = AsyncMock(return_value=None)
        with patch("smritikosh.db.postgres.db_session", _acm(pg)):
            outcome = await jobs._run_embedding_migration_chunk(
                "00000000-0000-0000-0000-000000000000"
            )
        assert outcome == "stopped"

    async def test_cancelled_migration_stops(self):
        from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus

        import uuid as _uuid

        mig = EmbeddingMigration(
            id=_uuid.uuid4(), target_model="m", target_dim=8,
            status=EmbeddingMigrationStatus.CANCELLED,
        )
        pg = AsyncMock()
        pg.get = AsyncMock(return_value=mig)
        with patch("smritikosh.db.postgres.db_session", _acm(pg)):
            outcome = await jobs._run_embedding_migration_chunk(str(mig.id))
        assert outcome == "stopped"
        pg.execute.assert_not_awaited()

    async def test_no_stale_rows_marks_complete(self):
        from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus

        import uuid as _uuid

        mig = EmbeddingMigration(
            id=_uuid.uuid4(), target_model="m", target_dim=8,
            status=EmbeddingMigrationStatus.RUNNING,
            processed=7, errors=1, total=8,
        )
        pg = AsyncMock()
        pg.get = AsyncMock(return_value=mig)
        empty = MagicMock()
        empty.fetchall.return_value = []
        pg.execute = AsyncMock(return_value=empty)
        with (
            patch("smritikosh.db.postgres.db_session", _acm(pg)),
            patch.object(jobs, "_emit_reembed_complete_audit", AsyncMock()) as audit,
        ):
            outcome = await jobs._run_embedding_migration_chunk(str(mig.id))
        assert outcome == "complete"
        assert mig.status == EmbeddingMigrationStatus.COMPLETE
        assert mig.finished_at is not None
        audit.assert_awaited_once()

    async def test_chunk_advances_cursor_and_counts_errors(self):
        import uuid as _uuid
        from datetime import datetime, timezone
        from types import SimpleNamespace

        from smritikosh.db.models import EmbeddingMigration, EmbeddingMigrationStatus

        mig = EmbeddingMigration(
            id=_uuid.uuid4(), target_model="m", target_dim=8,
            status=EmbeddingMigrationStatus.RUNNING, processed=0, errors=0, total=2,
        )
        row1 = SimpleNamespace(
            id=_uuid.uuid4(), raw_text="ok", created_at=datetime.now(timezone.utc)
        )
        row2 = SimpleNamespace(
            id=_uuid.uuid4(), raw_text="boom", created_at=datetime.now(timezone.utc)
        )
        rows_result = MagicMock()
        rows_result.fetchall.return_value = [row1, row2]

        pg = AsyncMock()
        pg.get = AsyncMock(return_value=mig)
        pg.execute = AsyncMock(return_value=rows_result)

        llm = AsyncMock()
        llm.embed = AsyncMock(
            side_effect=lambda text: (_ for _ in ()).throw(RuntimeError("embed fail"))
            if text == "boom" else [0.1] * 8
        )
        with (
            patch("smritikosh.db.postgres.db_session", _acm(pg)),
            patch("smritikosh.api.deps.get_llm", return_value=llm),
        ):
            outcome = await jobs._run_embedding_migration_chunk(str(mig.id))

        assert outcome == "continue"
        assert mig.status == EmbeddingMigrationStatus.RUNNING
        assert mig.processed == 2
        assert mig.errors == 1
        # cursor advanced to the LAST row — including the failed one, so a
        # permanently-failing event can never wedge the migration
        assert mig.cursor_id == row2.id
        assert mig.cursor_created_at == row2.created_at
