"""Tests for custom Prometheus metrics and LLM usage accounting (G1 / B1 / D1).

Covers:
- track_job() — outcome counters, duration histogram, last-success gauge
- llm_context() / current_llm_context() — attribution ContextVar
- record_llm_usage() — token/cost counters + attributed persistence task
- LLMAdapter — calls/tokens recorded from a litellm response with usage
- MemoryScheduler — swallowed per-user failures increment JOB_USER_ERRORS

Prometheus collectors are process-global, so every assertion samples the
registry before and after and checks the delta.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from smritikosh import metrics
from smritikosh.llm.usage import (
    LLMCallContext,
    current_llm_context,
    llm_context,
    record_llm_usage,
)


def sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ── track_job ─────────────────────────────────────────────────────────────────


class TestTrackJob:
    def test_success_increments_and_freshens(self):
        runs_before = sample("smritikosh_job_runs_total", {"job": "tj_ok", "outcome": "success"})
        count_before = sample("smritikosh_job_duration_seconds_count", {"job": "tj_ok"})

        with metrics.track_job("tj_ok"):
            pass

        assert sample("smritikosh_job_runs_total", {"job": "tj_ok", "outcome": "success"}) == runs_before + 1
        assert sample("smritikosh_job_duration_seconds_count", {"job": "tj_ok"}) == count_before + 1
        assert sample("smritikosh_job_last_success_timestamp_seconds", {"job": "tj_ok"}) > 0

    def test_error_counts_and_propagates(self):
        errors_before = sample("smritikosh_job_runs_total", {"job": "tj_bad", "outcome": "error"})

        with pytest.raises(RuntimeError):
            with metrics.track_job("tj_bad"):
                raise RuntimeError("boom")

        assert sample("smritikosh_job_runs_total", {"job": "tj_bad", "outcome": "error"}) == errors_before + 1
        # A failed run must not advance the freshness gauge
        assert sample("smritikosh_job_last_success_timestamp_seconds", {"job": "tj_bad"}) == 0
        # Duration is still observed on failure
        assert sample("smritikosh_job_duration_seconds_count", {"job": "tj_bad"}) == 1


# ── llm_context ───────────────────────────────────────────────────────────────


class TestLlmContext:
    def test_default_is_unknown(self):
        ctx = current_llm_context()
        assert ctx == LLMCallContext()
        assert ctx.source == "unknown"

    def test_set_and_reset(self):
        with llm_context(user_id="alice", app_id="myapp", source="encode"):
            ctx = current_llm_context()
            assert ctx.user_id == "alice"
            assert ctx.app_id == "myapp"
            assert ctx.source == "encode"
        assert current_llm_context().source == "unknown"

    def test_nesting_restores_outer(self):
        with llm_context(source="consolidation"):
            with llm_context(user_id="bob", source="encode"):
                assert current_llm_context().source == "encode"
            assert current_llm_context().source == "consolidation"
            assert current_llm_context().user_id is None


# ── record_llm_usage ──────────────────────────────────────────────────────────


def fake_response(prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
    )


class TestRecordLlmUsage:
    @pytest.mark.asyncio
    async def test_counters_incremented(self):
        labels_p = {"model": "m1", "kind": "chat", "token_type": "prompt"}
        labels_c = {"model": "m1", "kind": "chat", "token_type": "completion"}
        p_before = sample("smritikosh_llm_tokens_total", labels_p)
        c_before = sample("smritikosh_llm_tokens_total", labels_c)

        with patch("smritikosh.llm.usage._persist_usage_row", new_callable=AsyncMock):
            record_llm_usage(model="m1", kind="chat", response=fake_response(100, 40))

        assert sample("smritikosh_llm_tokens_total", labels_p) == p_before + 100
        assert sample("smritikosh_llm_tokens_total", labels_c) == c_before + 40

    @pytest.mark.asyncio
    async def test_persist_carries_attribution(self):
        with patch(
            "smritikosh.llm.usage._persist_usage_row", new_callable=AsyncMock
        ) as mock_persist:
            with llm_context(user_id="alice", app_id="myapp", source="encode"):
                record_llm_usage(model="m1", kind="chat", response=fake_response())

        mock_persist.assert_called_once()
        kwargs = mock_persist.call_args.kwargs
        assert kwargs["user_id"] == "alice"
        assert kwargs["app_id"] == "myapp"
        assert kwargs["source"] == "encode"
        assert kwargs["prompt_tokens"] == 10
        assert kwargs["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_no_usage_data_skips_persist(self):
        with patch(
            "smritikosh.llm.usage._persist_usage_row", new_callable=AsyncMock
        ) as mock_persist:
            record_llm_usage(
                model="m1", kind="chat", response=SimpleNamespace(usage=None)
            )
        mock_persist.assert_not_called()

    def test_never_raises_outside_event_loop(self):
        # Counters recorded; persistence silently skipped without a loop.
        record_llm_usage(model="m1", kind="chat", response=fake_response())

    @pytest.mark.asyncio
    async def test_malformed_usage_treated_as_zero(self):
        with patch(
            "smritikosh.llm.usage._persist_usage_row", new_callable=AsyncMock
        ) as mock_persist:
            record_llm_usage(model="m1", kind="chat", response=MagicMock())
        mock_persist.assert_not_called()  # non-numeric token fields count as 0


# ── LLMAdapter instrumentation ────────────────────────────────────────────────


def make_adapter():
    from smritikosh.config import Settings
    from smritikosh.llm.adapter import LLMAdapter

    return LLMAdapter(
        Settings(
            llm_provider="claude",
            llm_model="test-model",
            llm_api_key="test-key",
            jwt_secret="a" * 40,
        )
    )


def litellm_response(content="ok", prompt_tokens=20, completion_tokens=8):
    choice = MagicMock()
    choice.message.content = content
    choice.message.reasoning_content = None
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class TestAdapterInstrumentation:
    @pytest.mark.asyncio
    async def test_complete_records_call_and_tokens(self):
        calls_labels = {"model": "test-model", "kind": "chat", "outcome": "success"}
        tokens_labels = {"model": "test-model", "kind": "chat", "token_type": "prompt"}
        calls_before = sample("smritikosh_llm_calls_total", calls_labels)
        tokens_before = sample("smritikosh_llm_tokens_total", tokens_labels)

        adapter = make_adapter()
        with (
            patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm,
            patch("smritikosh.llm.usage._persist_usage_row", new_callable=AsyncMock),
        ):
            mock_llm.return_value = litellm_response()
            await adapter.complete([{"role": "user", "content": "hi"}])

        assert sample("smritikosh_llm_calls_total", calls_labels) == calls_before + 1
        assert sample("smritikosh_llm_tokens_total", tokens_labels) == tokens_before + 20

    @pytest.mark.asyncio
    async def test_failed_call_records_error_outcome(self):
        error_labels = {"model": "test-model", "kind": "chat", "outcome": "error"}
        errors_before = sample("smritikosh_llm_calls_total", error_labels)

        from tenacity import RetryError

        adapter = make_adapter()
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("provider down")
            with pytest.raises(RetryError):
                await adapter.complete([{"role": "user", "content": "hi"}])

        # tenacity retries 3 times; each billed attempt counts as an error
        assert sample("smritikosh_llm_calls_total", error_labels) == errors_before + 3


# ── Scheduler per-user error counter (B1) ─────────────────────────────────────


class TestSchedulerUserErrors:
    @pytest.mark.asyncio
    async def test_swallowed_consolidation_failure_counts(self):
        from smritikosh.processing.scheduler import MemoryScheduler

        errors_before = sample(
            "smritikosh_job_user_errors_total", {"job": "consolidation"}
        )

        consolidator = AsyncMock()
        consolidator.run = AsyncMock(side_effect=RuntimeError("LLM down"))
        with patch("smritikosh.processing.scheduler.AsyncIOScheduler"):
            scheduler = MemoryScheduler(
                consolidator=consolidator, pruner=AsyncMock(), episodic=AsyncMock()
            )

        with (
            patch("smritikosh.processing.scheduler.db_session"),
            patch("smritikosh.processing.scheduler.neo4j_session"),
        ):
            result = await scheduler.run_consolidation_now(user_id="u1")

        assert result.skipped is True  # failure swallowed, as designed
        assert sample(
            "smritikosh_job_user_errors_total", {"job": "consolidation"}
        ) == errors_before + 1
