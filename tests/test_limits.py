"""
Tests for request-size and upload-abuse controls (item C4):
    enforce_session_size()      — max turns / max chars on session ingest
    enforce_media_concurrency() — per-tenant in-flight upload cap + staleness
    body_size_limit_middleware  — global Content-Length cap, media exempt

These cap the size of a SINGLE request — the dimension rate limits (frequency)
and quotas (windowed totals) both leave open.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from smritikosh.api.limits import (
    BODY_LIMIT_EXEMPT_PATHS,
    body_size_limit_middleware,
    enforce_media_concurrency,
    enforce_session_size,
)


def turns(count: int, chars_each: int = 10) -> list:
    return [SimpleNamespace(role="user", content="x" * chars_each) for _ in range(count)]


# ── session size ──────────────────────────────────────────────────────────────


class TestSessionSize:
    def test_allows_a_normal_transcript(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 500
            s.max_session_chars = 200_000
            enforce_session_size(turns(10), user_id="u1")   # no raise

    def test_rejects_too_many_turns(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 5
            s.max_session_chars = 0
            with pytest.raises(HTTPException) as exc:
                enforce_session_size(turns(6), user_id="u1")

        assert exc.value.status_code == 413
        assert "Too many turns" in exc.value.detail
        assert "partial=true" in exc.value.detail   # tells the caller what to do

    def test_allows_exactly_the_turn_limit(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 5
            s.max_session_chars = 0
            enforce_session_size(turns(5), user_id="u1")

    def test_rejects_too_many_chars(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 0
            s.max_session_chars = 100
            with pytest.raises(HTTPException) as exc:
                enforce_session_size(turns(3, chars_each=50), user_id="u1")

        assert exc.value.status_code == 413
        assert "Transcript too large" in exc.value.detail

    def test_chars_are_summed_across_turns(self):
        """A single small turn passes; many small turns together must not."""
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 0
            s.max_session_chars = 100
            enforce_session_size(turns(2, chars_each=40), user_id="u1")      # 80 ok
            with pytest.raises(HTTPException):
                enforce_session_size(turns(4, chars_each=40), user_id="u1")  # 160

    def test_zero_disables_both_limits(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 0
            s.max_session_chars = 0
            enforce_session_size(turns(10_000, chars_each=1000), user_id="u1")

    def test_tolerates_missing_content(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_session_turns = 0
            s.max_session_chars = 100
            enforce_session_size(
                [SimpleNamespace(role="user", content=None)], user_id="u1"
            )


# ── media concurrency ─────────────────────────────────────────────────────────


def session_with_count(count: int) -> AsyncMock:
    pg = AsyncMock()
    pg.scalar = AsyncMock(return_value=count)
    return pg


class TestMediaConcurrency:
    @pytest.mark.asyncio
    async def test_allows_below_the_limit(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_concurrent_media_uploads = 3
            s.media_processing_stale_minutes = 60
            await enforce_media_concurrency(session_with_count(2), "u1", "default")

    @pytest.mark.asyncio
    async def test_rejects_at_the_limit(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_concurrent_media_uploads = 3
            s.media_processing_stale_minutes = 60
            with pytest.raises(HTTPException) as exc:
                await enforce_media_concurrency(session_with_count(3), "u1", "default")

        assert exc.value.status_code == 429
        assert exc.value.headers["Retry-After"] == "60"
        assert "3/3" in exc.value.detail

    @pytest.mark.asyncio
    async def test_zero_disables_the_check(self):
        pg = session_with_count(999)
        with patch("smritikosh.api.limits.settings") as s:
            s.max_concurrent_media_uploads = 0
            await enforce_media_concurrency(pg, "u1", "default")

        # Disabled means no query at all, not just no raise.
        pg.scalar.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stale_rows_are_excluded_by_a_cutoff(self):
        """A crashed worker must not lock a user out of uploading forever."""
        pg = session_with_count(0)
        with patch("smritikosh.api.limits.settings") as s:
            s.max_concurrent_media_uploads = 3
            s.media_processing_stale_minutes = 60
            await enforce_media_concurrency(pg, "u1", "default")

        # The query must constrain created_at — that is the staleness escape hatch.
        rendered = str(pg.scalar.await_args.args[0])
        assert "created_at" in rendered

    @pytest.mark.asyncio
    async def test_none_count_is_treated_as_zero(self):
        pg = AsyncMock()
        pg.scalar = AsyncMock(return_value=None)
        with patch("smritikosh.api.limits.settings") as s:
            s.max_concurrent_media_uploads = 1
            s.media_processing_stale_minutes = 60
            await enforce_media_concurrency(pg, "u1", "default")


# ── global body cap ───────────────────────────────────────────────────────────


def make_request(path: str, content_length: str | None) -> MagicMock:
    request = MagicMock()
    request.url = MagicMock()
    request.url.path = path
    request.headers = {"content-length": content_length} if content_length else {}
    return request


async def passthrough(_request):
    return "downstream-response"


class TestBodySizeMiddleware:
    @pytest.mark.asyncio
    async def test_allows_a_small_body(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(
                make_request("/memory/event", "500"), passthrough
            )

        assert result == "downstream-response"

    @pytest.mark.asyncio
    async def test_rejects_an_oversized_body_before_reading_it(self):
        request = make_request("/ingest/session", "2000")
        called = False

        async def should_not_run(_request):
            nonlocal called
            called = True
            return "downstream-response"

        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(request, should_not_run)

        assert result.status_code == 413
        # The point of the header check: the handler never ran, body never read.
        assert called is False

    @pytest.mark.asyncio
    async def test_media_upload_is_exempt(self):
        """Media carries its own, much larger, per-content-type limits."""
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(
                make_request("/ingest/media", "50000000"), passthrough
            )

        assert result == "downstream-response"
        assert "/ingest/media" in BODY_LIMIT_EXEMPT_PATHS

    @pytest.mark.asyncio
    async def test_missing_content_length_passes_through(self):
        """Chunked requests carry no length — documented as proxy-enforced."""
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(
                make_request("/memory/event", None), passthrough
            )

        assert result == "downstream-response"

    @pytest.mark.asyncio
    async def test_malformed_content_length_passes_through(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(
                make_request("/memory/event", "not-a-number"), passthrough
            )

        assert result == "downstream-response"

    @pytest.mark.asyncio
    async def test_zero_disables_the_cap(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 0
            result = await body_size_limit_middleware(
                make_request("/memory/event", "999999999"), passthrough
            )

        assert result == "downstream-response"

    @pytest.mark.asyncio
    async def test_exactly_at_the_limit_is_allowed(self):
        with patch("smritikosh.api.limits.settings") as s:
            s.max_request_body_bytes = 1000
            result = await body_size_limit_middleware(
                make_request("/memory/event", "1000"), passthrough
            )

        assert result == "downstream-response"
