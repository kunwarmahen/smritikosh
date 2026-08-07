"""
Request-size and upload-abuse controls (item C4).

The existing protections cap the wrong dimensions for this failure mode:

    rate limits (A2)  cap request *frequency*  — 60 encodes/minute
    quotas (D2)       cap *totals* over a window — N events/tokens per day

Neither stops a **single** enormous request. `POST /ingest/session` feeds every
user turn to the extraction LLM, so one 50 MB transcript is one very expensive
call that passes both checks. This module caps the size of an individual
request, and bounds how much expensive media processing one tenant can have
in flight at once.

Three controls, all disabled by setting the corresponding setting to 0:

    enforce_session_size()      MAX_SESSION_TURNS / MAX_SESSION_CHARS → 413
    enforce_media_concurrency() MAX_CONCURRENT_MEDIA_UPLOADS → 429 + Retry-After
    body size middleware        MAX_REQUEST_BODY_BYTES → 413 (Content-Length)
"""

import logging
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.config import settings
from smritikosh.db.models import MediaIngest

logger = logging.getLogger(__name__)

# Paths exempt from the global body cap: they enforce their own, larger,
# per-content-type limits (a meeting recording may legitimately be 500 MB).
BODY_LIMIT_EXEMPT_PATHS: tuple[str, ...] = ("/ingest/media",)


# ── Session transcript size ───────────────────────────────────────────────────


def enforce_session_size(turns: list, *, user_id: str) -> None:
    """
    Reject a session payload that is too large to extract affordably.

    Raises 413 rather than truncating: silently dropping turns would corrupt
    the transcript the caller believes was stored.
    """
    max_turns = settings.max_session_turns
    if max_turns and len(turns) > max_turns:
        logger.warning(
            "Rejected oversized session ingest (turns)",
            extra={"user_id": user_id, "turns": len(turns), "limit": max_turns},
        )
        raise HTTPException(
            status_code=413,
            detail=(
                f"Too many turns in one request: {len(turns)} (limit {max_turns}). "
                "Split the transcript across several partial=true windows."
            ),
        )

    max_chars = settings.max_session_chars
    if max_chars:
        total = sum(len(getattr(t, "content", "") or "") for t in turns)
        if total > max_chars:
            logger.warning(
                "Rejected oversized session ingest (chars)",
                extra={"user_id": user_id, "chars": total, "limit": max_chars},
            )
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Transcript too large: {total} characters (limit {max_chars}). "
                    "Split the transcript across several partial=true windows."
                ),
            )


# ── Concurrent media uploads ──────────────────────────────────────────────────


async def enforce_media_concurrency(
    pg: AsyncSession, user_id: str, app_id: str
) -> None:
    """
    Cap how many uploads a tenant can have processing simultaneously.

    Rows stuck in `processing` for longer than MEDIA_PROCESSING_STALE_MINUTES
    are ignored — a worker that died mid-job must not lock the user out of
    uploading forever.
    """
    limit = settings.max_concurrent_media_uploads
    if not limit:
        return

    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=settings.media_processing_stale_minutes
    )
    in_flight = await pg.scalar(
        select(func.count())
        .select_from(MediaIngest)
        .where(
            MediaIngest.user_id == user_id,
            MediaIngest.app_id == app_id,
            MediaIngest.status == "processing",
            MediaIngest.created_at >= cutoff,
        )
    )
    if (in_flight or 0) >= limit:
        logger.warning(
            "Rejected media upload — concurrency limit reached",
            extra={"user_id": user_id, "in_flight": in_flight, "limit": limit},
        )
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many uploads already processing ({in_flight}/{limit}). "
                "Wait for one to finish, then retry."
            ),
            headers={"Retry-After": "60"},
        )


# ── Global request body cap ───────────────────────────────────────────────────


async def body_size_limit_middleware(request: Request, call_next):
    """
    Reject oversized bodies from Content-Length, before reading them.

    Checking the header means the body is never buffered — which is the point,
    since buffering is the resource we are protecting. Chunked requests send no
    Content-Length and pass through unchecked; enforce those at the proxy.
    """
    max_bytes = settings.max_request_body_bytes
    if max_bytes and not any(
        request.url.path.startswith(p) for p in BODY_LIMIT_EXEMPT_PATHS
    ):
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                length = int(raw_length)
            except ValueError:
                length = 0
            if length > max_bytes:
                logger.warning(
                    "Rejected oversized request body",
                    extra={
                        "path": request.url.path,
                        "content_length": length,
                        "limit": max_bytes,
                    },
                )
                return Response(
                    content=(
                        f'{{"detail":"Request body too large: {length} bytes '
                        f'(limit {max_bytes})."}}'
                    ),
                    status_code=413,
                    media_type="application/json",
                )
    return await call_next(request)
