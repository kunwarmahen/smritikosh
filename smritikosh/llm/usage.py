"""
Per-call LLM usage accounting (item D1).

Two sinks, both fed by record_llm_usage() in the adapter:

  1. Prometheus counters (smritikosh.metrics) — fleet-wide tokens/cost/calls,
     no cardinality risk, always on.
  2. A per-call row in the llm_usage Postgres table — attributable spend for
     per-tenant reporting and future quotas (D2). Fire-and-forget; a failed
     write never breaks the pipeline.

Attribution uses a ContextVar set at pipeline entry points (encode, context
build, each scheduler job), so the adapter itself never needs user_id/app_id
threaded through its signatures:

    with llm_context(user_id="alice", app_id="myapp", source="encode"):
        await hippocampus.encode(...)   # all LLM calls inside are attributed
"""

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from smritikosh import metrics

logger = logging.getLogger(__name__)


# ── Call attribution context ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMCallContext:
    user_id: str | None = None
    app_id: str | None = None
    source: str = "unknown"   # encode | context | consolidation | … | unknown


_call_context: ContextVar[LLMCallContext] = ContextVar(
    "llm_call_context", default=LLMCallContext()
)


@contextmanager
def llm_context(
    user_id: str | None = None,
    app_id: str | None = None,
    source: str = "unknown",
):
    """Attribute all LLM calls made inside this block (async-safe, ContextVar)."""
    token = _call_context.set(LLMCallContext(user_id, app_id, source))
    try:
        yield
    finally:
        _call_context.reset(token)


def current_llm_context() -> LLMCallContext:
    return _call_context.get()


# ── Usage extraction ──────────────────────────────────────────────────────────


def _as_token_count(value: Any) -> int:
    """Coerce a usage field to int; anything non-numeric (None, mocks) is 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _usage_tokens(response: Any) -> tuple[int, int]:
    """Best-effort (prompt_tokens, completion_tokens) from a litellm response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return (
        _as_token_count(getattr(usage, "prompt_tokens", 0)),
        _as_token_count(getattr(usage, "completion_tokens", 0)),
    )


def _estimate_cost(response: Any) -> float:
    """USD cost from litellm's pricing table; 0.0 for local/unknown models."""
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0


# ── Recording ─────────────────────────────────────────────────────────────────


def record_llm_usage(*, model: str, kind: str, response: Any) -> None:
    """
    Record one successful LLM call: Prometheus counters plus a fire-and-forget
    llm_usage row attributed via the ambient llm_context(). Never raises.
    """
    prompt_tokens, completion_tokens = _usage_tokens(response)
    cost_usd = _estimate_cost(response)

    if prompt_tokens:
        metrics.LLM_TOKENS.labels(model=model, kind=kind, token_type="prompt").inc(prompt_tokens)
    if completion_tokens:
        metrics.LLM_TOKENS.labels(model=model, kind=kind, token_type="completion").inc(completion_tokens)
    if cost_usd:
        metrics.LLM_COST.labels(model=model, kind=kind).inc(cost_usd)

    if not (prompt_tokens or completion_tokens or cost_usd):
        return  # nothing to persist (provider returned no usage data)

    ctx = current_llm_context()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop (sync/test context) — counters are still recorded
    loop.create_task(
        _persist_usage_row(
            user_id=ctx.user_id,
            app_id=ctx.app_id,
            source=ctx.source,
            model=model,
            kind=kind,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
    )


async def _persist_usage_row(**fields) -> None:
    try:
        from smritikosh.db.models import LlmUsage
        from smritikosh.db.postgres import get_async_sessionmaker

        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            session.add(LlmUsage(**fields))
            await session.commit()
    except Exception as exc:
        # Accounting must never break the pipeline — log and move on.
        logger.debug("llm_usage persist failed (non-fatal): %s", exc)
