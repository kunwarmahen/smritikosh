"""
ReconsolidationEngine — updates recalled memories in light of new context.

In the human brain, every act of recall destabilises a memory briefly.
The brain then re-saves it, potentially incorporating new context or
associations — this is called Memory Reconsolidation.

Here we replicate that function: when an event surfaces through hybrid
search, the engine uses the LLM to refine its summary by weaving in the
recall context (the current query and surrounding memories).  This means
frequently-recalled memories gradually become more accurate, context-rich
summaries rather than raw interaction text.

Gate conditions (all must pass before an LLM call is made):
    recall_count >= min_recall_count    — memory has been recalled before
    importance_score >= min_importance  — worth the LLM cost
    cooldown hours since last reconsolidation   — don't reconsolidate repeatedly

The engine is intentionally lightweight — one LLM call per event, one DB
update.  It runs as a background task so the API response is never blocked.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event
from smritikosh.db.postgres import db_session
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory, SearchResult

logger = logging.getLogger(__name__)

# ── Default gate thresholds ───────────────────────────────────────────────────

DEFAULT_MIN_RECALL_COUNT = 2      # recalled at least twice before
DEFAULT_MIN_IMPORTANCE = 0.4      # non-trivial importance score
DEFAULT_COOLDOWN_HOURS = 1        # re-consolidate at most once per hour per event
DEFAULT_MAX_EVENTS = 1            # how many top recalled events to process

_SCHEMA = (
    "summary (string): the refined 1-2 sentence memory summary, "
    "changed (boolean): true if the summary was meaningfully updated, false otherwise"
)
_EXAMPLE = {
    "summary": (
        "User is building smritikosh, an AI memory startup; "
        "frequently revisits product direction and architecture questions."
    ),
    "changed": True,
}


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ReconsolidationResult:
    """Outcome of a single event reconsolidation attempt."""

    event_id: str
    user_id: str
    updated: bool = False          # True if the LLM produced a meaningfully different summary
    skipped: bool = False          # True if gate conditions blocked the attempt
    skip_reason: str = ""
    old_summary: str = ""
    new_summary: str = ""


@dataclass
class BatchReconsolidationResult:
    """Outcome of reconsolidating a batch of recalled events."""

    user_id: str
    events_evaluated: int = 0
    events_updated: int = 0
    events_skipped: int = 0
    results: list[ReconsolidationResult] = field(default_factory=list)


# ── ReconsolidationEngine ─────────────────────────────────────────────────────


class ReconsolidationEngine:
    """
    Refines event summaries when memories are recalled in new contexts.

    Designed to run as a FastAPI BackgroundTask after every context build:
        background_tasks.add_task(engine.reconsolidate_after_recall, results, query)

    Uses its own db_session() to avoid depending on the request session,
    which may already be closed when the background task runs.

    Usage:
        engine = ReconsolidationEngine(llm=llm, episodic=episodic)
        await engine.reconsolidate_after_recall(search_results, query="...")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        *,
        min_recall_count: int = DEFAULT_MIN_RECALL_COUNT,
        min_importance: float = DEFAULT_MIN_IMPORTANCE,
        cooldown_hours: int = DEFAULT_COOLDOWN_HOURS,
        max_events: int = DEFAULT_MAX_EVENTS,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.min_recall_count = min_recall_count
        self.min_importance = min_importance
        self.cooldown_hours = cooldown_hours
        self.max_events = max_events
        self.audit = audit

    # ── Primary entry point (called as BackgroundTask) ────────────────────

    async def reconsolidate_after_recall(
        self,
        search_results: list[SearchResult],
        query: str,
        user_id: str,
    ) -> BatchReconsolidationResult:
        """
        Reconsolidate the top recalled events in the background.

        Opens its own DB session (independent of the request session)
        so this method is safe to run as a FastAPI BackgroundTask.

        Only the top `max_events` results are processed to keep latency
        and LLM cost proportional.
        """
        batch = BatchReconsolidationResult(user_id=user_id)

        candidates = search_results[: self.max_events]
        batch.events_evaluated = len(candidates)

        async with db_session() as session:
            for sr in candidates:
                result = await self._reconsolidate_one(session, sr.event, query)
                batch.results.append(result)
                if result.updated:
                    batch.events_updated += 1
                else:
                    batch.events_skipped += 1

        logger.info(
            "Reconsolidation batch complete",
            extra={
                "user_id": user_id,
                "evaluated": batch.events_evaluated,
                "updated": batch.events_updated,
            },
        )

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.MEMORY_RECONSOLIDATE_RUN,
                user_id=user_id,
                app_id="default",
                payload={
                    "events_evaluated": batch.events_evaluated,
                    "events_updated": batch.events_updated,
                    "events_skipped": batch.events_skipped,
                    "skip_reasons": [
                        r.skip_reason for r in batch.results if r.skipped
                    ],
                },
            ))

        return batch

    async def reconsolidate_event(
        self,
        event_id_str: str,
        query: str,
        user_id: str,
    ) -> ReconsolidationResult:
        """
        Reconsolidate a single event by ID (for manual/admin triggers).

        Opens its own DB session — safe to call from background tasks or admin routes.
        """
        import uuid as _uuid_mod

        try:
            eid = _uuid_mod.UUID(event_id_str)
        except ValueError:
            return ReconsolidationResult(
                event_id=event_id_str, user_id=user_id,
                skipped=True, skip_reason="Invalid UUID format.",
            )

        async with db_session() as session:
            event = await session.get(Event, eid)
            if event is None:
                return ReconsolidationResult(
                    event_id=event_id_str, user_id=user_id,
                    skipped=True, skip_reason="Event not found.",
                )
            return await self._reconsolidate_one(session, event, query)

    # ── Core logic ────────────────────────────────────────────────────────

    async def _reconsolidate_one(
        self,
        session: AsyncSession,
        event: Event,
        query: str,
    ) -> ReconsolidationResult:
        """
        Attempt to reconsolidate a single event.

        Gate → LLM refine → DB update.
        """
        result = ReconsolidationResult(
            event_id=str(event.id),
            user_id=event.user_id,
            old_summary=event.summary or event.raw_text,
        )

        # ── Gate checks ───────────────────────────────────────────────────
        gate_fail = self._check_gate(event)
        if gate_fail:
            result.skipped = True
            result.skip_reason = gate_fail
            logger.debug(
                "Reconsolidation gated",
                extra={"event_id": str(event.id), "reason": gate_fail},
            )
            return result

        # ── LLM refinement ────────────────────────────────────────────────
        prompt = _build_prompt(event, query)
        try:
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_SCHEMA,
                example_output=_EXAMPLE,
            )
        except Exception as exc:
            result.skipped = True
            result.skip_reason = f"LLM call failed: {exc}"
            logger.warning(
                "Reconsolidation LLM call failed",
                extra={"event_id": str(event.id), "error": str(exc)},
            )
            return result

        new_summary = extracted.get("summary", "").strip()
        changed = extracted.get("changed", False)

        if not new_summary or not changed:
            result.skipped = True
            result.skip_reason = "LLM reported no meaningful change."
            return result

        # ── DB update ─────────────────────────────────────────────────────
        await self.episodic.update_summary(session, event.id, new_summary)
        result.updated = True
        result.new_summary = new_summary

        logger.info(
            "Memory reconsolidated",
            extra={
                "event_id": str(event.id),
                "user_id": event.user_id,
                "reconsolidation_count": (event.reconsolidation_count or 0) + 1,
            },
        )

        if self.audit:
            from smritikosh.audit.logger import AuditEvent, EventType
            await self.audit.emit(AuditEvent(
                event_type=EventType.MEMORY_RECONSOLIDATED,
                user_id=event.user_id,
                app_id=event.app_id,
                event_id=str(event.id),
                payload={
                    "recall_context": query,
                    "old_summary": result.old_summary,
                    "new_summary": new_summary,
                    "reconsolidation_count": (event.reconsolidation_count or 0) + 1,
                },
            ))

        return result

    def _check_gate(self, event: Event) -> str:
        """
        Return a non-empty skip reason string if reconsolidation should be blocked,
        or an empty string if all gates pass.
        """
        if (event.recall_count or 0) < self.min_recall_count:
            return (
                f"recall_count={event.recall_count} < min={self.min_recall_count}"
            )

        if (event.importance_score or 0.0) < self.min_importance:
            return (
                f"importance_score={event.importance_score:.2f} < min={self.min_importance}"
            )

        if event.last_reconsolidated_at is not None:
            last = event.last_reconsolidated_at
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            cooldown_end = last + timedelta(hours=self.cooldown_hours)
            if datetime.now(timezone.utc) < cooldown_end:
                return (
                    f"cooldown active — last reconsolidated at {last.isoformat()}"
                )

        return ""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_prompt(event: Event, query: str) -> str:
    """Build the LLM prompt for a single event reconsolidation."""
    current_text = event.summary or event.raw_text
    lines = [
        "You are refining a stored memory summary in light of a new recall context.",
        "",
        f'Original memory: "{current_text}"',
        "",
        f'Recalled in the context of: "{query}"',
        "",
        "Refine the memory summary to incorporate any new connections, patterns, "
        "or insights revealed by this recall. Keep it concise (1-2 sentences). "
        "If the existing summary is already complete and accurate, return it unchanged "
        "and set changed=false.",
    ]
    return "\n".join(lines)
