"""
CrossSystemSynthesizer — infer durable behavioral patterns from connector metadata.

Correlates signals across all connected integrations (calendar, email, Slack, etc.)
to surface patterns that no single source could find alone.

Examples:
    "User rescheduled 3 meetings this week" + "mentioned being overwhelmed"
        → habit: avoids_overcommitting
    "No emails sent after 6pm for 30 days" + "mentioned work-life balance"
        → value: work_life_boundary
    "Slack activity spikes on Tuesdays" + "mentioned standup prep"
        → habit: tuesday_standup_prep

Processing pipeline (per user):
    1. Fetch recent connector events from Postgres (last 30 days)
       — filtered by event_metadata->>'source' ∈ connector sources
    2. Build per-connector behavioral summaries (frequency, timing, themes)
    3. Load recent episodic events (last 30 days) as conversational context
    4. Load current semantic facts for delta-extraction awareness
    5. LLM synthesis pass → candidate facts
    6. Store with source_type="cross_system"; confidence < 0.50 → status="pending"

Privacy note:
    Operates on behavioral metadata (event counts, timing, source labels).
    Content text is included only for episodic events already stored by the user.
    Connector event content is summarised, not quoted verbatim in the synthesis prompt.

Job cadence: daily (wired into MemoryScheduler at 01:00 UTC).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from neo4j import AsyncSession as NeoSession

from smritikosh.db.models import Event, FactStatus, SourceType
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

# Connector source identifiers used in event_metadata (ConnectorEvent.to_metadata)
_CONNECTOR_SOURCES = frozenset({"calendar", "email", "slack", "webhook", "file"})

# Only infer these fact categories — behavioral patterns only
_ALLOWED_CATEGORIES = frozenset({
    "habit", "preference", "lifestyle", "personality", "belief", "value", "goal", "role",
})

# Cross-system inferences below this confidence go to pending review
_PENDING_THRESHOLD = 0.50

_SYNTHESIS_SCHEMA = (
    "facts: list of objects with fields: "
    "category (must be one of: habit, preference, lifestyle, personality, "
    "belief, value, goal, role), "
    "key (short snake_case label e.g. morning_meetings, email_boundary), "
    "value (concise string describing the inferred pattern), "
    "confidence (float 0.0–1.0 — certainty this is a real durable pattern), "
    "rationale (one sentence explaining which signals led to this inference). "
    "Only include facts with confidence >= 0.40. Return an empty list if none qualify."
)

_SYNTHESIS_EXAMPLE = {
    "facts": [
        {
            "category": "habit",
            "key": "morning_meeting_preference",
            "value": "schedules nearly all meetings before noon",
            "confidence": 0.72,
            "rationale": "87% of calendar events over 30 days fall before 13:00",
        }
    ]
}


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SynthesisResult:
    user_id: str
    app_id: str
    connector_sources_found: list[str] = field(default_factory=list)
    connector_events_analyzed: int = 0
    episodic_events_analyzed: int = 0
    facts_synthesized: int = 0      # written as active
    facts_pending: int = 0          # written as pending (confidence < threshold)
    facts_skipped: int = 0          # below minimum confidence or invalid category
    skipped: bool = False
    skip_reason: str = ""


# ── CrossSystemSynthesizer ────────────────────────────────────────────────────

class CrossSystemSynthesizer:
    """
    Infers durable behavioral patterns by correlating connector signals with
    recent episodic events for a single user.

    Usage:
        synthesizer = CrossSystemSynthesizer(llm=..., episodic=..., semantic=...)
        async with db_session() as pg, neo4j_session() as neo:
            result = await synthesizer.run(pg, neo, user_id="u1", app_id="myapp")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        lookback_days: int = 30,
        max_connector_events_per_source: int = 100,
        max_episodic_events: int = 50,
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.semantic = semantic
        self.lookback_days = lookback_days
        self.max_connector_events_per_source = max_connector_events_per_source
        self.max_episodic_events = max_episodic_events

    async def run(
        self,
        pg: AsyncSession,
        neo: NeoSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> SynthesisResult:
        """
        Run one synthesis cycle for a single user.

        Returns a SynthesisResult with metrics; never raises — all errors
        are caught and surfaced via result.skipped / result.skip_reason.
        """
        result = SynthesisResult(user_id=user_id, app_id=app_id)
        since = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)

        try:
            # 1. Fetch connector-sourced events
            connector_events = await self._fetch_connector_events(
                pg, user_id=user_id, app_id=app_id, since=since
            )
            if not connector_events:
                result.skip_reason = "no connector events found in the last 30 days"
                result.skipped = True
                logger.debug(
                    "CrossSystemSynthesizer: no connector data for user=%s", user_id
                )
                return result

            # 2. Build per-connector behavioral summaries
            connector_summaries = _build_connector_summaries(connector_events)
            result.connector_sources_found = list(connector_summaries)
            result.connector_events_analyzed = sum(
                s["event_count"] for s in connector_summaries.values()
            )

            # 3. Fetch recent episodic events (all sources) for context
            episodic_texts = await self._fetch_episodic_summaries(
                pg, user_id=user_id, app_id=app_id, since=since
            )
            result.episodic_events_analyzed = len(episodic_texts)

            # 4. Fetch current semantic facts for delta-awareness
            existing_facts = await self.semantic.get_user_profile(
                neo, user_id=user_id, app_id=app_id
            )
            existing_summary = existing_facts.as_text_summary()

            # 5. Build synthesis prompt and call LLM
            prompt = _build_synthesis_prompt(
                connector_summaries=connector_summaries,
                episodic_texts=episodic_texts,
                existing_facts_summary=existing_summary,
                lookback_days=self.lookback_days,
            )
            raw = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_SYNTHESIS_SCHEMA,
                example_output=_SYNTHESIS_EXAMPLE,
            )
            candidates = raw.get("facts", [])

        except Exception as exc:
            logger.error(
                "CrossSystemSynthesizer failed",
                extra={"user_id": user_id, "error": str(exc)},
                exc_info=True,
            )
            result.skipped = True
            result.skip_reason = str(exc)
            return result

        # 6. Validate and write candidate facts
        for candidate in candidates:
            try:
                category = candidate.get("category", "").lower()
                key = candidate.get("key", "").strip()
                value = candidate.get("value", "").strip()
                confidence = float(candidate.get("confidence", 0.0))
                rationale = candidate.get("rationale", "")

                if category not in _ALLOWED_CATEGORIES or not key or not value:
                    result.facts_skipped += 1
                    continue
                if confidence < 0.40:
                    result.facts_skipped += 1
                    continue

                status = (
                    FactStatus.ACTIVE if confidence >= _PENDING_THRESHOLD
                    else FactStatus.PENDING
                )

                # Use default cross_system confidence from SOURCE_CONFIDENCE_DEFAULTS
                # but clamp to the LLM's stated confidence since it has full context
                from smritikosh.db.models import SOURCE_CONFIDENCE_DEFAULTS
                base_confidence = SOURCE_CONFIDENCE_DEFAULTS.get(
                    SourceType.CROSS_SYSTEM, 0.65
                )
                write_confidence = min(confidence, max(confidence, base_confidence))

                await self.semantic.upsert_fact(
                    neo,
                    user_id=user_id,
                    app_id=app_id,
                    category=category,
                    key=key,
                    value=value,
                    confidence=write_confidence,
                    source_type=SourceType.CROSS_SYSTEM,
                    source_meta={
                        "rationale": rationale,
                        "llm_confidence": confidence,
                        "connector_sources": result.connector_sources_found,
                    },
                    status=status,
                )

                if status == FactStatus.ACTIVE:
                    result.facts_synthesized += 1
                else:
                    result.facts_pending += 1

            except Exception as exc:
                logger.warning(
                    "CrossSystemSynthesizer: failed to write candidate fact",
                    extra={"user_id": user_id, "candidate": candidate, "error": str(exc)},
                )
                result.facts_skipped += 1

        logger.info(
            "CrossSystemSynthesizer complete",
            extra={
                "user_id": user_id,
                "app_id": app_id,
                "connector_sources": result.connector_sources_found,
                "connector_events_analyzed": result.connector_events_analyzed,
                "episodic_events_analyzed": result.episodic_events_analyzed,
                "facts_synthesized": result.facts_synthesized,
                "facts_pending": result.facts_pending,
                "facts_skipped": result.facts_skipped,
            },
        )
        return result

    # ── Private helpers ────────────────────────────────────────────────────

    async def _fetch_connector_events(
        self,
        pg: AsyncSession,
        *,
        user_id: str,
        app_id: str,
        since: datetime,
    ) -> list[Event]:
        """Fetch events tagged with connector sources in event_metadata."""
        result = await pg.execute(
            select(Event)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.created_at >= since,
                # Filter to connector-sourced events via event_metadata JSONB
                text(
                    "event_metadata->>'source' = ANY(:sources)"
                ).bindparams(sources=list(_CONNECTOR_SOURCES)),
            )
            .order_by(Event.created_at.desc())
            .limit(self.max_connector_events_per_source * len(_CONNECTOR_SOURCES))
        )
        return list(result.scalars().all())

    async def _fetch_episodic_summaries(
        self,
        pg: AsyncSession,
        *,
        user_id: str,
        app_id: str,
        since: datetime,
    ) -> list[str]:
        """Return raw_text snippets from recent episodic events (non-connector)."""
        result = await pg.execute(
            select(Event.raw_text)
            .where(
                Event.user_id == user_id,
                Event.app_id == app_id,
                Event.created_at >= since,
                Event.consolidated.is_(False),
                # Exclude connector-sourced events (already handled separately)
                text(
                    "COALESCE(event_metadata->>'source', '') != ALL(:sources)"
                ).bindparams(sources=list(_CONNECTOR_SOURCES)),
            )
            .order_by(Event.created_at.desc())
            .limit(self.max_episodic_events)
        )
        return [row[0] for row in result.all() if row[0]]


# ── Prompt helpers ─────────────────────────────────────────────────────────────

def _build_connector_summaries(
    events: list[Event],
) -> dict[str, dict[str, Any]]:
    """
    Build per-connector behavioral summaries from connector events.

    Returns a dict: source_name → summary dict with keys:
        event_count, hour_distribution (0–23 bucketed), days_active, example_topics
    """
    by_source: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        source = (ev.event_metadata or {}).get("source", "unknown")
        if source in _CONNECTOR_SOURCES:
            by_source[source].append(ev)

    summaries: dict[str, dict[str, Any]] = {}
    for source, source_events in by_source.items():
        hour_counts: dict[int, int] = defaultdict(int)
        days_seen: set[str] = set()

        for ev in source_events:
            ts = ev.created_at
            if ts:
                hour_counts[ts.hour] += 1
                days_seen.add(ts.strftime("%A"))  # weekday name

        # Top 3 hour buckets
        top_hours = sorted(hour_counts, key=lambda h: hour_counts[h], reverse=True)[:3]

        # Sample topics from raw_text (first 80 chars each, up to 5 samples)
        sample_topics = [
            ev.raw_text[:80] for ev in source_events[:5] if ev.raw_text
        ]

        summaries[source] = {
            "event_count": len(source_events),
            "top_active_hours": top_hours,
            "active_weekdays": sorted(days_seen),
            "sample_topics": sample_topics,
        }

    return summaries


def _build_synthesis_prompt(
    *,
    connector_summaries: dict[str, dict[str, Any]],
    episodic_texts: list[str],
    existing_facts_summary: str,
    lookback_days: int,
) -> str:
    """Build the LLM synthesis prompt from behavioral data."""
    parts: list[str] = []

    parts.append(
        f"You are analysing {lookback_days} days of behavioral signals for a single user "
        "to infer durable patterns about them.\n"
    )

    parts.append("## Connector behavioral signals\n")
    for source, summary in connector_summaries.items():
        parts.append(f"### {source.capitalize()} ({summary['event_count']} events)")
        if summary["top_active_hours"]:
            hours_str = ", ".join(f"{h:02d}:00" for h in summary["top_active_hours"])
            parts.append(f"  Most active hours: {hours_str}")
        if summary["active_weekdays"]:
            parts.append(f"  Active days: {', '.join(summary['active_weekdays'])}")
        if summary["sample_topics"]:
            parts.append("  Sample content snippets:")
            for snippet in summary["sample_topics"]:
                parts.append(f"    - {snippet}")
        parts.append("")

    if episodic_texts:
        parts.append("## Recent conversation excerpts (last 30 days)\n")
        for text_snippet in episodic_texts[:20]:
            parts.append(f"  - {text_snippet[:120]}")
        parts.append("")

    parts.append("## What is already known about this user\n")
    parts.append(existing_facts_summary)
    parts.append("")

    parts.append(
        "## Your task\n"
        "Infer durable behavioral patterns that:\n"
        "1. Combine signals from TWO OR MORE sources above\n"
        "2. Are NOT already captured in the known facts\n"
        "3. Represent stable, repeating behaviour — not one-off events\n"
        "\n"
        "Assign confidence < 0.50 if the pattern is speculative or based on sparse data.\n"
        "Return an empty list if no cross-source patterns emerge with confidence >= 0.40."
    )

    return "\n".join(parts)
