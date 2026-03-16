"""
iCalendar (.ics) connector.

Parses RFC 5545 iCalendar files using the standard library — no extra
dependencies.  Extracts ``VEVENT`` components and converts each into a
``ConnectorEvent``.

Content built per event:
    "Calendar event: {SUMMARY}\n{DESCRIPTION}\nLocation: {LOCATION}\n
     Start: {DTSTART}  End: {DTEND}"

Properties skipped if empty.  Attendee / organiser info is stored in
``metadata`` only, not in the main content text.

Limitations:
    Recurring events (``RRULE``) are stored as their first occurrence only.
    Timezones in ``TZID`` parameters are preserved as-is in the ISO string.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, date, timezone
from typing import Any

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

logger = logging.getLogger(__name__)


class CalendarConnector(SourceConnector):
    """Parse an iCalendar (.ics) file into ConnectorEvents."""

    source_name = "calendar"

    async def extract_events(  # type: ignore[override]
        self,
        content: bytes,
        filename: str = "calendar.ics",
    ) -> list[ConnectorEvent]:
        try:
            text = content.decode("utf-8", errors="replace")
            return _parse_ics(text, filename)
        except Exception:
            logger.exception("CalendarConnector failed to parse %r", filename)
            return []


# ── parser ─────────────────────────────────────────────────────────────────────

def _parse_ics(text: str, filename: str) -> list[ConnectorEvent]:
    # Unfold continued lines (RFC 5545 §3.1)
    text = re.sub(r"\r?\n[ \t]", "", text)

    events: list[ConnectorEvent] = []
    for vevent_text in _split_vevents(text):
        ev = _parse_vevent(vevent_text, filename)
        if ev:
            events.append(ev)
    return events


def _split_vevents(text: str) -> list[str]:
    pattern = re.compile(r"BEGIN:VEVENT(.*?)END:VEVENT", re.DOTALL)
    return pattern.findall(text)


def _parse_vevent(block: str, filename: str) -> ConnectorEvent | None:
    props = _extract_props(block)

    summary     = props.get("SUMMARY", "").strip()
    description = props.get("DESCRIPTION", "").strip()
    location    = props.get("LOCATION", "").strip()
    uid         = props.get("UID", "").strip()
    dtstart_raw = props.get("DTSTART", "")
    dtend_raw   = props.get("DTEND", "")
    organizer   = props.get("ORGANIZER", "").strip()
    attendees   = props.get("ATTENDEE_LIST", [])  # may be a list

    if not summary and not description:
        return None

    # Build human-readable content
    parts: list[str] = []
    if summary:
        parts.append(f"Calendar event: {summary}")
    if description:
        parts.append(description)
    if location:
        parts.append(f"Location: {location}")
    dtstart_str = _format_dt(dtstart_raw)
    dtend_str   = _format_dt(dtend_raw)
    if dtstart_str or dtend_str:
        parts.append(f"When: {dtstart_str}" + (f" → {dtend_str}" if dtend_str else ""))

    content = "\n".join(parts)
    occurred_at = _parse_ical_dt(dtstart_raw)

    meta: dict[str, Any] = {
        "filename":       filename,
        "ical_uid":       uid,
        "ical_summary":   summary,
        "ical_location":  location,
        "ical_dtstart":   dtstart_raw,
        "ical_dtend":     dtend_raw,
    }
    if organizer:
        meta["ical_organizer"] = organizer.replace("mailto:", "")
    if attendees:
        meta["ical_attendees"] = [a.replace("mailto:", "") for a in attendees]

    return ConnectorEvent(
        content=content,
        source="calendar",
        source_id=uid or f"{filename}#{summary[:40]}",
        occurred_at=occurred_at,
        metadata=meta,
    )


# ── property extraction helpers ────────────────────────────────────────────────

def _extract_props(block: str) -> dict[str, Any]:
    """
    Parse a VEVENT block into a dict of property name → value.

    Handles:
      - Simple: ``SUMMARY:My meeting``
      - Parameterized: ``DTSTART;TZID=America/New_York:20240101T090000``
      - Multiple ATTENDEE lines → collected into ``ATTENDEE_LIST``
    """
    props: dict[str, Any] = {}
    attendees: list[str] = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        colon_idx = line.find(":")
        if colon_idx == -1:
            continue
        name_part = line[:colon_idx]
        value = line[colon_idx + 1:]

        # Strip parameters (e.g. DTSTART;TZID=...)
        name = name_part.split(";")[0].upper()

        if name == "ATTENDEE":
            attendees.append(value)
        else:
            props[name] = value

    if attendees:
        props["ATTENDEE_LIST"] = attendees
    return props


def _parse_ical_dt(raw: str) -> datetime | None:
    """Parse iCal datetime/date strings into a timezone-aware datetime."""
    raw = raw.strip()
    if not raw:
        return None
    # Date-time with Z (UTC)
    if raw.endswith("Z"):
        raw = raw[:-1]
        try:
            return datetime.strptime(raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    # Date-time without timezone
    if "T" in raw:
        try:
            return datetime.strptime(raw[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    # Date only
    try:
        d = datetime.strptime(raw[:8], "%Y%m%d").replace(tzinfo=timezone.utc)
        return d
    except ValueError:
        return None


def _format_dt(raw: str) -> str:
    dt = _parse_ical_dt(raw)
    if dt is None:
        return raw  # return raw string if unparseable
    return dt.strftime("%Y-%m-%d %H:%M UTC") if "T" in raw else dt.strftime("%Y-%m-%d")
