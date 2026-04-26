"""Google Calendar connector via Google Calendar API.

Fetches events from Google Calendar using the Calendar API v3.
Requires an OAuth access token with calendar.readonly scope.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

logger = logging.getLogger(__name__)

_CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"


class GcalConnector(SourceConnector):
    """Fetch events from Google Calendar via the Calendar API."""

    source_name = "gcal"

    async def extract_events(
        self,
        access_token: str,
        *,
        time_min: datetime,
        time_max: datetime,
        max_results: int = 50,
    ) -> list[ConnectorEvent]:
        """
        Fetch calendar events and convert to ConnectorEvents.

        Args:
            access_token: Google OAuth access token with calendar.readonly scope
            time_min: Start of time range (inclusive)
            time_max: End of time range (exclusive)
            max_results: Max number of events to fetch (1–2500)

        Returns: List of ConnectorEvent objects, one per event.
        """
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{_CALENDAR_API_BASE}/calendars/primary/events",
                    headers=headers,
                    params={
                        "timeMin": time_min.isoformat(),
                        "timeMax": time_max.isoformat(),
                        "maxResults": min(max_results, 2500),
                        "orderBy": "startTime",
                        "singleEvents": True,  # Expand recurring events
                    },
                )
                response.raise_for_status()
                data = response.json()

                events: list[ConnectorEvent] = []
                for item in data.get("items", []):
                    try:
                        ev = self._event_to_connector_event(item)
                        if ev:
                            events.append(ev)
                    except Exception:
                        logger.exception(f"Failed to convert calendar event {item.get('id')}")

                logger.info(f"GcalConnector extracted {len(events)} events")
                return events

        except Exception:
            logger.exception("GcalConnector failed")
            return []

    def _event_to_connector_event(self, item: dict) -> Optional[ConnectorEvent]:
        """Convert a Calendar API event to a ConnectorEvent."""
        summary = item.get("summary", "").strip()
        description = item.get("description", "").strip()
        location = item.get("location", "").strip()
        event_id = item.get("id", "")

        # Extract time information
        start_raw = item.get("start", {})
        end_raw = item.get("end", {})

        start_dt = self._parse_datetime(start_raw)
        end_dt = self._parse_datetime(end_raw)

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

        start_is_date = "date" in start_raw and "dateTime" not in start_raw
        end_is_date = "date" in end_raw and "dateTime" not in end_raw
        start_str = self._format_datetime(start_dt, date_only=start_is_date) if start_dt else ""
        end_str = self._format_datetime(end_dt, date_only=end_is_date) if end_dt else ""
        if start_str or end_str:
            time_str = start_str
            if end_str:
                time_str += f" → {end_str}"
            parts.append(f"When: {time_str}")

        content = "\n".join(parts)

        # Extract organizer and attendees
        meta: dict = {
            "gcal_event_id": event_id,
            "gcal_summary": summary,
            "gcal_location": location,
            "gcal_start": start_raw.get("dateTime") or start_raw.get("date", ""),
            "gcal_end": end_raw.get("dateTime") or end_raw.get("date", ""),
        }

        organizer = item.get("organizer", {})
        if organizer.get("email"):
            meta["gcal_organizer"] = organizer["email"]

        attendees = [a.get("email") for a in item.get("attendees", []) if a.get("email")]
        if attendees:
            meta["gcal_attendees"] = attendees

        return ConnectorEvent(
            content=content,
            source="gcal",
            source_id=event_id or f"{summary[:40]}",
            occurred_at=start_dt,
            metadata=meta,
        )

    def _parse_datetime(self, dt_dict: dict) -> Optional[datetime]:
        """Parse a dateTime or date object from Calendar API."""
        if "dateTime" in dt_dict:
            try:
                dt_str = dt_dict["dateTime"]
                # ISO 8601 datetime string (may include timezone)
                if dt_str.endswith("Z"):
                    dt_str = dt_str[:-1] + "+00:00"
                return datetime.fromisoformat(dt_str)
            except Exception:
                pass

        if "date" in dt_dict:
            try:
                date_str = dt_dict["date"]
                return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                pass

        return None

    def _format_datetime(self, dt: Optional[datetime], *, date_only: bool = False) -> str:
        """Format datetime for display."""
        if not dt:
            return ""
        if date_only:
            return dt.strftime("%Y-%m-%d")
        return dt.strftime("%Y-%m-%d %H:%M%z")
