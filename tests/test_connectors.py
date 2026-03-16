"""
Unit tests for external source connectors.

All connector tests are pure-Python (no I/O) — the connectors themselves
are the unit under test.  IMAP is the only connector that uses I/O; its
network layer is patched out via the sync _sync_fetch helper.
"""

import asyncio
import json
from datetime import timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smritikosh.connectors.base import ConnectorEvent
from smritikosh.connectors.calendar import CalendarConnector, _parse_ics
from smritikosh.connectors.email import EmailConnector, IMAPConfig
from smritikosh.connectors.file import FileConnector
from smritikosh.connectors.slack import SlackConnector
from smritikosh.connectors.webhook import WebhookConnector, _parse_ts


# ── WebhookConnector ──────────────────────────────────────────────────────────


class TestWebhookConnector:
    @pytest.mark.asyncio
    async def test_extracts_content_field(self):
        wc = WebhookConnector()
        events = await wc.extract_events({"content": "user prefers dark mode"})
        assert len(events) == 1
        assert events[0].content == "user prefers dark mode"

    @pytest.mark.asyncio
    async def test_custom_content_field(self):
        wc = WebhookConnector()
        events = await wc.extract_events(
            {"text": "hello world"}, content_field="text"
        )
        assert events[0].content == "hello world"

    @pytest.mark.asyncio
    async def test_empty_content_returns_empty_list(self):
        wc = WebhookConnector()
        events = await wc.extract_events({"content": "   "})
        assert events == []

    @pytest.mark.asyncio
    async def test_missing_content_field_returns_empty(self):
        wc = WebhookConnector()
        events = await wc.extract_events({"other": "data"})
        assert events == []

    @pytest.mark.asyncio
    async def test_extra_fields_preserved_in_metadata(self):
        wc = WebhookConnector()
        events = await wc.extract_events({
            "content": "event text",
            "channel": "slack",
            "team": "engineering",
        })
        meta = events[0].metadata
        assert meta["channel"] == "slack"
        assert meta["team"] == "engineering"

    @pytest.mark.asyncio
    async def test_source_label_applied(self):
        wc = WebhookConnector()
        events = await wc.extract_events({"content": "x"}, source_label="zapier")
        assert events[0].source == "zapier"

    @pytest.mark.asyncio
    async def test_parses_unix_timestamp(self):
        wc = WebhookConnector()
        events = await wc.extract_events({"content": "x", "timestamp": 1700000000.0})
        assert events[0].occurred_at is not None
        assert events[0].occurred_at.tzinfo is not None

    @pytest.mark.asyncio
    async def test_parses_iso_timestamp(self):
        wc = WebhookConnector()
        events = await wc.extract_events(
            {"content": "x", "timestamp": "2024-01-15T10:30:00Z"}
        )
        assert events[0].occurred_at is not None

    def test_parse_ts_none(self):
        assert _parse_ts(None) is None

    def test_parse_ts_invalid_string(self):
        assert _parse_ts("not-a-date") is None


# ── FileConnector ─────────────────────────────────────────────────────────────


class TestFileConnector:
    @pytest.mark.asyncio
    async def test_text_file_splits_on_blank_lines(self):
        fc = FileConnector()
        content = b"First paragraph here.\n\nSecond paragraph here."
        events = await fc.extract_events(content, "notes.txt")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_text_file_skips_short_paragraphs(self):
        fc = FileConnector()
        content = b"Hi.\n\nThis is a much longer paragraph with actual content worth storing."
        events = await fc.extract_events(content, "notes.txt")
        assert len(events) == 1
        assert "longer paragraph" in events[0].content

    @pytest.mark.asyncio
    async def test_markdown_file_treated_as_text(self):
        fc = FileConnector()
        content = b"# My Project\n\nThis is the first section with enough content to matter here."
        events = await fc.extract_events(content, "readme.md")
        assert len(events) >= 1
        assert events[0].metadata["format"] == "text"

    @pytest.mark.asyncio
    async def test_csv_one_event_per_row(self):
        fc = FileConnector()
        content = b"name,value\nfoo,bar is a test value here\nbaz,qux another value"
        events = await fc.extract_events(content, "data.csv")
        assert len(events) == 2
        assert "name: foo" in events[0].content
        assert events[0].metadata["format"] == "csv"

    @pytest.mark.asyncio
    async def test_json_array_of_strings(self):
        fc = FileConnector()
        data = ["user prefers dark mode", "user is building an AI startup"]
        events = await fc.extract_events(json.dumps(data).encode(), "memories.json")
        assert len(events) == 2
        assert events[0].content == "user prefers dark mode"
        assert events[0].metadata["format"] == "json"

    @pytest.mark.asyncio
    async def test_json_array_of_objects(self):
        fc = FileConnector()
        data = [{"key": "value", "name": "test"}]
        events = await fc.extract_events(json.dumps(data).encode(), "data.json")
        assert len(events) == 1
        assert "key" in events[0].content

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty_list(self):
        fc = FileConnector()
        events = await fc.extract_events(b"", "empty.txt")
        assert events == []

    @pytest.mark.asyncio
    async def test_source_id_contains_filename(self):
        fc = FileConnector()
        content = b"Some paragraph with enough words to be stored in memory here."
        events = await fc.extract_events(content, "myfile.txt")
        assert "myfile.txt" in events[0].source_id

    @pytest.mark.asyncio
    async def test_source_name_is_file(self):
        fc = FileConnector()
        content = b"A long enough paragraph for this to matter during testing."
        events = await fc.extract_events(content, "notes.txt")
        assert events[0].source == "file"


# ── SlackConnector ────────────────────────────────────────────────────────────


class TestSlackConnector:
    def make_message_payload(self, text="Hello world", bot_id=None, subtype=None):
        event = {
            "type": "message",
            "text": text,
            "user": "U123",
            "channel": "C456",
            "ts": "1700000000.000",
        }
        if bot_id:
            event["bot_id"] = bot_id
        if subtype:
            event["subtype"] = subtype
        return {
            "type": "event_callback",
            "team_id": "T789",
            "event": event,
        }

    @pytest.mark.asyncio
    async def test_extracts_message_text(self):
        sc = SlackConnector()
        events = await sc.extract_events(self.make_message_payload("I want to build an AI"))
        assert len(events) == 1
        assert events[0].content == "I want to build an AI"

    @pytest.mark.asyncio
    async def test_metadata_fields_populated(self):
        sc = SlackConnector()
        events = await sc.extract_events(self.make_message_payload())
        meta = events[0].metadata
        assert meta["slack_channel"] == "C456"
        assert meta["slack_user"] == "U123"
        assert meta["slack_team"] == "T789"
        assert meta["slack_event_type"] == "message"

    @pytest.mark.asyncio
    async def test_url_verification_returns_empty(self):
        sc = SlackConnector()
        events = await sc.extract_events({
            "type": "url_verification",
            "challenge": "abc123",
        })
        assert events == []

    @pytest.mark.asyncio
    async def test_bot_message_skipped_by_default(self):
        sc = SlackConnector()
        events = await sc.extract_events(
            self.make_message_payload(bot_id="B001")
        )
        assert events == []

    @pytest.mark.asyncio
    async def test_bot_message_included_when_flag_off(self):
        sc = SlackConnector()
        events = await sc.extract_events(
            self.make_message_payload(text="bot says hi", bot_id="B001"),
            skip_bot_messages=False,
        )
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        sc = SlackConnector()
        payload = self.make_message_payload(text="")
        events = await sc.extract_events(payload)
        assert events == []

    @pytest.mark.asyncio
    async def test_non_message_event_type_skipped(self):
        sc = SlackConnector()
        events = await sc.extract_events({
            "type": "event_callback",
            "event": {"type": "reaction_added", "reaction": "thumbsup"},
        })
        assert events == []

    @pytest.mark.asyncio
    async def test_app_mention_included(self):
        sc = SlackConnector()
        payload = {
            "type": "event_callback",
            "team_id": "T1",
            "event": {
                "type": "app_mention",
                "text": "<@BOT> what is my goal?",
                "user": "U1",
                "channel": "C1",
                "ts": "1700000001.0",
            },
        }
        events = await sc.extract_events(payload)
        assert len(events) == 1

    def test_verify_signature_wrong_secret(self):
        valid = SlackConnector.verify_signature(
            signing_secret="correct_secret",
            raw_body=b'{"type":"event_callback"}',
            timestamp="1700000000",
            signature="v0=badhash",
        )
        assert valid is False

    def test_verify_signature_stale_timestamp(self):
        """Timestamps older than 5 minutes should be rejected."""
        import time
        old_ts = str(int(time.time()) - 600)
        valid = SlackConnector.verify_signature(
            signing_secret="secret",
            raw_body=b"body",
            timestamp=old_ts,
            signature="v0=anything",
        )
        assert valid is False


# ── CalendarConnector ─────────────────────────────────────────────────────────

ICS_SAMPLE = b"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:meeting-001@example.com
SUMMARY:Product Strategy Meeting
DESCRIPTION:Discuss Q2 roadmap and AI features for smritikosh
LOCATION:Zoom Video Conference
DTSTART:20240315T100000Z
DTEND:20240315T110000Z
ORGANIZER:mailto:ceo@example.com
ATTENDEE:mailto:cto@example.com
ATTENDEE:mailto:pm@example.com
END:VEVENT
BEGIN:VEVENT
UID:lunch-002@example.com
SUMMARY:Lunch with investor
DTSTART:20240316T120000Z
DTEND:20240316T130000Z
END:VEVENT
END:VCALENDAR"""


class TestCalendarConnector:
    @pytest.mark.asyncio
    async def test_extracts_correct_number_of_events(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_summary_in_content(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert "Product Strategy Meeting" in events[0].content

    @pytest.mark.asyncio
    async def test_description_in_content(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert "Q2 roadmap" in events[0].content

    @pytest.mark.asyncio
    async def test_location_in_content(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert "Zoom" in events[0].content

    @pytest.mark.asyncio
    async def test_when_line_in_content(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert "2024-03-15" in events[0].content

    @pytest.mark.asyncio
    async def test_metadata_fields(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        meta = events[0].metadata
        assert meta["ical_uid"] == "meeting-001@example.com"
        assert meta["ical_summary"] == "Product Strategy Meeting"
        assert meta["ical_location"] == "Zoom Video Conference"
        assert "ceo@example.com" in meta["ical_organizer"]

    @pytest.mark.asyncio
    async def test_attendees_in_metadata(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        attendees = events[0].metadata["ical_attendees"]
        assert "cto@example.com" in attendees
        assert "pm@example.com" in attendees

    @pytest.mark.asyncio
    async def test_occurred_at_parsed(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert events[0].occurred_at is not None
        assert events[0].occurred_at.year == 2024
        assert events[0].occurred_at.month == 3

    @pytest.mark.asyncio
    async def test_source_is_calendar(self):
        cc = CalendarConnector()
        events = await cc.extract_events(ICS_SAMPLE, "calendar.ics")
        assert all(e.source == "calendar" for e in events)

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty(self):
        cc = CalendarConnector()
        events = await cc.extract_events(b"BEGIN:VCALENDAR\nEND:VCALENDAR", "empty.ics")
        assert events == []

    @pytest.mark.asyncio
    async def test_vevent_without_summary_and_description_skipped(self):
        ics = b"""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:no-content@test
DTSTART:20240315T100000Z
END:VEVENT
END:VCALENDAR"""
        cc = CalendarConnector()
        events = await cc.extract_events(ics, "test.ics")
        assert events == []


# ── ConnectorEvent.to_metadata ────────────────────────────────────────────────


class TestConnectorEventToMetadata:
    def test_includes_source_and_source_id(self):
        ev = ConnectorEvent(content="x", source="slack", source_id="C1:123")
        meta = ev.to_metadata()
        assert meta["source"] == "slack"
        assert meta["source_id"] == "C1:123"

    def test_includes_occurred_at_as_iso(self):
        from datetime import datetime, timezone
        dt = datetime(2024, 3, 15, 10, 0, 0, tzinfo=timezone.utc)
        ev = ConnectorEvent(content="x", source="s", occurred_at=dt)
        meta = ev.to_metadata()
        assert "occurred_at" in meta
        assert "2024-03-15" in meta["occurred_at"]

    def test_omits_occurred_at_when_none(self):
        ev = ConnectorEvent(content="x", source="s")
        meta = ev.to_metadata()
        assert "occurred_at" not in meta

    def test_extra_metadata_merged(self):
        ev = ConnectorEvent(content="x", source="s", metadata={"channel": "C1"})
        meta = ev.to_metadata()
        assert meta["channel"] == "C1"
