"""
Slack Events API connector.

Handles the three Slack interaction patterns:

    1. URL verification challenge (``url_verification`` type).
       Returns ``{"challenge": "..."}`` — handled at the route layer.

    2. Event callback (``event_callback`` type) for:
       - ``message``       — channel messages
       - ``app_mention``   — bot was @mentioned
       - ``message.im``    — direct messages to the app

    3. Signature verification: every inbound Slack request must carry
       ``X-Slack-Request-Timestamp`` and ``X-Slack-Signature`` headers.
       Verification uses HMAC-SHA256 over ``v0:{timestamp}:{raw_body}``
       and the app's signing secret.

Metadata stored per event:
    slack_channel, slack_user, slack_ts, slack_thread_ts, slack_team,
    slack_event_type, slack_bot_id (set if the message came from a bot)

Bot messages are skipped by default (``skip_bot_messages=True``).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from typing import Any

from smritikosh.connectors.base import ConnectorEvent, SourceConnector
from smritikosh.connectors.webhook import _parse_ts

logger = logging.getLogger(__name__)

_SLACK_MAX_AGE_SECONDS = 300  # 5 minutes — Slack's replay-attack window


class SlackConnector(SourceConnector):
    """Parse a Slack Events API payload into ConnectorEvents."""

    source_name = "slack"

    # ── signature verification ─────────────────────────────────────────────────

    @staticmethod
    def verify_signature(
        signing_secret: str,
        raw_body: bytes,
        timestamp: str,
        signature: str,
    ) -> bool:
        """
        Return True if the request is genuinely from Slack.

        Rejects requests older than 5 minutes to prevent replay attacks.
        """
        try:
            ts_int = int(timestamp)
        except ValueError:
            return False
        if abs(time.time() - ts_int) > _SLACK_MAX_AGE_SECONDS:
            logger.warning("Slack signature timestamp too old: %s", timestamp)
            return False

        base = f"v0:{timestamp}:{raw_body.decode('utf-8', errors='replace')}"
        expected = (
            "v0="
            + hmac.new(
                signing_secret.encode(),
                base.encode(),
                hashlib.sha256,
            ).hexdigest()
        )
        return hmac.compare_digest(expected, signature)

    # ── event extraction ───────────────────────────────────────────────────────

    async def extract_events(  # type: ignore[override]
        self,
        payload: dict[str, Any],
        *,
        skip_bot_messages: bool = True,
    ) -> list[ConnectorEvent]:
        """
        Return ConnectorEvents from a ``event_callback`` payload.

        Returns an empty list for non-message events or bot messages
        (when ``skip_bot_messages=True``).
        """
        if payload.get("type") != "event_callback":
            return []

        event: dict[str, Any] = payload.get("event", {})
        event_type: str = event.get("type", "")

        if event_type not in ("message", "app_mention", "message.im"):
            return []

        # Skip bot messages
        if skip_bot_messages and (event.get("bot_id") or event.get("subtype") == "bot_message"):
            return []

        text: str = (event.get("text") or "").strip()
        if not text:
            return []

        ts_raw = event.get("ts")
        occurred_at = _parse_ts(ts_raw)

        meta: dict[str, Any] = {
            "slack_channel":    event.get("channel", ""),
            "slack_user":       event.get("user", ""),
            "slack_ts":         ts_raw or "",
            "slack_thread_ts":  event.get("thread_ts", ""),
            "slack_team":       payload.get("team_id", ""),
            "slack_event_type": event_type,
        }
        if event.get("bot_id"):
            meta["slack_bot_id"] = event["bot_id"]

        return [
            ConnectorEvent(
                content=text,
                source="slack",
                source_id=f"{meta['slack_channel']}:{ts_raw or ''}",
                occurred_at=occurred_at,
                metadata=meta,
            )
        ]
