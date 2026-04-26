"""Gmail connector via Google API.

Fetches emails from Gmail using the Google Gmail API v1.
Requires an OAuth access token with gmail.readonly scope.
"""

import base64
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

logger = logging.getLogger(__name__)

_GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1/users/me"
_MAX_BODY_CHARS = 4000


class GmailConnector(SourceConnector):
    """Fetch emails from Gmail via the Google Gmail API."""

    source_name = "gmail"

    async def extract_events(
        self,
        access_token: str,
        *,
        limit: int = 20,
        query: str = "is:unread",
    ) -> list[ConnectorEvent]:
        """
        Fetch emails from Gmail and convert to ConnectorEvents.

        Args:
            access_token: Google OAuth access token with gmail.readonly scope
            limit: Max number of messages to fetch (1–100)
            query: Gmail search query (e.g. "is:unread", "from:alice@example.com")

        Returns: List of ConnectorEvent objects, one per email.
        """
        try:
            headers = {"Authorization": f"Bearer {access_token}"}
            async with httpx.AsyncClient() as client:
                # Step 1: Get message IDs
                message_ids = await self._fetch_message_ids(
                    client, headers, query=query, limit=limit
                )
                if not message_ids:
                    logger.info("No Gmail messages found matching query")
                    return []

                # Step 2: Fetch full message bodies
                events: list[ConnectorEvent] = []
                for msg_id in message_ids:
                    try:
                        ev = await self._fetch_message(client, headers, msg_id)
                        if ev:
                            events.append(ev)
                    except Exception:
                        logger.exception(f"Failed to fetch Gmail message {msg_id}")

                logger.info(f"GmailConnector extracted {len(events)} messages")
                return events

        except Exception:
            logger.exception("GmailConnector failed")
            return []

    async def _fetch_message_ids(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        query: str,
        limit: int,
    ) -> list[str]:
        """Fetch message IDs matching the query."""
        response = await client.get(
            f"{_GMAIL_API_BASE}/messages",
            headers=headers,
            params={"q": query, "maxResults": min(limit, 100)},
        )
        response.raise_for_status()
        data = response.json()
        return [msg["id"] for msg in data.get("messages", [])]

    async def _fetch_message(
        self, client: httpx.AsyncClient, headers: dict[str, str], msg_id: str
    ) -> Optional[ConnectorEvent]:
        """Fetch a single message and convert to ConnectorEvent."""
        response = await client.get(
            f"{_GMAIL_API_BASE}/messages/{msg_id}",
            headers=headers,
            params={"format": "full"},
        )
        response.raise_for_status()
        msg = response.json()

        headers_dict = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        subject = headers_dict.get("Subject", "")
        from_addr = headers_dict.get("From", "")
        date_str = headers_dict.get("Date", "")

        # Extract plaintext body
        body = self._extract_body(msg.get("payload", {}))
        if not body and not subject:
            return None

        content = f"From: {from_addr}\nSubject: {subject}\n\n{body}".strip()
        content = content[:_MAX_BODY_CHARS]

        return ConnectorEvent(
            content=content,
            source="gmail",
            source_id=msg_id,
            occurred_at=self._parse_date(date_str),
            metadata={
                "gmail_message_id": msg_id,
                "gmail_thread_id": msg.get("threadId", ""),
                "gmail_from": from_addr,
                "gmail_subject": subject,
                "gmail_date": date_str,
            },
        )

    def _extract_body(self, payload: dict) -> str:
        """Extract plaintext body from email payload, handling nested multipart."""
        # Single part — check if it's plaintext
        if payload.get("mimeType") == "text/plain":
            data = payload.get("body", {}).get("data", "")
            if data:
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace").strip()

        # Walk parts recursively — handles multipart/mixed → multipart/alternative → text/plain
        for part in payload.get("parts", []):
            result = self._extract_body(part)
            if result:
                return result

        return ""

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse RFC 2822 date string to datetime."""
        if not date_str:
            return None
        try:
            dt = parsedate_to_datetime(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
