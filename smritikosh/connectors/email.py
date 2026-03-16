"""
Email / IMAP connector.

Connects to an IMAP server using the standard library ``imaplib`` (SSL) in a
thread-pool executor to avoid blocking the event loop.  No extra dependencies.

What gets ingested per email:
    content  =  "From: {from}\nSubject: {subject}\n\n{plain-text body}"
    metadata =  {email_uid, email_from, email_subject, email_date, mailbox}

Attachment text is *not* extracted — only the first ``text/plain`` part.

Security note:
    IMAP credentials are provided per-request and never stored.  The caller
    must decide whether it is safe to transmit credentials to this endpoint
    (TLS on the FastAPI server is assumed).
"""

from __future__ import annotations

import asyncio
import email as _email
import imaplib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.header import decode_header
from email.utils import parseaddr, parsedate_to_datetime
from functools import partial
from typing import Any

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

logger = logging.getLogger(__name__)

_MAX_BODY_CHARS = 4_000


@dataclass
class IMAPConfig:
    host: str
    username: str
    password: str
    port: int = 993
    mailbox: str = "INBOX"
    limit: int = 20
    unseen_only: bool = True


class EmailConnector(SourceConnector):
    """Fetch emails via IMAP and convert them to ConnectorEvents."""

    source_name = "email"

    async def extract_events(  # type: ignore[override]
        self,
        config: IMAPConfig,
    ) -> list[ConnectorEvent]:
        loop = asyncio.get_running_loop()
        try:
            events = await loop.run_in_executor(None, partial(_sync_fetch, config))
        except Exception:
            logger.exception(
                "EmailConnector IMAP fetch failed for %s@%s", config.username, config.host
            )
            events = []
        return events


# ── sync IMAP helpers (run in executor) ───────────────────────────────────────

def _sync_fetch(config: IMAPConfig) -> list[ConnectorEvent]:
    with imaplib.IMAP4_SSL(config.host, config.port) as imap:
        imap.login(config.username, config.password)
        imap.select(config.mailbox, readonly=True)

        search_key = "UNSEEN" if config.unseen_only else "ALL"
        _, data = imap.search(None, search_key)
        uids: list[bytes] = data[0].split() if data and data[0] else []

        # Take the most recent `limit` emails
        uids = uids[-config.limit:]

        events: list[ConnectorEvent] = []
        for uid in reversed(uids):  # newest first
            try:
                ev = _fetch_one(imap, uid)
                if ev:
                    events.append(ev)
            except Exception:
                logger.exception("EmailConnector: failed to parse UID %s", uid)
        return events


def _fetch_one(imap: imaplib.IMAP4_SSL, uid: bytes) -> ConnectorEvent | None:
    _, raw = imap.fetch(uid, "(RFC822)")
    if not raw or not raw[0]:
        return None
    raw_bytes = raw[0][1] if isinstance(raw[0], tuple) else None
    if not raw_bytes:
        return None

    msg = _email.message_from_bytes(raw_bytes)

    subject = _decode_header_value(msg.get("Subject", ""))
    from_raw = msg.get("From", "")
    _, from_addr = parseaddr(from_raw)
    date_str = msg.get("Date", "")
    occurred_at = _parse_date(date_str)

    body = _extract_text(msg)
    if not body and not subject:
        return None

    content = f"From: {from_raw}\nSubject: {subject}\n\n{body}".strip()
    content = content[:_MAX_BODY_CHARS]

    return ConnectorEvent(
        content=content,
        source="email",
        source_id=f"{uid.decode()}@{msg.get('Message-ID', '')}",
        occurred_at=occurred_at,
        metadata={
            "email_uid":     uid.decode(),
            "email_from":    from_addr,
            "email_subject": subject,
            "email_date":    date_str,
        },
    )


def _extract_text(msg: Any) -> str:
    """Return the first text/plain part of an email."""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                charset = part.get_content_charset() or "utf-8"
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(charset, errors="replace").strip()
        return ""
    if msg.get_content_type() == "text/plain":
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace").strip()
    return ""


def _decode_header_value(raw: str) -> str:
    parts = decode_header(raw)
    result = []
    for fragment, charset in parts:
        if isinstance(fragment, bytes):
            result.append(fragment.decode(charset or "utf-8", errors="replace"))
        else:
            result.append(fragment)
    return "".join(result)


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
