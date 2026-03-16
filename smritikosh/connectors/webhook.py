"""
Generic webhook / push connector.

Accepts arbitrary JSON payloads and extracts a text event from them.
Callers decide what field carries the content, defaulting to ``"content"``.

Metadata preservation:
    All top-level JSON fields (except the content field) are stored in
    ``event_metadata`` so the original payload is not lost.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from smritikosh.connectors.base import ConnectorEvent, SourceConnector


class WebhookConnector(SourceConnector):
    """
    Normalise a generic JSON push payload into a ConnectorEvent.

    The caller passes the parsed payload dict and optionally the name of the
    field that holds the text content (default: ``"content"``).

    If the payload has a ``"timestamp"`` or ``"ts"`` field that looks like an
    ISO-8601 string or a Unix float, it is parsed as ``occurred_at``.
    """

    source_name = "webhook"

    async def extract_events(  # type: ignore[override]
        self,
        payload: dict[str, Any],
        *,
        content_field: str = "content",
        source_label: str = "webhook",
    ) -> list[ConnectorEvent]:
        content = str(payload.get(content_field, "")).strip()
        if not content:
            return []

        source_id = str(payload.get("id", payload.get("event_id", "")))
        occurred_at = _parse_ts(payload.get("timestamp") or payload.get("ts"))

        # Preserve all remaining fields as provenance metadata
        meta: dict[str, Any] = {
            k: v for k, v in payload.items() if k not in (content_field, "id", "timestamp", "ts")
        }

        return [
            ConnectorEvent(
                content=content,
                source=source_label,
                source_id=source_id,
                occurred_at=occurred_at,
                metadata=meta,
            )
        ]


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw), tz=timezone.utc)
        except (ValueError, OSError):
            return None
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
    return None
