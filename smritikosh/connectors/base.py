"""
Base connector contract and the normalised event type.

All source connectors produce ``ConnectorEvent`` objects.  The caller is
responsible for passing them to ``Hippocampus.encode()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ConnectorEvent:
    """
    A single memory-worthy item produced by any source connector.

    Fields:
        content     Human-readable text to be encoded into memory.
        source      Connector name, e.g. ``"slack"``, ``"email"``, ``"file"``.
        source_id   Original ID (message ts, email UID, file name) — kept in
                    ``event_metadata`` for provenance and deduplication.
        occurred_at When the original event happened (if known).
        metadata    Arbitrary source-specific context stored verbatim in
                    ``event_metadata`` on the resulting ``Event`` row.
    """

    content: str
    source: str
    source_id: str = ""
    occurred_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Merge provenance fields into a single metadata dict for storage."""
        meta = {"source": self.source, "source_id": self.source_id, **self.metadata}
        if self.occurred_at is not None:
            meta["occurred_at"] = self.occurred_at.isoformat()
        return meta


class SourceConnector:
    """Abstract base class for all source connectors."""

    source_name: str = "unknown"

    async def extract_events(self, *args: Any, **kwargs: Any) -> list[ConnectorEvent]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement extract_events()"
        )
