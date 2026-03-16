"""
Smritikosh source connectors.

Each connector normalises an external source (file, webhook, Slack, email,
calendar) into a list of ``ConnectorEvent`` objects that can be handed
directly to ``Hippocampus.encode()``.

Usage pattern (in a FastAPI route):
    connector = FileConnector()
    events    = await connector.extract_events(file_bytes, filename)
    for ev in events:
        await hippocampus.encode(pg, neo, user_id=..., raw_text=ev.content, metadata=ev.metadata)
"""

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

__all__ = ["ConnectorEvent", "SourceConnector"]
