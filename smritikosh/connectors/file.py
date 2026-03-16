"""
File upload connector.

Supports four common formats, auto-detected by filename extension:

    .txt / .md   Plain text — split on blank lines into paragraph chunks.
                 Each paragraph ≥ 20 chars becomes one ConnectorEvent.
    .csv         Each non-header row becomes one ConnectorEvent using all
                 columns joined as "column: value" pairs.
    .json        Expects a JSON array of strings or objects.
                 String elements are used directly; objects are JSON-dumped.
    (other)      Treated as plain text.

Chunk size cap:
    Paragraphs longer than MAX_CHUNK_CHARS are split on sentence boundaries
    (periods / newlines) so that single events remain focused.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from smritikosh.connectors.base import ConnectorEvent, SourceConnector

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 2_000


class FileConnector(SourceConnector):
    """Parse an uploaded file into memory-ready chunks."""

    source_name = "file"

    async def extract_events(  # type: ignore[override]
        self,
        content: bytes,
        filename: str = "upload.txt",
    ) -> list[ConnectorEvent]:
        ext = Path(filename).suffix.lower()
        try:
            if ext == ".csv":
                return _from_csv(content, filename)
            if ext == ".json":
                return _from_json(content, filename)
            # .txt, .md, or unknown
            return _from_text(content, filename)
        except Exception:
            logger.exception("FileConnector failed to parse %r", filename)
            return []


# ── parsers ────────────────────────────────────────────────────────────────────

def _from_text(raw: bytes, filename: str) -> list[ConnectorEvent]:
    text = raw.decode("utf-8", errors="replace")
    paragraphs = re.split(r"\n{2,}", text)
    events: list[ConnectorEvent] = []
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 20:
            continue
        for chunk in _split_long(para):
            events.append(
                ConnectorEvent(
                    content=chunk,
                    source="file",
                    source_id=f"{filename}#para{i}",
                    metadata={"filename": filename, "format": "text"},
                )
            )
    return events


def _from_csv(raw: bytes, filename: str) -> list[ConnectorEvent]:
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    events: list[ConnectorEvent] = []
    for row_num, row in enumerate(reader, start=2):  # row 1 = header
        parts = [f"{k}: {v}" for k, v in row.items() if v and str(v).strip()]
        content = "\n".join(parts)
        if len(content) < 5:
            continue
        events.append(
            ConnectorEvent(
                content=content,
                source="file",
                source_id=f"{filename}#row{row_num}",
                metadata={"filename": filename, "format": "csv", "row": row_num},
            )
        )
    return events


def _from_json(raw: bytes, filename: str) -> list[ConnectorEvent]:
    data: Any = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, list):
        data = [data]
    events: list[ConnectorEvent] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            content = item.strip()
        else:
            content = json.dumps(item, ensure_ascii=False)
        if len(content) < 5:
            continue
        events.append(
            ConnectorEvent(
                content=content,
                source="file",
                source_id=f"{filename}#item{i}",
                metadata={"filename": filename, "format": "json", "index": i},
            )
        )
    return events


def _split_long(text: str) -> list[str]:
    """Split text that exceeds MAX_CHUNK_CHARS on sentence boundaries."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    # Split on ". " or ".\n" keeping the period
    sentences = re.split(r"(?<=\.)\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + " " + sent).strip()
    if current:
        chunks.append(current.strip())
    return chunks or [text[:MAX_CHUNK_CHARS]]
