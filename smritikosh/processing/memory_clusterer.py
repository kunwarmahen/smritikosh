"""
MemoryClusterer — groups episodic events into topical clusters.

Mirrors the brain's cortical organisation of memories into conceptual
categories: the same episodic facts are mentally filed under "work",
"relationships", "health", etc.

Pipeline:
    1. Fetch all events with embeddings for a user.
    2. Guard: skip if fewer than min_events.
    3. Cluster by cosine similarity (greedy centroid algorithm, numpy only).
    4. Label each cluster via LLM (short 2–4 word topic phrase).
    5. Write cluster_id + cluster_label back to the events table.

The greedy centroid algorithm runs in O(n·k) where k is the number of
clusters found — efficient for hundreds to low thousands of events.

No external ML library required beyond numpy (already a transitive
dependency via pgvector).

Run: periodically via the Scheduler (e.g. every 6 hours).
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.db.models import Event
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.episodic import EpisodicMemory

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_EVENTS_TO_CLUSTER = 5       # skip clustering if fewer events exist
DEFAULT_SIMILARITY_THRESHOLD = 0.75  # events above this similarity share a cluster
MAX_LABEL_EXAMPLES = 5          # events shown to LLM for cluster labelling

_LABEL_SCHEMA = (
    "label (string): a 2-4 word topic phrase describing what these memories are about"
)
_LABEL_EXAMPLE = {"label": "AI infrastructure projects"}


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class ClusterResult:
    user_id: str
    app_id: str
    events_processed: int = 0
    clusters_found: int = 0
    events_clustered: int = 0
    skipped: bool = False
    skip_reason: str = ""


# ── MemoryClusterer ───────────────────────────────────────────────────────────


class MemoryClusterer:
    """
    Groups episodic events by embedding similarity and labels each cluster.

    Usage:
        clusterer = MemoryClusterer(llm=llm, episodic=episodic)

        async with db_session() as pg:
            result = await clusterer.run(pg, user_id="u1")
    """

    def __init__(
        self,
        llm: LLMAdapter,
        episodic: EpisodicMemory,
        *,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        min_events: int = MIN_EVENTS_TO_CLUSTER,
        audit=None,   # AuditLogger | None
    ) -> None:
        self.llm = llm
        self.episodic = episodic
        self.similarity_threshold = similarity_threshold
        self.min_events = min_events
        self.audit = audit

    # ── Primary entry point ────────────────────────────────────────────────

    async def run(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        app_id: str = "default",
    ) -> ClusterResult:
        """
        Run one clustering cycle for a single user.

        Steps:
            1. Fetch all events with embeddings.
            2. Guard: skip if fewer than min_events.
            3. Assign cluster IDs via greedy centroid algorithm.
            4. Label each cluster via LLM.
            5. Bulk-write cluster_id + cluster_label back to Postgres.
        """
        result = ClusterResult(user_id=user_id, app_id=app_id)

        # ── 1. Fetch events with embeddings ───────────────────────────────
        events = await _fetch_events_with_embeddings(session, user_id, app_id)
        result.events_processed = len(events)

        # ── 2. Guard ──────────────────────────────────────────────────────
        if len(events) < self.min_events:
            result.skipped = True
            result.skip_reason = (
                f"Only {len(events)} events with embeddings — "
                f"need at least {self.min_events}."
            )
            logger.debug(
                "Clustering skipped",
                extra={"user_id": user_id, "reason": result.skip_reason},
            )
            return result

        # ── 3. Cluster ────────────────────────────────────────────────────
        embeddings = [list(e.embedding) for e in events]
        assignments = cluster_embeddings(embeddings, self.similarity_threshold)

        # Group event IDs by cluster assignment
        cluster_groups: dict[int, list[Event]] = {}
        for event, cluster_id in zip(events, assignments):
            cluster_groups.setdefault(cluster_id, []).append(event)

        result.clusters_found = len(cluster_groups)

        # ── 4. Label and write back ───────────────────────────────────────
        now = datetime.now(timezone.utc)
        for cluster_id, cluster_events in cluster_groups.items():
            label = await self._label_cluster(user_id, cluster_id, cluster_events)
            event_ids = [e.id for e in cluster_events]
            await session.execute(
                update(Event)
                .where(Event.id.in_(event_ids))
                .values(
                    cluster_id=cluster_id,
                    cluster_label=label,
                    updated_at=now,
                )
            )
            result.events_clustered += len(cluster_events)

        logger.info(
            "Clustering complete",
            extra={
                "user_id": user_id,
                "events_processed": result.events_processed,
                "clusters_found": result.clusters_found,
                "events_clustered": result.events_clustered,
            },
        )

        if self.audit and result.clusters_found:
            from smritikosh.audit.logger import AuditEvent, EventType
            cluster_summary = [
                {
                    "cluster_id": cid,
                    "label": cluster_events[0].cluster_label or f"cluster_{cid}",
                    "event_count": len(cluster_events),
                }
                for cid, cluster_events in cluster_groups.items()
            ]
            await self.audit.emit(AuditEvent(
                event_type=EventType.MEMORY_CLUSTERED,
                user_id=user_id,
                app_id=app_id,
                payload={
                    "events_processed": result.events_processed,
                    "clusters_found": result.clusters_found,
                    "events_clustered": result.events_clustered,
                    "clusters": cluster_summary,
                },
            ))

        return result

    # ── Helpers ────────────────────────────────────────────────────────────

    async def _label_cluster(
        self, user_id: str, cluster_id: int, events: list[Event]
    ) -> str:
        """Generate a short topic label for a cluster via LLM, with fallback."""
        try:
            prompt = _build_label_prompt(events)
            extracted = await self.llm.extract_structured(
                prompt=prompt,
                schema_description=_LABEL_SCHEMA,
                example_output=_LABEL_EXAMPLE,
            )
            label = extracted.get("label", "").strip()
            return label or f"cluster_{cluster_id}"
        except Exception as exc:
            logger.warning(
                "Cluster labelling failed — using fallback",
                extra={"user_id": user_id, "cluster_id": cluster_id, "error": str(exc)},
            )
            return f"cluster_{cluster_id}"


# ── Clustering algorithm ──────────────────────────────────────────────────────


def cluster_embeddings(
    embeddings: list[list[float]],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[int]:
    """
    Greedy centroid-based clustering.

    For each embedding, compute cosine similarity to all existing cluster
    centroids. If the best similarity meets the threshold, assign to that
    cluster and update its centroid (running average). Otherwise start a
    new cluster.

    Args:
        embeddings:           List of float vectors (must all have the same length).
        similarity_threshold: Minimum cosine similarity to join an existing cluster.

    Returns:
        List of integer cluster IDs, one per input embedding.
        IDs are 0-indexed and contiguous (0, 1, 2, …).
    """
    if not embeddings:
        return []

    vecs = [np.array(e, dtype=np.float32) for e in embeddings]
    centroids: list[np.ndarray] = []
    centroid_counts: list[int] = []
    assignments: list[int] = []

    for vec in vecs:
        if not centroids:
            centroids.append(vec.copy())
            centroid_counts.append(1)
            assignments.append(0)
            continue

        sims = [_cosine_sim(vec, c) for c in centroids]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= similarity_threshold:
            n = centroid_counts[best_idx]
            centroids[best_idx] = (centroids[best_idx] * n + vec) / (n + 1)
            centroid_counts[best_idx] += 1
            assignments.append(best_idx)
        else:
            centroids.append(vec.copy())
            centroid_counts.append(1)
            assignments.append(len(centroids) - 1)

    return assignments


# ── Private helpers ───────────────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _build_label_prompt(events: list[Event]) -> str:
    samples = events[:MAX_LABEL_EXAMPLES]
    lines = [
        "What is the common topic of these memory entries? "
        "Provide a 2-4 word label.\n\nMemories:"
    ]
    for event in samples:
        text = (event.summary or event.raw_text)[:120]
        lines.append(f"- {text}")
    return "\n".join(lines)


async def _fetch_events_with_embeddings(
    session: AsyncSession, user_id: str, app_id: str
) -> list[Event]:
    result = await session.execute(
        select(Event)
        .where(
            Event.user_id == user_id,
            Event.app_id == app_id,
            Event.embedding.is_not(None),
        )
        .order_by(Event.created_at.asc())
    )
    return list(result.scalars().all())
