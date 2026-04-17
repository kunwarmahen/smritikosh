"""
GET /graph/facts/{user_id} — fact graph for a user from Neo4j.

Returns nodes (user + fact) and edges (user→fact relations, fact→fact
RELATED_TO links) suitable for direct React Flow rendering.
"""

import hashlib
import logging

from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncSession as NeoSession

from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_neo4j_session
from smritikosh.api.schemas import FactGraphEdge, FactGraphNode, FactGraphResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["graph"])


def _fact_id(category: str, key: str, value: str) -> str:
    """Stable, URL-safe node ID derived from the fact triple."""
    digest = hashlib.md5(f"{category}|{key}|{value}".encode()).hexdigest()[:12]
    return f"fact-{digest}"


@router.get("/facts/{user_id}", response_model=FactGraphResponse)
async def get_fact_graph(
    user_id: str,
    app_id: str = "default",
    min_confidence: float = 0.0,
    neo: NeoSession = Depends(get_neo4j_session),
    current_user: dict = Depends(get_current_user),
) -> FactGraphResponse:
    """
    Return the full fact graph for a user.

    Nodes:
      - One "user" node
      - One "fact" node per unique (category, key, value) triple

    Edges:
      - User → Fact  (typed by category: HAS_PREFERENCE, HAS_INTEREST, …)
      - Fact → Fact  (RELATED_TO links between facts belonging to this user)
    """
    assert_self_or_admin(current_user, user_id)
    try:
        # ── 1. User → Fact edges ──────────────────────────────────────────────
        user_fact_query = """
        MATCH (u:User {user_id: $user_id, app_id: $app_id})-[r]->(f:Fact)
        WHERE r.confidence >= $min_confidence
        RETURN
            f.category          AS category,
            f.key               AS key,
            f.value             AS value,
            r.confidence        AS confidence,
            r.frequency_count   AS frequency_count,
            r.source_event_ids  AS source_event_ids,
            type(r)             AS rel_type
        ORDER BY r.frequency_count DESC
        """
        user_fact_result = await neo.run(
            user_fact_query,
            user_id=user_id,
            app_id=app_id,
            min_confidence=min_confidence,
        )
        user_fact_records = await user_fact_result.data()

        if not user_fact_records:
            # User exists but has no facts — return graph with just the user node
            return FactGraphResponse(
                user_id=user_id,
                app_id=app_id,
                nodes=[FactGraphNode(id=f"user-{user_id}", label=user_id, node_type="user")],
                edges=[],
            )

        # ── 2. Fact → Fact edges (RELATED_TO) ────────────────────────────────
        fact_rel_query = """
        MATCH (u:User {user_id: $user_id, app_id: $app_id})-[]->(f1:Fact)
        MATCH (u)-[]->(f2:Fact)
        OPTIONAL MATCH (f1)-[r:RELATED_TO]->(f2)
        WITH f1, f2, r WHERE r IS NOT NULL
        RETURN
            f1.category AS from_cat,
            f1.key      AS from_key,
            f1.value    AS from_value,
            f2.category AS to_cat,
            f2.key      AS to_key,
            f2.value    AS to_value,
            r.strength  AS strength
        """
        fact_rel_result = await neo.run(
            fact_rel_query, user_id=user_id, app_id=app_id
        )
        fact_rel_records = await fact_rel_result.data()

    except Exception as exc:
        logger.error(
            "Fact graph query failed",
            extra={"user_id": user_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    # ── Build node list ───────────────────────────────────────────────────────
    user_node_id = f"user-{user_id}"
    seen_fact_ids: set[str] = set()
    nodes: list[FactGraphNode] = [
        FactGraphNode(id=user_node_id, label=user_id, node_type="user")
    ]
    edges: list[FactGraphEdge] = []

    for i, rec in enumerate(user_fact_records):
        fid = _fact_id(rec["category"], rec["key"], rec["value"])

        if fid not in seen_fact_ids:
            seen_fact_ids.add(fid)
            nodes.append(
                FactGraphNode(
                    id=fid,
                    label=str(rec["value"]),
                    node_type="fact",
                    category=rec["category"],
                    key=rec.get("key"),
                    confidence=rec["confidence"],
                    frequency_count=rec["frequency_count"],
                    source_event_ids=list(rec.get("source_event_ids") or []),
                )
            )

        edges.append(
            FactGraphEdge(
                id=f"uf-{i}",
                source=user_node_id,
                target=fid,
                relation=rec["rel_type"],
            )
        )

    # ── Build fact→fact edges ─────────────────────────────────────────────────
    for i, rec in enumerate(fact_rel_records):
        src_id = _fact_id(rec["from_cat"], rec["from_key"], rec["from_value"])
        tgt_id = _fact_id(rec["to_cat"], rec["to_key"], rec["to_value"])
        # Only include if both nodes are in the graph
        if src_id in seen_fact_ids and tgt_id in seen_fact_ids:
            edges.append(
                FactGraphEdge(
                    id=f"ff-{i}",
                    source=src_id,
                    target=tgt_id,
                    relation="RELATED_TO",
                    strength=rec["strength"],
                )
            )

    return FactGraphResponse(
        user_id=user_id,
        app_id=app_id,
        nodes=nodes,
        edges=edges,
    )
