"""
GET /identity/{user_id} — synthesized user identity from semantic memory.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from neo4j import AsyncSession as NeoSession
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.api.deps import get_identity_builder, get_neo4j_session
from smritikosh.api.schemas import BeliefItem, IdentityDimensionItem, IdentityResponse
from smritikosh.db.postgres import get_session
from smritikosh.memory.identity import IdentityBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/identity", tags=["identity"])


@router.get("/{user_id}", response_model=IdentityResponse)
async def get_identity(
    user_id: str,
    app_id: str = "default",
    pg: AsyncSession = Depends(get_session),
    neo: NeoSession = Depends(get_neo4j_session),
    builder: IdentityBuilder = Depends(get_identity_builder),
    current_user: dict = Depends(get_current_user),
) -> IdentityResponse:
    """
    Return the synthesized identity model for a user.

    Aggregates all semantic facts from Neo4j, groups them by category,
    generates a narrative summary via LLM, and includes any inferred
    beliefs from the user_beliefs table.
    """
    assert_self_or_admin(current_user, user_id)
    try:
        identity = await builder.build(
            neo, user_id=user_id, app_id=app_id, pg_session=pg
        )
    except Exception as exc:
        logger.error(
            "Identity build failed",
            extra={"user_id": user_id, "error": str(exc)},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return IdentityResponse(
        user_id=identity.user_id,
        app_id=identity.app_id,
        summary=identity.summary,
        dimensions=[
            IdentityDimensionItem(
                category=dim.category,
                dominant_value=dim.dominant_value,
                confidence=dim.confidence,
                fact_count=len(dim.facts),
            )
            for dim in identity.dimensions
        ],
        beliefs=[
            BeliefItem(
                statement=b.statement,
                category=b.category,
                confidence=b.confidence,
                evidence_count=b.evidence_count,
            )
            for b in identity.beliefs
        ],
        total_facts=identity.total_facts,
        computed_at=identity.computed_at.isoformat(),
        is_empty=identity.is_empty(),
    )
