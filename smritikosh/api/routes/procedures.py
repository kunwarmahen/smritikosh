"""
Procedural memory routes — CRUD for behavioral rules.

POST   /procedures             Create a behavioral rule for a user.
GET    /procedures/{user_id}   List all rules for a user.
PATCH  /procedures/{id}        Update a rule (any fields optional).
DELETE /procedures/{id}        Delete a specific rule.
DELETE /procedures/user/{uid}  Wipe all rules for a user.
"""

import logging
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_procedural
from smritikosh.api.schemas import (
    DeleteProcedureResponse,
    DeleteUserProceduresResponse,
    ProcedureItem,
    ProcedureListResponse,
    ProcedureRequest,
    ProcedureResponse,
    ProcedureUpdateRequest,
)
from smritikosh.db.postgres import get_session
from smritikosh.memory.procedural import ProceduralMemory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/procedures", tags=["procedural"])


@router.post("", response_model=ProcedureResponse, status_code=201)
async def create_procedure(
    request: ProcedureRequest,
    procedural: Annotated[ProceduralMemory, Depends(get_procedural)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> ProcedureResponse:
    """
    Store a behavioral rule for a user.

    The rule fires automatically whenever the trigger phrase is detected
    in a context query, injecting the instruction into the assembled
    memory context before the LLM call.

    Example:
        trigger="LLM deployment"
        instruction="mention GPU optimization, batching, and quantization"
    """
    proc = await procedural.store(
        pg,
        user_id=request.user_id,
        trigger=request.trigger,
        instruction=request.instruction,
        app_id=request.app_id,
        category=request.category,
        priority=request.priority,
        confidence=request.confidence,
        source=request.source,
    )
    return ProcedureResponse(
        procedure_id=str(proc.id),
        user_id=proc.user_id,
        trigger=proc.trigger,
        instruction=proc.instruction,
        category=proc.category,
        priority=proc.priority,
        is_active=proc.is_active,
        hit_count=proc.hit_count,
        confidence=proc.confidence,
        source=proc.source,
        created_at=proc.created_at.isoformat() if proc.created_at else "",
    )


@router.get("/{user_id}", response_model=ProcedureListResponse)
async def list_procedures(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    active_only: Annotated[bool, Query(description="Only return active rules")] = True,
    category: Annotated[Optional[str], Query(description="Filter by category")] = None,
    procedural: Annotated[ProceduralMemory, Depends(get_procedural)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
) -> ProcedureListResponse:
    """Return all behavioral rules for a user, ordered by priority descending."""
    procs = await procedural.get_all(
        pg, user_id, app_id, active_only=active_only, category=category
    )
    return ProcedureListResponse(
        user_id=user_id,
        app_id=app_id,
        procedures=[
            ProcedureItem(
                procedure_id=str(p.id),
                trigger=p.trigger,
                instruction=p.instruction,
                category=p.category,
                priority=p.priority,
                is_active=p.is_active,
                hit_count=p.hit_count,
            )
            for p in procs
        ],
    )


@router.patch("/{procedure_id}", response_model=ProcedureResponse)
async def update_procedure(
    procedure_id: str,
    request: ProcedureUpdateRequest,
    procedural: Annotated[ProceduralMemory, Depends(get_procedural)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> ProcedureResponse:
    """
    Partially update a behavioral rule.

    Only fields present in the request body are updated.
    Set ``is_active=false`` to disable without deleting.
    """
    try:
        pid = uuid.UUID(procedure_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid procedure_id UUID format.")

    proc = await procedural.update(
        pg,
        pid,
        trigger=request.trigger,
        instruction=request.instruction,
        category=request.category,
        priority=request.priority,
        is_active=request.is_active,
        confidence=request.confidence,
    )
    if proc is None:
        raise HTTPException(status_code=404, detail=f"Procedure {procedure_id} not found.")

    return ProcedureResponse(
        procedure_id=str(proc.id),
        user_id=proc.user_id,
        trigger=proc.trigger,
        instruction=proc.instruction,
        category=proc.category,
        priority=proc.priority,
        is_active=proc.is_active,
        hit_count=proc.hit_count,
        confidence=proc.confidence,
        source=proc.source,
        created_at=proc.created_at.isoformat() if proc.created_at else "",
    )


@router.delete("/user/{user_id}", response_model=DeleteUserProceduresResponse)
async def delete_user_procedures(
    user_id: str,
    app_id: Annotated[str, Query(description="Application namespace")] = "default",
    procedural: Annotated[ProceduralMemory, Depends(get_procedural)] = None,
    pg: Annotated[AsyncSession, Depends(get_session)] = None,
) -> DeleteUserProceduresResponse:
    """Delete all behavioral rules for a user within an app namespace."""
    count = await procedural.delete_all_for_user(pg, user_id, app_id)
    logger.info(
        "Deleted all user procedures",
        extra={"user_id": user_id, "app_id": app_id, "count": count},
    )
    return DeleteUserProceduresResponse(
        procedures_deleted=count, user_id=user_id, app_id=app_id
    )


@router.delete("/{procedure_id}", response_model=DeleteProcedureResponse)
async def delete_procedure(
    procedure_id: str,
    procedural: Annotated[ProceduralMemory, Depends(get_procedural)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> DeleteProcedureResponse:
    """Delete a specific behavioral rule by ID."""
    try:
        pid = uuid.UUID(procedure_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid procedure_id UUID format.")

    deleted = await procedural.delete(pg, pid)
    return DeleteProcedureResponse(deleted=deleted, procedure_id=procedure_id)
