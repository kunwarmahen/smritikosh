"""
Auth routes — login, register, and current-user info.

POST /auth/token     Login with username + password → JWT
POST /auth/register  Create a new user (admin only)
GET  /auth/me        Return the currently authenticated user's profile
"""

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.schemas import RegisterRequest, TokenRequest, TokenResponse, UserResponse
from smritikosh.auth.deps import get_current_user, require_admin
from smritikosh.auth.utils import create_access_token, hash_password, verify_password
from smritikosh.db.models import AppUser, UserRole
from smritikosh.db.postgres import get_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
async def login(
    request: TokenRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> TokenResponse:
    """
    Authenticate with username and password.

    Returns a Bearer JWT valid for `JWT_EXPIRE_DAYS` days (default 30).
    Include it in all subsequent requests as:
        Authorization: Bearer <token>
    """
    result = await pg.execute(
        select(AppUser).where(AppUser.username == request.username)
    )
    user: AppUser | None = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled.",
        )

    token = create_access_token(
        user_id=user.username,
        role=user.role,
        app_ids=user.app_ids,
    )

    logger.info("User logged in", extra={"username": user.username, "role": user.role})

    return TokenResponse(
        access_token=token,
        user_id=user.username,
        role=user.role,
        app_ids=user.app_ids,
    )


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(
    request: RegisterRequest,
    pg: Annotated[AsyncSession, Depends(get_session)],
    _admin: Annotated[dict, Depends(require_admin)],
) -> UserResponse:
    """
    Create a new user account.

    Admin access required. The `username` becomes the `user_id` used
    throughout the memory system — choose it carefully.
    """
    # Validate role
    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid role '{request.role}'. Must be 'user' or 'admin'.",
        )

    # Check for duplicate username
    existing = await pg.execute(
        select(AppUser).where(AppUser.username == request.username)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Username '{request.username}' is already taken.",
        )

    new_user = AppUser(
        username=request.username,
        email=request.email,
        password_hash=hash_password(request.password),
        role=role,
        app_ids=request.app_ids,
        is_active=True,
    )
    pg.add(new_user)
    await pg.flush()

    logger.info(
        "New user created",
        extra={"username": new_user.username, "role": new_user.role, "by": _admin["sub"]},
    )

    created_at = new_user.created_at or datetime.now(timezone.utc)
    return UserResponse(
        user_id=new_user.username,
        username=new_user.username,
        role=new_user.role,
        app_ids=new_user.app_ids,
        email=new_user.email,
        is_active=new_user.is_active,
        created_at=created_at.isoformat(),
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    pg: Annotated[AsyncSession, Depends(get_session)],
    current_user: Annotated[dict, Depends(get_current_user)],
) -> UserResponse:
    """
    Return the profile of the currently authenticated user.

    Useful for the UI to confirm role and app_id on startup.
    """
    result = await pg.execute(
        select(AppUser).where(AppUser.username == current_user["sub"])
    )
    user: AppUser | None = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")

    return UserResponse(
        user_id=user.username,
        username=user.username,
        role=user.role,
        app_ids=user.app_ids,
        email=user.email,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
    )
