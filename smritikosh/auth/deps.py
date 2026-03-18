"""
FastAPI auth dependencies.

get_current_user      — Bearer JWT **or** API key → identity dict.
require_admin         — raises 403 if not admin.
assert_self_or_admin  — raises 403 if caller is not the target user or an admin.

Usage in routes:
    @router.get("/protected")
    async def protected(user = Depends(get_current_user)):
        return {"user_id": user["sub"]}

    @router.post("/admin-only")
    async def admin_only(user = Depends(require_admin)):
        ...

    @router.get("/memory/{user_id}")
    async def get_memory(user_id: str, user = Depends(get_current_user)):
        assert_self_or_admin(user, user_id)
        ...
"""

from datetime import datetime, timezone
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.auth.utils import hash_api_key, is_api_key, verify_token
from smritikosh.config import settings
from smritikosh.db.postgres import get_session

_bearer = HTTPBearer(auto_error=False)

_BOOTSTRAP_PAYLOAD = {"sub": "__bootstrap__", "role": "admin", "app_id": "default"}


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """
    Resolve a Bearer token to an identity dict { sub, role, app_id }.

    Accepts two token formats:
      • JWT  — validated via signature; fast, no DB hit.
      • API key (sk-smriti-…) — hashed and looked up in api_keys table.

    Raises 401 if missing, expired, invalid, or revoked.
    The returned dict always contains: sub (username), role, app_id.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── API key path ──────────────────────────────────────────────────────────
    if is_api_key(token):
        from smritikosh.db.models import ApiKey, AppUser  # avoid circular at module level

        key_hash = hash_api_key(token)
        result = await pg.execute(
            select(ApiKey, AppUser)
            .join(AppUser, AppUser.username == ApiKey.user_id)
            .where(ApiKey.key_hash == key_hash)
            .where(ApiKey.revoked_at.is_(None))
            .where(AppUser.is_active.is_(True))
        )
        row = result.first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or revoked API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        api_key, app_user = row

        # Update last_used_at asynchronously (best-effort)
        await pg.execute(
            update(ApiKey)
            .where(ApiKey.id == api_key.id)
            .values(last_used_at=datetime.now(timezone.utc))
        )

        return {"sub": app_user.username, "role": app_user.role, "app_id": api_key.app_id}

    # ── JWT path ──────────────────────────────────────────────────────────────
    try:
        payload = verify_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload


async def require_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    pg: Annotated[AsyncSession, Depends(get_session)],
) -> dict:
    """
    Extend get_current_user — additionally enforce the admin role.

    When BOOTSTRAP_ADMIN=1 and no token is provided, returns a synthetic
    admin payload so the very first admin account can be created.
    Raises 403 if the authenticated user is not an admin.
    """
    if credentials is None:
        if settings.bootstrap_admin:
            return _BOOTSTRAP_PAYLOAD
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_current_user(credentials, pg)

    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user


def assert_self_or_admin(current_user: dict, requested_user_id: str) -> None:
    """
    Raise HTTP 403 if `current_user` is not the owner of `requested_user_id`
    and is not an admin.

    Call this inside any route that takes a user_id parameter.
    """
    if current_user["role"] == "admin":
        return
    if current_user["sub"] != requested_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied.",
        )
