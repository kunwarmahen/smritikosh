"""
FastAPI auth dependencies.

get_current_user      — Bearer JWT **or** API key → identity dict.
require_admin         — raises 403 if not admin.
require_write_scope   — raises 403 if the caller's API key lacks the 'write' scope.
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

    # Restrict a write endpoint to keys that have the 'write' scope:
    @router.post("/memory/event")
    async def encode(user = Depends(require_write_scope)):
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

_BOOTSTRAP_PAYLOAD = {"sub": "__bootstrap__", "role": "admin", "app_ids": ["default"]}


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
    The returned dict always contains: sub (username), role, app_ids.
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

        scopes = api_key.scopes if api_key.scopes else ["read", "write"]
        return {"sub": app_user.username, "role": app_user.role, "app_ids": api_key.app_ids, "scopes": scopes}

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


def assert_app_access(current_user: dict, app_id: str) -> None:
    """Raise 403 if the token does not have access to app_id (admin bypasses)."""
    if current_user["role"] == "admin":
        return
    app_ids = current_user.get("app_ids", [])
    if app_id not in app_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access to app '{app_id}' denied.",
        )


async def require_write_scope(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Extend get_current_user — additionally enforce the 'write' scope.

    JWT tokens always have full access (no scope restriction).
    API keys with only 'read' scope are rejected with 403.
    """
    # JWTs have no scopes field — they always have full access
    scopes = current_user.get("scopes")
    if scopes is not None and "write" not in scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This API key does not have write access. Re-generate with scopes=['read','write'].",
        )
    return current_user
