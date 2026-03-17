"""
FastAPI auth dependencies.

get_current_user  — extracts and validates the Bearer JWT from the request.
require_admin     — raises 403 if the authenticated user is not an admin.

Usage in routes:
    @router.get("/protected")
    async def protected(user = Depends(get_current_user)):
        return {"user_id": user["sub"]}

    @router.post("/admin-only")
    async def admin_only(user = Depends(require_admin)):
        ...
"""

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from smritikosh.auth.utils import verify_token
from smritikosh.config import settings

_bearer = HTTPBearer(auto_error=False)

_BOOTSTRAP_PAYLOAD = {"sub": "__bootstrap__", "role": "admin", "app_id": "default"}


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> dict:
    """
    Validate the Bearer JWT and return its payload.

    Raises 401 if the token is missing, expired, or invalid.
    The returned dict always contains: sub (user_id), role, app_id.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = verify_token(credentials.credentials)
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

    try:
        user = verify_token(credentials.credentials)
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

    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user
