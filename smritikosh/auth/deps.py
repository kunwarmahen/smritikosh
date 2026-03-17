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
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from smritikosh.auth.utils import verify_token

_bearer = HTTPBearer(auto_error=True)


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> dict:
    """
    Validate the Bearer JWT and return its payload.

    Raises 401 if the token is missing, expired, or invalid.
    The returned dict always contains: sub (user_id), role, app_id.
    """
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
    user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    """
    Extend get_current_user — additionally enforce the admin role.

    Raises 403 if the authenticated user is not an admin.
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return user
