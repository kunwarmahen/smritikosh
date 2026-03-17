"""
Auth utilities — password hashing and JWT creation/verification.

Uses passlib[bcrypt] for passwords and PyJWT for tokens.
Both are stateless — no DB calls, safe to call from anywhere.
"""

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from smritikosh.config import settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── Passwords ─────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plain-text password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches the stored `hashed` password."""
    return _pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_access_token(
    user_id: str,
    role: str,
    app_id: str,
    *,
    expire_days: int | None = None,
) -> str:
    """
    Create a signed JWT carrying { sub, role, app_id, exp }.

    The `sub` claim is the username, which is also the `user_id` used
    throughout the memory system.
    """
    days = expire_days if expire_days is not None else settings.jwt_expire_days
    exp = datetime.now(timezone.utc) + timedelta(days=days)
    payload = {
        "sub": user_id,
        "role": role,
        "app_id": app_id,
        "exp": exp,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> dict:
    """
    Decode and verify a JWT.

    Returns the payload dict on success.
    Raises jwt.InvalidTokenError (or subclass) on any failure:
        - jwt.ExpiredSignatureError   — token has expired
        - jwt.InvalidSignatureError   — tampered token
        - jwt.DecodeError             — malformed token
    """
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
    )
