"""
Auth utilities — password hashing, JWT creation/verification, and API key generation.

Uses bcrypt directly for passwords, PyJWT for tokens, and SHA-256 + secrets
for API keys.  All functions are stateless — no DB calls, safe to call from anywhere.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from smritikosh.config import settings

_KEY_PREFIX = "sk-smriti-"


# ── Passwords ─────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the plain-text password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if `plain` matches the stored `hashed` password."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_access_token(
    user_id: str,
    role: str,
    app_ids: list[str],
    *,
    expire_days: int | None = None,
) -> str:
    """
    Create a signed JWT carrying { sub, role, app_ids, exp }.

    The `sub` claim is the username, which is also the `user_id` used
    throughout the memory system.
    """
    days = expire_days if expire_days is not None else settings.jwt_expire_days
    exp = datetime.now(timezone.utc) + timedelta(days=days)
    payload = {
        "sub": user_id,
        "role": role,
        "app_ids": app_ids,
        "exp": exp,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


# ── API keys ──────────────────────────────────────────────────────────────────


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns (full_key, key_hash, key_prefix) where:
        full_key   — the complete key shown to the user once, e.g. sk-smriti-a1b2c3...
        key_hash   — SHA-256 hex digest, stored in DB for lookup
        key_prefix — first 8 hex chars of the random part, stored for display
    """
    random_part = secrets.token_hex(24)          # 48 hex chars = 192 bits entropy
    full_key = f"{_KEY_PREFIX}{random_part}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = random_part[:8]
    return full_key, key_hash, key_prefix


def hash_api_key(full_key: str) -> str:
    """Return the SHA-256 hex digest of a full API key for DB lookup."""
    return hashlib.sha256(full_key.encode()).hexdigest()


def is_api_key(token: str) -> bool:
    """Return True if the Bearer token looks like an API key (not a JWT)."""
    return token.startswith(_KEY_PREFIX)


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
