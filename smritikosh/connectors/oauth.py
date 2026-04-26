"""OAuth2 utilities for Google connectors.

Handles:
- State JWT encoding/decoding (secure round-trip of user_id+app_id during OAuth flow)
- Fernet encryption/decryption for storing access/refresh tokens
- Google OAuth2 token exchange and refresh via httpx
"""

import base64
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx
import jwt
from cryptography.fernet import Fernet, InvalidToken

from smritikosh.config import settings

logger = logging.getLogger(__name__)

# Google OAuth2 endpoints
_GOOGLE_AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"

# OAuth scopes for Gmail and Calendar
DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",  # Gmail: read messages
    "https://www.googleapis.com/auth/calendar.readonly",  # Calendar: read events
]


# ── Fernet token encryption ────────────────────────────────────────────────────


def _get_fernet_key() -> bytes:
    """Derive a Fernet key from settings.jwt_secret.

    SHA-256 produces 32 raw bytes; Fernet requires those bytes as URL-safe base64,
    which is what base64.urlsafe_b64encode produces (44 chars from 32 bytes).
    """
    key_bytes = hashlib.sha256(settings.jwt_secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key_bytes)


def encrypt_tokens(tokens: dict[str, Any]) -> str:
    """
    Encrypt a token dict to a Fernet-protected string.

    tokens dict should contain: access_token, refresh_token, expires_in, token_type, etc.
    Returns a URL-safe Fernet string.
    """
    key = _get_fernet_key()
    fernet = Fernet(key)
    payload = json.dumps(tokens, default=str).encode("utf-8")
    encrypted = fernet.encrypt(payload)
    return encrypted.decode("utf-8")


def decrypt_tokens(encrypted: str) -> dict[str, Any]:
    """
    Decrypt a Fernet-protected string back to a token dict.

    Raises InvalidToken if decryption fails (e.g. wrong key, corrupted data).
    """
    key = _get_fernet_key()
    fernet = Fernet(key)
    try:
        payload = fernet.decrypt(encrypted.encode("utf-8"))
        return json.loads(payload.decode("utf-8"))
    except InvalidToken:
        logger.error("Token decryption failed (wrong key or corrupted data)")
        raise


# ── OAuth state JWT ────────────────────────────────────────────────────────────


def build_state_jwt(user_id: str, app_id: str, *, expire_hours: int = 1) -> str:
    """
    Encode user_id + app_id + expiry into a signed JWT for the OAuth state param.

    This avoids needing to store state in a session or database.
    """
    payload = {
        "user_id": user_id,
        "app_id": app_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=expire_hours),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_state_jwt(state: str) -> tuple[str, str]:
    """
    Decode and verify an OAuth state JWT.

    Returns (user_id, app_id) or raises jwt.InvalidTokenError on failure.
    """
    try:
        payload = jwt.decode(state, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload["user_id"], payload["app_id"]
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid OAuth state JWT: {e}")
        raise


# ── Google OAuth2 flow ─────────────────────────────────────────────────────────


def build_authorization_url(
    state: str,
    scopes: Optional[list[str]] = None,
) -> str:
    """
    Build a Google OAuth2 authorization URL.

    Args:
        state: Signed JWT containing user_id+app_id (verified in callback)
        scopes: List of OAuth scopes to request (defaults to DEFAULT_SCOPES)

    Returns: Full authorization URL for the user to visit.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not configured")

    if scopes is None:
        scopes = DEFAULT_SCOPES

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "state": state,
        "access_type": "offline",  # Request refresh token
        "prompt": "consent",  # Force consent screen to ensure refresh token
    }
    return f"{_GOOGLE_AUTH_URI}?{urlencode(params)}"


async def exchange_code(code: str) -> dict[str, Any]:
    """
    Exchange an OAuth2 authorization code for access/refresh tokens.

    Calls Google's token endpoint with the code.
    Returns token dict: {access_token, refresh_token, expires_in, token_type, ...}
    Raises httpx.HTTPError on failure.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            _GOOGLE_TOKEN_URI,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        response.raise_for_status()
        tokens = response.json()
        logger.debug(f"Exchanged authorization code for tokens (expires_in={tokens.get('expires_in')})")
        return tokens


async def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """
    Refresh an expired access token using the refresh token.

    Returns token dict with updated access_token and expires_in.
    Raises httpx.HTTPError on failure.
    """
    if not settings.google_client_id or not settings.google_client_secret:
        raise ValueError("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET not configured")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            _GOOGLE_TOKEN_URI,
            data={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
        )
        response.raise_for_status()
        tokens = response.json()
        logger.debug(f"Refreshed access token (expires_in={tokens.get('expires_in')})")
        return tokens
