"""
Rate limiting — per-user limits on expensive API endpoints.

Key function extracts user identity from the incoming request:
  - JWT Bearer token  → decoded (no signature check) to read the `sub` claim
  - API key (sk-smriti-…) → raw token string used as the key directly
  - No / invalid token → client IP address used as fallback

Using user identity (not IP) is important for a multi-tenant API:
  - Multiple users may share the same egress IP (corporate NAT, VPN)
  - A single user should have one rate limit regardless of IP changes

The limiter is wired into the FastAPI app in main.py.
Routes opt in by applying @limiter.limit("N/period") and adding a
`request: Request` parameter to their signature.

To disable rate limiting entirely, set the relevant env var to "":
    RATE_LIMIT_ENCODE=""
"""

import base64
import json
import logging

from slowapi import Limiter
from starlette.requests import Request

logger = logging.getLogger(__name__)


def _user_key(request: Request) -> str:
    """
    Extract a stable rate-limit key from the request.

    Priority:
      1. JWT `sub` claim (decoded without signature verification)
      2. Raw API key string (already unique per caller)
      3. Client IP address (fallback for unauthenticated requests)
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return _client_ip(request)

    token = auth_header[len("Bearer "):]

    # API keys start with "sk-smriti-" — use the token itself as the key
    if token.startswith("sk-smriti-"):
        return f"apikey:{token}"

    # JWT: base64-decode the payload segment to read `sub` without a DB hit.
    # We don't verify the signature here — real verification happens inside
    # get_current_user. An invalid JWT will fail auth before consuming quota.
    try:
        parts = token.split(".")
        if len(parts) == 3:
            payload_b64 = parts[1]
            # Add padding so base64 doesn't complain about incomplete bytes
            padded = payload_b64 + "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
    except Exception:
        pass

    return _client_ip(request)


def _client_ip(request: Request) -> str:
    """Return the client's IP, honouring X-Forwarded-For if present."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# Single shared limiter instance — imported by routes and wired into main.py.
# Uses an in-memory store (default); swap to Redis via storage_uri for multi-process.
limiter = Limiter(key_func=_user_key)
