# client.py
import httpx
import os

BASE_URL = os.getenv("SMRITIKOSH_URL", "http://localhost:8080")


class SmritikoshClient:
    """
    Minimal sync client for the Smritikosh REST API.

    Two authentication modes:

    1. Username + password (exchanges credentials for a short-lived JWT):
        client = SmritikoshClient(username="alice", password="secret")

    2. API key (no login round-trip, key never expires unless revoked):
        client = SmritikoshClient(api_key="sk-smriti-...")

    The API key can also be set via the SMRITIKOSH_API_KEY environment variable:
        client = SmritikoshClient()   # reads from env
    """

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        *,
        api_key: str | None = None,
        app_id: str = "default",
    ):
        self.app_id = app_id

        # API key takes priority; fall back to env var; then username/password
        resolved_key = api_key or os.getenv("SMRITIKOSH_API_KEY")

        if resolved_key:
            token = resolved_key
        elif username and password:
            token = self._login(username, password)
        else:
            raise ValueError(
                "Provide either (username + password) or an api_key "
                "(or set SMRITIKOSH_API_KEY in your environment)."
            )

        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login(self, username: str, password: str) -> str:
        resp = httpx.post(
            f"{BASE_URL}/auth/token",
            json={"username": username, "password": password},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]

    # ── Core API ──────────────────────────────────────────────────────────────

    def remember(self, user_id: str, text: str) -> dict:
        """Store a piece of text as a memory event."""
        resp = httpx.post(
            f"{BASE_URL}/memory/event",
            headers=self._headers,
            json={"user_id": user_id, "content": text, "app_id": self.app_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_context(self, user_id: str, query: str) -> str:
        """Retrieve the memory context block for an LLM call."""
        resp = httpx.post(
            f"{BASE_URL}/context",
            headers=self._headers,
            json={"user_id": user_id, "query": query, "app_id": self.app_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("context_text", "")

    def search(self, user_id: str, query: str, limit: int = 5) -> list[dict]:
        """Search a user's memories and return scored results."""
        resp = httpx.post(
            f"{BASE_URL}/memory/search",
            headers=self._headers,
            json={
                "user_id": user_id,
                "query": query,
                "app_id": self.app_id,
                "limit": limit,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
