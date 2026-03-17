# client.py
import httpx
import os

BASE_URL = os.getenv("SMRITIKOSH_URL", "http://localhost:8080")


class SmritikoshClient:
    """Minimal sync client for the Smritikosh REST API."""

    def __init__(self, username: str, password: str, app_id: str = "default"):
        self.app_id = app_id
        self._token = self._login(username, password)
        self._headers = {
            "Authorization": f"Bearer {self._token}",
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
