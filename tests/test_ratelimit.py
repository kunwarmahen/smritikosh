"""Tests for the rate limiter storage backend selection (item A2).

Covers:
- using_persistent_storage() / _build_limiter() — Redis vs. in-memory selection
- _user_key() — rate-limit key extraction from JWT / API key / client IP
- _warn_runtime_topology() — multi-replica warning when REDIS_URL is unset
"""

import base64
import json

from slowapi import Limiter

# Import main at collection time so logging.basicConfig(force=True) runs before
# any per-test caplog handler is installed (otherwise it would be wiped).
from smritikosh.api import main, ratelimit


def _fake_request(headers=None, client_host="1.2.3.4"):
    """Minimal stand-in for a Starlette Request (only what _user_key reads)."""

    class _Client:
        host = client_host

    class _Req:
        def __init__(self):
            self.headers = headers or {}
            self.client = _Client() if client_host else None

    return _Req()


class TestStorageSelection:
    def test_in_memory_when_redis_url_unset(self, monkeypatch):
        monkeypatch.setattr(ratelimit.settings, "redis_url", None)
        assert ratelimit.using_persistent_storage() is False
        assert isinstance(ratelimit._build_limiter(), Limiter)

    def test_persistent_when_redis_url_set(self, monkeypatch):
        monkeypatch.setattr(ratelimit.settings, "redis_url", "redis://localhost:6379/0")
        assert ratelimit.using_persistent_storage() is True
        # Limiter is built lazily — no live Redis connection is required here.
        assert isinstance(ratelimit._build_limiter(), Limiter)


class TestUserKey:
    def test_api_key_uses_token_directly(self):
        req = _fake_request({"Authorization": "Bearer sk-smriti-abc123"})
        assert ratelimit._user_key(req) == "apikey:sk-smriti-abc123"

    def test_jwt_uses_sub_claim(self):
        payload = base64.urlsafe_b64encode(json.dumps({"sub": "alice"}).encode()).decode().rstrip("=")
        token = f"header.{payload}.signature"
        req = _fake_request({"Authorization": f"Bearer {token}"})
        assert ratelimit._user_key(req) == "user:alice"

    def test_no_auth_falls_back_to_client_ip(self):
        req = _fake_request(headers={}, client_host="9.9.9.9")
        assert ratelimit._user_key(req) == "9.9.9.9"

    def test_x_forwarded_for_is_honoured(self):
        req = _fake_request({"X-Forwarded-For": "203.0.113.7, 10.0.0.1"})
        assert ratelimit._user_key(req) == "203.0.113.7"

    def test_malformed_jwt_falls_back_to_ip(self):
        req = _fake_request({"Authorization": "Bearer not-a-jwt"}, client_host="5.5.5.5")
        assert ratelimit._user_key(req) == "5.5.5.5"


class TestRuntimeTopologyWarning:
    def test_production_without_redis_warns(self, monkeypatch, caplog):
        from smritikosh.config import Settings

        caplog.set_level("INFO", logger="smritikosh.api.main")
        monkeypatch.setattr(main, "settings", Settings(app_env="production", jwt_secret="x" * 40))
        monkeypatch.setattr(main, "using_persistent_storage", lambda: False)
        main._warn_runtime_topology()
        assert "REDIS_URL" in caplog.text
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_development_without_redis_only_logs_info(self, monkeypatch, caplog):
        from smritikosh.config import Settings

        caplog.set_level("INFO", logger="smritikosh.api.main")
        monkeypatch.setattr(main, "settings", Settings(app_env="development", jwt_secret="x" * 40))
        monkeypatch.setattr(main, "using_persistent_storage", lambda: False)
        main._warn_runtime_topology()
        assert "REDIS_URL" in caplog.text
        assert not any(r.levelname == "WARNING" for r in caplog.records)

    def test_redis_configured_emits_no_warning(self, monkeypatch, caplog):
        caplog.set_level("INFO", logger="smritikosh.api.main")
        monkeypatch.setattr(main, "using_persistent_storage", lambda: True)
        main._warn_runtime_topology()
        assert "REDIS_URL" not in caplog.text
