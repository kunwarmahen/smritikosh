"""Tests for the CORS policy (item C2).

Covers:
- Settings.cors_origin_list — parsing of the comma-separated env value
- configure_cors() — middleware only registered when origins are set,
  explicit-allowlist behaviour, wildcard disables credentials
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from smritikosh.api.main import configure_cors
from smritikosh.config import Settings

_STRONG = "a" * 40


def _settings(**overrides) -> Settings:
    base = dict(jwt_secret=_STRONG, app_env="test")
    base.update(overrides)
    return Settings(**base)


def _app_with_cors(origins: list[str]) -> TestClient:
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    configure_cors(app, origins)
    return TestClient(app)


class TestCorsOriginParsing:
    def test_default_is_empty(self):
        assert _settings().cors_origin_list == []

    def test_empty_string_is_empty(self):
        assert _settings(cors_allowed_origins="").cors_origin_list == []

    def test_single_origin(self):
        s = _settings(cors_allowed_origins="https://app.example.com")
        assert s.cors_origin_list == ["https://app.example.com"]

    def test_multiple_origins_with_whitespace(self):
        s = _settings(cors_allowed_origins=" https://a.com , https://b.com ")
        assert s.cors_origin_list == ["https://a.com", "https://b.com"]

    def test_trailing_comma_ignored(self):
        s = _settings(cors_allowed_origins="https://a.com,")
        assert s.cors_origin_list == ["https://a.com"]

    def test_wildcard(self):
        assert _settings(cors_allowed_origins="*").cors_origin_list == ["*"]


class TestConfigureCors:
    def test_disabled_when_no_origins(self):
        app = FastAPI()
        assert configure_cors(app, []) is False

        @app.get("/ping")
        def ping():
            return {"ok": True}

        client = TestClient(app)
        resp = client.get("/ping", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in resp.headers

    def test_enabled_returns_true(self):
        assert configure_cors(FastAPI(), ["https://a.com"]) is True

    def test_allowlisted_origin_gets_cors_headers(self):
        client = _app_with_cors(["https://app.example.com"])
        resp = client.get("/ping", headers={"Origin": "https://app.example.com"})
        assert resp.headers["access-control-allow-origin"] == "https://app.example.com"
        # Explicit allowlist supports credentials
        assert resp.headers["access-control-allow-credentials"] == "true"

    def test_unlisted_origin_gets_no_cors_headers(self):
        client = _app_with_cors(["https://app.example.com"])
        resp = client.get("/ping", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in resp.headers

    def test_preflight_allowed_for_listed_origin(self):
        client = _app_with_cors(["https://app.example.com"])
        resp = client.options(
            "/ping",
            headers={
                "Origin": "https://app.example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "https://app.example.com"
        assert "authorization" in resp.headers["access-control-allow-headers"].lower()

    def test_wildcard_disables_credentials(self):
        client = _app_with_cors(["*"])
        resp = client.get("/ping", headers={"Origin": "https://anywhere.example"})
        assert resp.headers["access-control-allow-origin"] == "*"
        # CORS spec forbids credentials with a wildcard origin
        assert "access-control-allow-credentials" not in resp.headers
