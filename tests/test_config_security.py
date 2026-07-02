"""Tests for runtime security validation (item C1).

Covers:
- is_production() — which APP_ENV values are treated as production
- security_warnings() — detection of default / weak secrets
- _enforce_runtime_security() — refuse-to-boot in prod, warn-only in dev
- connector token encryption key selection (jwt_secret fallback vs. dedicated key)
"""

import pytest

# Import main at collection time: importing it runs logging.basicConfig(force=True),
# which wipes root log handlers. Doing it here (before any caplog fixture is set up)
# keeps per-test caplog capture intact.
from smritikosh.api import main
from smritikosh.config import Settings, is_production, security_warnings

_STRONG = "a" * 40  # passes the 32-char minimum


def _settings(**overrides) -> Settings:
    """Build a Settings instance with explicit overrides (ignores ambient .env)."""
    base = dict(jwt_secret=_STRONG, app_env="production")
    base.update(overrides)
    return Settings(**base)


class TestIsProduction:
    @pytest.mark.parametrize("env", ["development", "dev", "local", "test", "testing", "ci"])
    def test_non_production_envs(self, env):
        assert is_production(_settings(app_env=env)) is False

    @pytest.mark.parametrize("env", ["production", "staging", "prod", "PRODUCTION", "anything"])
    def test_production_envs(self, env):
        # Fail closed: anything not explicitly non-prod is treated as production.
        assert is_production(_settings(app_env=env)) is True

    def test_case_and_whitespace_insensitive(self):
        assert is_production(_settings(app_env="  Development  ")) is False


class TestSecurityWarnings:
    def test_default_jwt_secret_is_flagged(self):
        problems = security_warnings(_settings(jwt_secret="change-me-in-production"))
        assert len(problems) == 1
        assert "JWT_SECRET" in problems[0]
        assert "default" in problems[0]

    def test_short_jwt_secret_is_flagged(self):
        problems = security_warnings(_settings(jwt_secret="tooshort"))
        assert len(problems) == 1
        assert "too short" in problems[0]

    def test_strong_jwt_secret_passes(self):
        assert security_warnings(_settings(jwt_secret=_STRONG)) == []

    def test_short_connector_key_is_flagged(self):
        problems = security_warnings(_settings(connector_encryption_key="short"))
        assert any("CONNECTOR_ENCRYPTION_KEY" in p for p in problems)

    def test_unset_connector_key_is_ok(self):
        assert security_warnings(_settings(connector_encryption_key=None)) == []

    def test_strong_connector_key_passes(self):
        assert security_warnings(_settings(connector_encryption_key=_STRONG)) == []

    def test_multiple_problems_accumulate(self):
        problems = security_warnings(
            _settings(jwt_secret="change-me-in-production", connector_encryption_key="short")
        )
        assert len(problems) == 2


class TestEnforceRuntimeSecurity:
    def test_production_with_bad_secret_refuses_boot(self, monkeypatch):
        monkeypatch.setattr(
            main, "settings", _settings(app_env="production", jwt_secret="change-me-in-production")
        )
        with pytest.raises(RuntimeError, match="Refusing to start"):
            main._enforce_runtime_security()

    def test_development_with_bad_secret_only_warns(self, monkeypatch, caplog):
        monkeypatch.setattr(
            main, "settings", _settings(app_env="development", jwt_secret="change-me-in-production")
        )
        # Should NOT raise — development is allowed to run insecurely.
        main._enforce_runtime_security()
        assert "Insecure configuration" in caplog.text

    def test_production_with_strong_secret_boots(self, monkeypatch):
        monkeypatch.setattr(main, "settings", _settings(app_env="production", jwt_secret=_STRONG))
        main._enforce_runtime_security()  # no raise


class TestConnectorEncryptionKey:
    def test_falls_back_to_jwt_secret_when_unset(self, monkeypatch):
        from smritikosh.connectors import oauth

        monkeypatch.setattr(oauth.settings, "connector_encryption_keys", "")
        monkeypatch.setattr(oauth.settings, "connector_encryption_key", None)
        monkeypatch.setattr(oauth.settings, "jwt_secret", "jwt-derived-secret-value")

        assert oauth.settings.connector_key_list == ["jwt-derived-secret-value"]

    def test_dedicated_key_overrides_jwt_secret(self, monkeypatch):
        from smritikosh.connectors import oauth

        monkeypatch.setattr(oauth.settings, "connector_encryption_keys", "")
        monkeypatch.setattr(oauth.settings, "jwt_secret", "jwt-derived-secret-value")
        monkeypatch.setattr(oauth.settings, "connector_encryption_key", "a-dedicated-connector-key")

        assert oauth.settings.connector_key_list == ["a-dedicated-connector-key"]
        # Distinct secrets derive distinct Fernet keys.
        assert oauth._derive_fernet_key("a-dedicated-connector-key") != oauth._derive_fernet_key(
            "jwt-derived-secret-value"
        )

    def test_keys_list_takes_priority_and_preserves_order(self, monkeypatch):
        from smritikosh.connectors import oauth

        monkeypatch.setattr(oauth.settings, "connector_encryption_keys", " new-key , old-key ")
        monkeypatch.setattr(oauth.settings, "connector_encryption_key", "ignored-key")

        assert oauth.settings.connector_key_list == ["new-key", "old-key"]

    def test_encrypt_decrypt_round_trip_with_dedicated_key(self, monkeypatch):
        from smritikosh.connectors import oauth

        monkeypatch.setattr(oauth.settings, "connector_encryption_key", "a-dedicated-connector-key-32x")
        tokens = {"access_token": "ya29.abc", "refresh_token": "1//xyz", "expires_in": 3600}
        encrypted = oauth.encrypt_tokens(tokens)
        assert oauth.decrypt_tokens(encrypted) == tokens
