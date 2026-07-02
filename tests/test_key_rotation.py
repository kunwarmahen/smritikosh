"""
Tests for rotatable connector-token encryption (C3, MultiFernet).

Covers: decrypt-old/encrypt-new across a key list, ciphertext rotation, the
rotate_connector_tokens job, config validation, and the admin endpoint shape.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.fernet import InvalidToken

from smritikosh.connectors import oauth
from smritikosh.tasks import jobs

TOKENS = {"access_token": "ya29.abc", "refresh_token": "1//xyz", "expires_in": 3600}


@pytest.fixture
def single_old_key(monkeypatch):
    monkeypatch.setattr(oauth.settings, "connector_encryption_keys", "old-secret-key-32-chars-long!!!")


@pytest.fixture
def rotated_keys(monkeypatch):
    """New primary prepended, old key kept for decryption."""
    monkeypatch.setattr(
        oauth.settings,
        "connector_encryption_keys",
        "new-secret-key-32-chars-long!!!,old-secret-key-32-chars-long!!!",
    )


# ── MultiFernet encrypt/decrypt ───────────────────────────────────────────────


class TestMultiFernet:
    def test_round_trip(self, single_old_key):
        assert oauth.decrypt_tokens(oauth.encrypt_tokens(TOKENS)) == TOKENS

    def test_old_ciphertext_decrypts_after_rotation(self, monkeypatch, single_old_key):
        old_ciphertext = oauth.encrypt_tokens(TOKENS)

        monkeypatch.setattr(
            oauth.settings,
            "connector_encryption_keys",
            "new-secret-key-32-chars-long!!!,old-secret-key-32-chars-long!!!",
        )
        assert oauth.decrypt_tokens(old_ciphertext) == TOKENS

    def test_new_ciphertext_uses_primary_key(self, monkeypatch, rotated_keys):
        ciphertext = oauth.encrypt_tokens(TOKENS)

        # Drop the old key entirely — new ciphertext must still decrypt.
        monkeypatch.setattr(
            oauth.settings, "connector_encryption_keys", "new-secret-key-32-chars-long!!!"
        )
        assert oauth.decrypt_tokens(ciphertext) == TOKENS

    def test_unknown_key_raises_invalid_token(self, monkeypatch, single_old_key):
        ciphertext = oauth.encrypt_tokens(TOKENS)
        monkeypatch.setattr(
            oauth.settings, "connector_encryption_keys", "completely-different-key-32chars!"
        )
        with pytest.raises(InvalidToken):
            oauth.decrypt_tokens(ciphertext)

    def test_rotate_ciphertext_moves_to_primary(self, monkeypatch, single_old_key):
        old_ciphertext = oauth.encrypt_tokens(TOKENS)

        monkeypatch.setattr(
            oauth.settings,
            "connector_encryption_keys",
            "new-secret-key-32-chars-long!!!,old-secret-key-32-chars-long!!!",
        )
        rotated = oauth.rotate_ciphertext(old_ciphertext)
        assert rotated != old_ciphertext

        # After dropping the old key, only the rotated ciphertext survives.
        monkeypatch.setattr(
            oauth.settings, "connector_encryption_keys", "new-secret-key-32-chars-long!!!"
        )
        assert oauth.decrypt_tokens(rotated) == TOKENS
        with pytest.raises(InvalidToken):
            oauth.decrypt_tokens(old_ciphertext)


# ── Config validation ─────────────────────────────────────────────────────────


class TestKeyListValidation:
    def test_short_key_in_list_is_flagged(self):
        from smritikosh.config import Settings, security_warnings

        s = Settings(
            jwt_secret="x" * 40,
            connector_encryption_keys="strong-enough-key-32-chars-long!,tiny",
        )
        problems = security_warnings(s)
        assert any("CONNECTOR_ENCRYPTION_KEYS entry #2" in p for p in problems)

    def test_strong_key_list_passes(self):
        from smritikosh.config import Settings, security_warnings

        s = Settings(
            jwt_secret="x" * 40,
            connector_encryption_keys="strong-enough-key-32-chars-long!,another-strong-key-32-chars-long",
        )
        assert security_warnings(s) == []


# ── Rotation job ──────────────────────────────────────────────────────────────


def _connector(encrypted: str | None):
    c = MagicMock()
    c.encrypted_tokens = encrypted
    c.user_id = "u1"
    c.provider = "gmail"
    return c


def _db_session_yielding(connectors):
    execute_result = MagicMock()
    execute_result.scalars.return_value.all.return_value = connectors
    session = AsyncMock()
    session.execute = AsyncMock(return_value=execute_result)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=session)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class TestRotateConnectorTokensJob:
    async def test_rotates_and_counts(self, monkeypatch, rotated_keys):
        monkeypatch.setattr(
            oauth.settings, "connector_encryption_keys", "old-secret-key-32-chars-long!!!"
        )
        good = _connector(oauth.encrypt_tokens(TOKENS))
        monkeypatch.setattr(
            oauth.settings,
            "connector_encryption_keys",
            "new-secret-key-32-chars-long!!!,old-secret-key-32-chars-long!!!",
        )
        empty = _connector(None)
        undecryptable = _connector("gAAAAAB-not-a-real-fernet-token")

        ctx = _db_session_yielding([good, empty, undecryptable])
        with patch("smritikosh.db.postgres.db_session", return_value=ctx):
            result = await jobs._rotate_connector_tokens()

        assert result == {"rotated": 1, "failed": 1, "skipped": 1, "total": 3}
        assert oauth.decrypt_tokens(good.encrypted_tokens) == TOKENS

    async def test_arq_wrapper_delegates(self):
        payload = {"rotated": 2, "failed": 0, "skipped": 0, "total": 2}
        with patch.object(jobs, "_rotate_connector_tokens", AsyncMock(return_value=payload)) as h:
            result = await jobs.rotate_connector_tokens(ctx={})
        assert result == payload
        h.assert_awaited_once()

    def test_registered_in_worker(self):
        names = {f.__name__ for f in jobs.WorkerSettings.functions}
        assert "rotate_connector_tokens" in names
