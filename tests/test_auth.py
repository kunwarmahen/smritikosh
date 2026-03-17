"""
Tests for the auth layer:
    - hash_password / verify_password
    - create_access_token / verify_token
    - get_current_user / require_admin FastAPI deps
    - POST /auth/token, POST /auth/register, GET /auth/me routes
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import jwt

from smritikosh.auth.utils import (
    hash_password,
    verify_password,
    create_access_token,
    verify_token,
)
from smritikosh.db.models import AppUser, UserRole


# ── Password utilities ────────────────────────────────────────────────────────


class TestPasswordUtils:
    def test_hash_is_not_plain_text(self):
        h = hash_password("secret123")
        assert h != "secret123"

    def test_verify_correct_password(self):
        h = hash_password("correct-horse")
        assert verify_password("correct-horse", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("correct-horse")
        assert verify_password("wrong-battery", h) is False

    def test_hashes_are_unique(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2   # bcrypt salts each hash


# ── JWT utilities ─────────────────────────────────────────────────────────────


class TestJWTUtils:
    def test_create_and_verify_roundtrip(self):
        token = create_access_token("alice", "user", "default")
        payload = verify_token(token)
        assert payload["sub"] == "alice"
        assert payload["role"] == "user"
        assert payload["app_id"] == "default"

    def test_admin_role_preserved(self):
        token = create_access_token("admin", "admin", "default")
        payload = verify_token(token)
        assert payload["role"] == "admin"

    def test_custom_app_id_preserved(self):
        token = create_access_token("bob", "user", "my-app")
        payload = verify_token(token)
        assert payload["app_id"] == "my-app"

    def test_expired_token_raises(self):
        token = create_access_token("alice", "user", "default", expire_days=-1)
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_token(token)

    def test_tampered_token_raises(self):
        token = create_access_token("alice", "user", "default")
        bad_token = token[:-4] + "XXXX"
        with pytest.raises(jwt.InvalidTokenError):
            verify_token(bad_token)

    def test_garbage_token_raises(self):
        with pytest.raises(jwt.InvalidTokenError):
            verify_token("not.a.token")


# ── FastAPI auth deps ─────────────────────────────────────────────────────────


class TestAuthDeps:
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        from smritikosh.auth.deps import get_current_user
        from fastapi.security import HTTPAuthorizationCredentials

        token = create_access_token("alice", "user", "default")
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        payload = await get_current_user(creds)
        assert payload["sub"] == "alice"

    @pytest.mark.asyncio
    async def test_get_current_user_expired_raises_401(self):
        from smritikosh.auth.deps import get_current_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        token = create_access_token("alice", "user", "default", expire_days=-1)
        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_bad_token_raises_401(self):
        from smritikosh.auth.deps import get_current_user
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials

        creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="garbage")
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_require_admin_passes_for_admin(self):
        from smritikosh.auth.deps import require_admin

        payload = {"sub": "admin", "role": "admin", "app_id": "default"}
        result = await require_admin(payload)
        assert result == payload

    @pytest.mark.asyncio
    async def test_require_admin_raises_403_for_user(self):
        from smritikosh.auth.deps import require_admin
        from fastapi import HTTPException

        payload = {"sub": "alice", "role": "user", "app_id": "default"}
        with pytest.raises(HTTPException) as exc_info:
            await require_admin(payload)
        assert exc_info.value.status_code == 403


# ── Auth routes ───────────────────────────────────────────────────────────────


def _make_app_user(
    username="alice",
    role=UserRole.USER,
    app_id="default",
    is_active=True,
) -> AppUser:
    user = AppUser(
        username=username,
        password_hash=hash_password("password123"),
        role=role,
        app_id=app_id,
        is_active=is_active,
    )
    user.created_at = datetime.now(timezone.utc)
    return user


class TestLoginRoute:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from smritikosh.api.routes.auth import router
        from smritikosh.db.postgres import get_session

        app = FastAPI()
        app.include_router(router)

        mock_session = AsyncMock()
        app.dependency_overrides[get_session] = lambda: mock_session
        return TestClient(app), mock_session

    def test_valid_credentials_return_token(self, client):
        tc, mock_session = client
        user = _make_app_user()

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.post("/auth/token", json={"username": "alice", "password": "password123"})
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["user_id"] == "alice"
        assert data["role"] == "user"
        assert data["token_type"] == "bearer"

    def test_wrong_password_returns_401(self, client):
        tc, mock_session = client
        user = _make_app_user()

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.post("/auth/token", json={"username": "alice", "password": "wrong"})
        assert resp.status_code == 401

    def test_unknown_user_returns_401(self, client):
        tc, mock_session = client

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.post("/auth/token", json={"username": "nobody", "password": "any"})
        assert resp.status_code == 401

    def test_inactive_user_returns_403(self, client):
        tc, mock_session = client
        user = _make_app_user(is_active=False)

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.post("/auth/token", json={"username": "alice", "password": "password123"})
        assert resp.status_code == 403

    def test_returned_token_is_verifiable(self, client):
        tc, mock_session = client
        user = _make_app_user()

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.post("/auth/token", json={"username": "alice", "password": "password123"})
        token = resp.json()["access_token"]
        payload = verify_token(token)
        assert payload["sub"] == "alice"


class TestRegisterRoute:
    @pytest.fixture
    def client_with_admin(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from smritikosh.api.routes.auth import router
        from smritikosh.auth.deps import require_admin
        from smritikosh.db.postgres import get_session

        app = FastAPI()
        app.include_router(router)

        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()

        admin_payload = {"sub": "admin", "role": "admin", "app_id": "default"}
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[require_admin] = lambda: admin_payload
        return TestClient(app), mock_session

    def test_register_creates_user(self, client_with_admin):
        tc, mock_session = client_with_admin

        # First execute: check duplicate (returns None) — second: not needed (flush handles it)
        no_dupe = MagicMock()
        no_dupe.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=no_dupe)

        resp = tc.post(
            "/auth/register",
            json={"username": "bob", "password": "securepass", "role": "user"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "bob"
        assert data["role"] == "user"

    def test_duplicate_username_returns_409(self, client_with_admin):
        tc, mock_session = client_with_admin

        existing_user = _make_app_user(username="bob")
        dupe = MagicMock()
        dupe.scalar_one_or_none.return_value = existing_user
        mock_session.execute = AsyncMock(return_value=dupe)

        resp = tc.post(
            "/auth/register",
            json={"username": "bob", "password": "securepass", "role": "user"},
        )
        assert resp.status_code == 409

    def test_invalid_role_returns_422(self, client_with_admin):
        tc, _ = client_with_admin
        resp = tc.post(
            "/auth/register",
            json={"username": "bob", "password": "securepass", "role": "superuser"},
        )
        assert resp.status_code == 422

    def test_short_password_returns_422(self, client_with_admin):
        tc, _ = client_with_admin
        resp = tc.post(
            "/auth/register",
            json={"username": "bob", "password": "short", "role": "user"},
        )
        assert resp.status_code == 422


class TestGetMeRoute:
    @pytest.fixture
    def client_as_alice(self):
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from smritikosh.api.routes.auth import router
        from smritikosh.auth.deps import get_current_user
        from smritikosh.db.postgres import get_session

        app = FastAPI()
        app.include_router(router)

        mock_session = AsyncMock()
        alice_payload = {"sub": "alice", "role": "user", "app_id": "default"}
        app.dependency_overrides[get_session] = lambda: mock_session
        app.dependency_overrides[get_current_user] = lambda: alice_payload
        return TestClient(app), mock_session

    def test_returns_current_user_profile(self, client_as_alice):
        tc, mock_session = client_as_alice
        user = _make_app_user()

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.get("/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "alice"
        assert data["role"] == "user"

    def test_missing_user_returns_404(self, client_as_alice):
        tc, mock_session = client_as_alice

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=result_mock)

        resp = tc.get("/auth/me")
        assert resp.status_code == 404
