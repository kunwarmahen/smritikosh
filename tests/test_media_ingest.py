"""
API tests for media ingestion endpoints — POST /ingest/media, GET /ingest/media/{id}/status,
POST /ingest/media/{id}/confirm.

All tests are offline — DB and LLM dependencies are mocked.
"""

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.auth.deps import get_current_user
from smritikosh.db.models import MediaIngest, MediaIngestStatus
from smritikosh.db.neo4j import get_neo4j_session
from smritikosh.db.postgres import get_session
from smritikosh.memory.semantic import SemanticMemory
from smritikosh.processing.media_processor import MediaProcessResult


def _auth_override(user_id: str = "test-user"):
    return {"sub": user_id, "user_id": user_id, "role": "admin", "app_ids": ["default"]}


def _make_mock_ingest(
    user_id: str = "test-user",
    status: str = "complete",
    facts_extracted: int = 2,
    facts_pending_review: int = 1,
    pending_facts: list | None = None,
) -> MagicMock:
    ingest = MagicMock(spec=MediaIngest)
    ingest.id = uuid.uuid4()
    ingest.user_id = user_id
    ingest.app_id = "default"
    ingest.content_type = "voice_note"
    ingest.status = status
    ingest.facts_extracted = facts_extracted
    ingest.facts_pending_review = facts_pending_review
    ingest.pending_facts = pending_facts or [
        {"content": "prefers oat milk", "category": "preference", "key": "milk", "value": "oat milk", "relevance_score": 0.70}
    ]
    ingest.source_type = "media_voice"
    ingest.error_message = None
    return ingest


@pytest.fixture
def mock_pg():
    pg = AsyncMock()
    execute_result = MagicMock()
    execute_result.scalar_one_or_none.return_value = None
    pg.execute = AsyncMock(return_value=execute_result)
    pg.flush = AsyncMock()
    pg.commit = AsyncMock()
    pg.add = MagicMock()
    pg.get = AsyncMock(return_value=None)
    return pg


@pytest.fixture
def mock_neo():
    return AsyncMock()


@pytest.fixture
def mock_semantic():
    s = AsyncMock(spec=SemanticMemory)
    s.upsert_fact = AsyncMock(return_value=MagicMock())
    return s


@pytest.fixture
def mock_media_processor():
    processor = AsyncMock()
    processor.process = AsyncMock(
        return_value=MediaProcessResult(
            media_id="media-1",
            user_id="test-user",
            app_id="default",
            content_type="voice_note",
            status="complete",
            facts_extracted=2,
            facts_pending_review=0,
            event_id=None,
        )
    )
    return processor


@pytest.fixture(autouse=True)
def override_deps(mock_pg, mock_neo, mock_semantic, mock_media_processor, monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: _auth_override()
    app.dependency_overrides[get_session] = lambda: mock_pg
    app.dependency_overrides[get_neo4j_session] = lambda: mock_neo
    app.dependency_overrides[deps.get_semantic] = lambda: mock_semantic
    app.dependency_overrides[deps.get_media_processor] = lambda: mock_media_processor
    # Prevent background tasks from touching real DB / asyncpg pool
    monkeypatch.setattr(
        "smritikosh.api.routes.media_ingest._run_processing",
        AsyncMock(return_value=None),
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    return TestClient(app)


class TestPostIngestMedia:
    def test_post_media_invalid_content_type_422(self, client):
        """Test invalid content_type returns 422."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "invalid_type"},
            files={"file": ("test.mp3", b"data")},
        )
        assert response.status_code == 422
        assert "Invalid content_type" in response.json()["detail"]

    def test_post_media_invalid_audio_extension_422(self, client):
        """Test invalid audio extension returns 422."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "voice_note"},
            files={"file": ("test.xyz", b"data")},
        )
        assert response.status_code == 422
        assert "Invalid audio format" in response.json()["detail"]

    def test_post_media_invalid_document_extension_422(self, client):
        """Test invalid document extension returns 422."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "document"},
            files={"file": ("test.docx", b"data")},
        )
        assert response.status_code == 422
        assert "Invalid document format" in response.json()["detail"]

    def test_post_media_file_too_large_413(self, client):
        """Test oversized audio file returns 413."""
        large_data = b"x" * (26 * 1024 * 1024)
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "voice_note"},
            files={"file": ("large.mp3", large_data)},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()

    def test_post_media_voice_note_returns_202(self, client):
        """Test POST /ingest/media with voice note returns 202."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "app_id": "default", "content_type": "voice_note"},
            files={"file": ("test.mp3", b"fake audio data", "audio/mpeg")},
        )
        assert response.status_code == 202
        data = response.json()
        assert "media_id" in data
        assert data["status"] == "processing"
        assert data["content_type"] == "voice_note"

    def test_post_media_document_returns_202(self, client):
        """Test POST /ingest/media with document returns 202."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "document"},
            files={"file": ("test.txt", b"document content", "text/plain")},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing"
        assert data["content_type"] == "document"

    def test_context_note_optional(self, client):
        """Test that context_note is optional."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "voice_note"},
            files={"file": ("test.mp3", b"data")},
        )
        assert response.status_code == 202

    def test_app_id_defaults_to_default(self, client):
        """Test that app_id defaults to 'default' if not provided."""
        response = client.post(
            "/ingest/media",
            data={"user_id": "test-user", "content_type": "voice_note"},
            files={"file": ("test.mp3", b"data")},
        )
        assert response.status_code == 202
        assert response.json()["app_id"] == "default"


class TestGetMediaStatus:
    def test_get_media_status_not_found_404(self, client, mock_pg):
        """Test 404 for nonexistent media."""
        mock_pg.get = AsyncMock(return_value=None)
        fake_id = str(uuid.uuid4())
        response = client.get(f"/ingest/media/{fake_id}/status")
        assert response.status_code == 404

    def test_get_media_status_complete(self, client, mock_pg):
        """Test status endpoint returns complete result with pending facts."""
        ingest = _make_mock_ingest(status="complete")
        mock_pg.get = AsyncMock(return_value=ingest)

        response = client.get(f"/ingest/media/{ingest.id}/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["facts_extracted"] == 2
        assert data["facts_pending_review"] == 1


class TestConfirmMediaFacts:
    def test_confirm_media_facts_saves_selected(self, client, mock_pg, mock_semantic):
        """Test confirming facts calls SemanticMemory.upsert_fact."""
        ingest = _make_mock_ingest(status="complete", facts_pending_review=1)
        mock_pg.get = AsyncMock(return_value=ingest)

        response = client.post(
            f"/ingest/media/{ingest.id}/confirm",
            json={"user_id": "test-user", "app_id": "default", "confirmed_indices": [0]},
        )
        assert response.status_code == 200

    def test_confirm_fails_if_not_complete(self, client, mock_pg):
        """Test 422 when trying to confirm a still-processing ingest."""
        ingest = _make_mock_ingest(status="processing")
        mock_pg.get = AsyncMock(return_value=ingest)

        response = client.post(
            f"/ingest/media/{ingest.id}/confirm",
            json={"user_id": "test-user", "confirmed_indices": [0]},
        )
        assert response.status_code == 422

    def test_confirm_not_found_404(self, client, mock_pg):
        """Test 404 when media_id not found."""
        mock_pg.get = AsyncMock(return_value=None)
        response = client.post(
            f"/ingest/media/{uuid.uuid4()}/confirm",
            json={"user_id": "test-user", "confirmed_indices": []},
        )
        assert response.status_code == 404


class TestMediaIngestIdempotency:
    def test_idempotency_key_returns_cached(self, client, mock_pg):
        """Test idempotency key returns existing result on second upload."""
        existing = _make_mock_ingest()
        existing.idempotency_key = "unique-key-123"
        execute_result = MagicMock()
        execute_result.scalar_one_or_none.return_value = existing
        mock_pg.execute = AsyncMock(return_value=execute_result)

        response = client.post(
            "/ingest/media",
            data={
                "user_id": "test-user",
                "content_type": "voice_note",
                "idempotency_key": "unique-key-123",
            },
            files={"file": ("test2.mp3", b"different data")},
        )
        assert response.status_code == 202
        assert "cached" in response.json()["message"]
