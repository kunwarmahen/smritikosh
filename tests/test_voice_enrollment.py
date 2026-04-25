"""
Tests for voice enrollment endpoints and meeting recording processing.

All tests are offline — DB and LLM dependencies are mocked.
"""

import io
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from smritikosh.api.main import app
from smritikosh.api import deps
from smritikosh.auth.deps import get_current_user
from smritikosh.db.models import UserVoiceProfile
from smritikosh.db.postgres import get_session
from smritikosh.processing.media_processor import MediaProcessor, MediaProcessResult


def _auth(user_id: str = "alice"):
    return {"sub": user_id, "user_id": user_id, "role": "admin", "app_ids": ["default"]}


def _voice_profile(user_id: str = "alice", embedding: list | None = None) -> MagicMock:
    p = MagicMock(spec=UserVoiceProfile)
    p.id = uuid.uuid4()
    p.user_id = user_id
    p.app_id = "default"
    p.embedding = embedding
    p.embedding_dim = len(embedding) if embedding else None
    p.enrolled_at = datetime.now(timezone.utc)
    p.updated_at = datetime.now(timezone.utc)
    return p


@pytest.fixture
def mock_pg():
    pg = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    pg.execute = AsyncMock(return_value=result)
    pg.flush = AsyncMock()
    pg.commit = AsyncMock()
    pg.add = MagicMock()
    pg.get = AsyncMock(return_value=None)
    return pg


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.embed_speaker = AsyncMock(return_value=[0.1] * 256)
    llm.transcribe = AsyncMock(return_value="I always drink oat milk in the morning")
    llm.diarize = AsyncMock(return_value=[
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ])
    return llm


# ── Voice enrollment endpoint tests ───────────────────────────────────────────


class TestVoiceEnrollmentPost:
    async def test_enroll_creates_profile(self, mock_pg, mock_llm):
        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg
        app.dependency_overrides[deps.get_llm] = lambda: mock_llm

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt "  # minimal WAV header bytes
                response = await client.post(
                    "/user/alice/voice-enrollment",
                    data={"app_id": "default"},
                    files={"file": ("sample.wav", io.BytesIO(audio_bytes), "audio/wav")},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["enrolled"] is True
            assert data["has_embedding"] is True
            assert data["embedding_dim"] == 256
        finally:
            app.dependency_overrides.clear()

    async def test_enroll_no_embedding_when_resemblyzer_unavailable(self, mock_pg):
        llm = AsyncMock()
        llm.embed_speaker = AsyncMock(return_value=None)  # resemblyzer not installed

        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg
        app.dependency_overrides[deps.get_llm] = lambda: llm

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/user/alice/voice-enrollment",
                    data={"app_id": "default"},
                    files={"file": ("sample.wav", io.BytesIO(b"WAV"), "audio/wav")},
                )

            assert response.status_code == 200
            data = response.json()
            assert data["enrolled"] is True
            assert data["has_embedding"] is False
            assert data["embedding_dim"] is None
            assert "resemblyzer" in data["message"]
        finally:
            app.dependency_overrides.clear()

    async def test_enroll_unsupported_format_rejected(self, mock_pg, mock_llm):
        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg
        app.dependency_overrides[deps.get_llm] = lambda: mock_llm

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/user/alice/voice-enrollment",
                    data={"app_id": "default"},
                    files={"file": ("sample.docx", io.BytesIO(b"PK"), "application/octet-stream")},
                )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    async def test_enroll_re_enrolls_updates_existing(self, mock_pg, mock_llm):
        existing = _voice_profile(embedding=[0.0] * 256)
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        mock_pg.execute = AsyncMock(return_value=result)

        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg
        app.dependency_overrides[deps.get_llm] = lambda: mock_llm

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/user/alice/voice-enrollment",
                    data={"app_id": "default"},
                    files={"file": ("sample.wav", io.BytesIO(b"WAV"), "audio/wav")},
                )

            assert response.status_code == 200
            assert existing.embedding == [0.1] * 256  # updated by mock
        finally:
            app.dependency_overrides.clear()

    async def test_enroll_forbidden_for_other_user(self, mock_pg, mock_llm):
        # Bob is a plain user (not admin) trying to enroll Alice's voice
        bob_auth = {"sub": "bob", "user_id": "bob", "role": "user", "app_ids": ["default"]}
        app.dependency_overrides[get_current_user] = lambda: bob_auth
        app.dependency_overrides[get_session] = lambda: mock_pg
        app.dependency_overrides[deps.get_llm] = lambda: mock_llm

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    "/user/alice/voice-enrollment",
                    data={"app_id": "default"},
                    files={"file": ("sample.wav", io.BytesIO(b"WAV"), "audio/wav")},
                )
            assert response.status_code == 403
        finally:
            app.dependency_overrides.clear()


class TestVoiceEnrollmentGet:
    async def test_get_status_not_enrolled(self, mock_pg):
        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/user/alice/voice-enrollment")
            assert response.status_code == 200
            data = response.json()
            assert data["enrolled"] is False
            assert data["has_embedding"] is False
        finally:
            app.dependency_overrides.clear()

    async def test_get_status_enrolled_with_embedding(self, mock_pg):
        profile = _voice_profile(embedding=[0.1] * 256)
        result = MagicMock()
        result.scalar_one_or_none.return_value = profile
        mock_pg.execute = AsyncMock(return_value=result)

        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/user/alice/voice-enrollment")
            assert response.status_code == 200
            data = response.json()
            assert data["enrolled"] is True
            assert data["has_embedding"] is True
        finally:
            app.dependency_overrides.clear()


class TestVoiceEnrollmentDelete:
    async def test_delete_enrollment(self, mock_pg):
        app.dependency_overrides[get_current_user] = lambda: _auth()
        app.dependency_overrides[get_session] = lambda: mock_pg

        try:
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.delete("/user/alice/voice-enrollment")
            assert response.status_code == 204
        finally:
            app.dependency_overrides.clear()


# ── MediaProcessor meeting_recording tests ────────────────────────────────────


class TestMeetingRecordingProcessing:
    def _make_processor(self, llm, hippocampus=None, semantic=None) -> MediaProcessor:
        hip = hippocampus or AsyncMock()
        hip.encode_preextracted = AsyncMock(return_value=MagicMock(event=MagicMock(id=uuid.uuid4())))
        sem = semantic or AsyncMock()
        sem.get_user_profile = AsyncMock(return_value=MagicMock(facts=[]))
        return MediaProcessor(llm=llm, hippocampus=hip, semantic=sem)

    async def test_meeting_recording_no_enrollment_first_person_filter(self, mock_llm):
        mock_llm.transcribe = AsyncMock(
            return_value="I think this is a great idea. She disagrees with me though."
        )
        mock_llm.extract_structured = AsyncMock(
            return_value={"facts": [{"content": "thinks idea is great", "category": "belief", "key": "idea", "value": "great"}]}
        )

        processor = self._make_processor(mock_llm)
        pg = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None  # no voice profile
        pg.execute = AsyncMock(return_value=result_mock)

        result = await processor.process(
            pg, AsyncMock(),
            media_id="m1",
            user_id="alice",
            app_id="default",
            content_type="meeting_recording",
            file_bytes=b"WAV",
            filename="meeting.wav",
        )

        assert result.status in ("complete", "nothing_found")
        # Transcript was filtered to first-person content
        mock_llm.transcribe.assert_awaited_once()

    async def test_meeting_recording_validation_rejects_pdf(self, mock_llm):
        processor = self._make_processor(mock_llm)
        pg = AsyncMock()

        result = await processor.process(
            pg, AsyncMock(),
            media_id="m2",
            user_id="alice",
            app_id="default",
            content_type="meeting_recording",
            file_bytes=b"%PDF",
            filename="meeting.pdf",
        )

        assert result.status == "failed"
        assert "Unsupported audio format" in (result.error_message or "")

    async def test_meeting_recording_with_enrolled_voice_diarizes(self, mock_llm):
        mock_llm.transcribe = AsyncMock(
            return_value="I plan to launch next week. Bob said he'll help me review it."
        )
        mock_llm.diarize = AsyncMock(return_value=[
            {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"},
            {"start": 8.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ])
        mock_llm.extract_structured = AsyncMock(
            return_value={"facts": [{"content": "plans to launch", "category": "goal", "key": "launch_timing", "value": "next week"}]}
        )

        processor = self._make_processor(mock_llm)

        pg = AsyncMock()
        profile = _voice_profile(embedding=[0.1] * 256)
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = profile
        pg.execute = AsyncMock(return_value=result_mock)

        with patch("smritikosh.config.settings") as mock_settings:
            mock_settings.diarization_provider = "pyannote"
            mock_settings.speaker_similarity_threshold = 0.0  # always match
            mock_settings.whisper_provider = "openai"

            result = await processor.process(
                pg, AsyncMock(),
                media_id="m3",
                user_id="alice",
                app_id="default",
                content_type="meeting_recording",
                file_bytes=b"WAV",
                filename="standup.wav",
            )

        # Diarization was invoked
        mock_llm.diarize.assert_awaited_once()
        assert result.status in ("complete", "nothing_found")

    async def test_meeting_recording_enrolled_no_embedding_falls_back(self, mock_llm):
        mock_llm.transcribe = AsyncMock(
            return_value="I prefer async standups. The team uses Slack."
        )
        mock_llm.extract_structured = AsyncMock(
            return_value={"facts": [{"content": "prefers async standups", "category": "preference", "key": "standup_style", "value": "async"}]}
        )

        processor = self._make_processor(mock_llm)

        pg = AsyncMock()
        profile = _voice_profile(embedding=None)  # enrolled but no embedding
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = profile
        pg.execute = AsyncMock(return_value=result_mock)

        result = await processor.process(
            pg, AsyncMock(),
            media_id="m4",
            user_id="alice",
            app_id="default",
            content_type="meeting_recording",
            file_bytes=b"WAV",
            filename="meeting.wav",
        )

        # Falls back to first-person filter — diarize should NOT be called
        mock_llm.diarize.assert_not_awaited()
        assert result.status in ("complete", "nothing_found")

    async def test_meeting_recording_file_too_large_rejected(self, mock_llm):
        processor = self._make_processor(mock_llm)
        big_bytes = b"x" * (501 * 1024 * 1024)  # 501 MB

        result = await processor.process(
            AsyncMock(), AsyncMock(),
            media_id="m5",
            user_id="alice",
            app_id="default",
            content_type="meeting_recording",
            file_bytes=big_bytes,
            filename="huge.wav",
        )

        assert result.status == "failed"
        assert "500 MB" in (result.error_message or "")


# ── LLM adapter embed_speaker tests ───────────────────────────────────────────


class TestEmbedSpeaker:
    async def test_embed_speaker_returns_none_when_resemblyzer_missing(self):
        from smritikosh.llm.adapter import LLMAdapter
        from smritikosh.config import Settings

        adapter = LLMAdapter(Settings())

        with patch.dict("sys.modules", {"resemblyzer": None}):
            result = await adapter.embed_speaker(b"WAV", "sample.wav")

        assert result is None

    async def test_diarize_returns_single_segment_when_provider_none(self):
        from smritikosh.llm.adapter import LLMAdapter
        from smritikosh.config import Settings

        cfg = Settings()
        # diarization_provider defaults to "none"
        adapter = LLMAdapter(cfg)

        segments = await adapter.diarize(b"WAV", "meeting.wav")
        assert len(segments) == 1
        assert segments[0]["speaker"] == "SPEAKER_00"
        assert segments[0]["start"] == 0.0
