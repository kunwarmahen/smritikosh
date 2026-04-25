"""
Voice enrollment routes — POST/GET/DELETE /user/{user_id}/voice-enrollment.

Stores a speaker d-vector embedding computed from a 30-second voice sample.
The embedding is used to identify the user's speech in meeting recordings
via cosine-similarity matching during the diarization pipeline.

Endpoints:
    POST   /user/{user_id}/voice-enrollment — enroll/re-enroll
    GET    /user/{user_id}/voice-enrollment — status
    DELETE /user/{user_id}/voice-enrollment — remove enrollment
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.api.deps import get_llm
from smritikosh.api.schemas import VoiceEnrollmentResponse
from smritikosh.auth.deps import assert_self_or_admin, get_current_user
from smritikosh.db.models import UserVoiceProfile
from smritikosh.db.postgres import get_session
from smritikosh.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["voice-enrollment"])

_VOICE_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
_MAX_ENROLLMENT_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/{user_id}/voice-enrollment", response_model=VoiceEnrollmentResponse)
async def enroll_voice(
    user_id: str,
    app_id: str = Form("default"),
    file: UploadFile = File(...),
    pg: AsyncSession = Depends(get_session),
    llm: LLMAdapter = Depends(get_llm),
    current_user: dict = Depends(get_current_user),
) -> VoiceEnrollmentResponse:
    """
    Upload a 30-second voice sample to enroll the user's speaker profile.

    The sample is transcribed and a speaker d-vector embedding is computed
    (requires resemblyzer installed). The embedding is stored in user_voice_profiles.
    Re-posting overwrites the existing enrollment.

    Returns enrollment status and whether a speaker embedding was computed.
    """
    assert_self_or_admin(current_user, user_id)

    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if file.filename and "." in file.filename else ""
    if ext not in _VOICE_EXT:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported audio format: {ext!r}. Supported: {', '.join(sorted(_VOICE_EXT))}",
        )

    audio_bytes = await file.read()
    if len(audio_bytes) > _MAX_ENROLLMENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Enrollment file too large: {len(audio_bytes) / 1024 / 1024:.1f} MB (max 10 MB)",
        )
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty file")

    # Compute speaker embedding (None if resemblyzer not installed)
    embedding = await llm.embed_speaker(audio_bytes, file.filename or "enrollment.wav")
    embedding_dim = len(embedding) if embedding else None

    if embedding is None:
        logger.warning(
            "Voice enrollment for user=%s: resemblyzer not installed — "
            "enrollment recorded but speaker matching disabled",
            user_id,
        )

    # Upsert UserVoiceProfile
    result = await pg.execute(
        select(UserVoiceProfile).where(
            UserVoiceProfile.user_id == user_id,
            UserVoiceProfile.app_id == app_id,
        )
    )
    profile = result.scalar_one_or_none()

    now = datetime.now(timezone.utc)

    if profile:
        profile.embedding = embedding
        profile.embedding_dim = embedding_dim
        profile.updated_at = now
    else:
        profile = UserVoiceProfile(
            id=uuid.uuid4(),
            user_id=user_id,
            app_id=app_id,
            embedding=embedding,
            embedding_dim=embedding_dim,
            enrolled_at=now,
            updated_at=now,
        )
        pg.add(profile)

    await pg.commit()

    enrolled_at_iso = (profile.enrolled_at or now).isoformat()

    return VoiceEnrollmentResponse(
        user_id=user_id,
        app_id=app_id,
        enrolled=True,
        has_embedding=embedding is not None,
        embedding_dim=embedding_dim,
        enrolled_at=enrolled_at_iso,
        message=(
            "Voice enrolled successfully with speaker embedding."
            if embedding
            else "Voice enrollment recorded. Install resemblyzer for speaker matching: pip install resemblyzer"
        ),
    )


@router.get("/{user_id}/voice-enrollment", response_model=VoiceEnrollmentResponse)
async def get_voice_enrollment_status(
    user_id: str,
    app_id: str = "default",
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> VoiceEnrollmentResponse:
    """Return the user's current voice enrollment status."""
    assert_self_or_admin(current_user, user_id)

    result = await pg.execute(
        select(UserVoiceProfile).where(
            UserVoiceProfile.user_id == user_id,
            UserVoiceProfile.app_id == app_id,
        )
    )
    profile = result.scalar_one_or_none()

    if not profile:
        return VoiceEnrollmentResponse(
            user_id=user_id,
            app_id=app_id,
            enrolled=False,
            has_embedding=False,
            embedding_dim=None,
            enrolled_at=None,
            message="Not enrolled. Upload a 30-second voice sample to enable speaker matching.",
        )

    return VoiceEnrollmentResponse(
        user_id=user_id,
        app_id=app_id,
        enrolled=True,
        has_embedding=profile.embedding is not None,
        embedding_dim=profile.embedding_dim,
        enrolled_at=profile.enrolled_at.isoformat(),
        message="Enrolled" + (" with speaker embedding." if profile.embedding else " (no embedding — resemblyzer not installed)."),
    )


@router.delete("/{user_id}/voice-enrollment", status_code=204)
async def delete_voice_enrollment(
    user_id: str,
    app_id: str = "default",
    pg: AsyncSession = Depends(get_session),
    current_user: dict = Depends(get_current_user),
) -> None:
    """Remove the user's voice enrollment profile."""
    assert_self_or_admin(current_user, user_id)

    await pg.execute(
        delete(UserVoiceProfile).where(
            UserVoiceProfile.user_id == user_id,
            UserVoiceProfile.app_id == app_id,
        )
    )
    await pg.commit()
