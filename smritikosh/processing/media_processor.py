"""
Media ingestion processor — transcription, parsing, filtering, and LLM-based relevance scoring.

Handles voice notes and documents, extracting structured facts with two-tier confidence strategy:
- High confidence (> 0.75): auto-saved to memory
- Medium confidence (0.60–0.75): queued for user confirmation
- Low confidence (< 0.60): discarded
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.hippocampus import Hippocampus
from smritikosh.memory.semantic import SemanticMemory

logger = logging.getLogger(__name__)

_FIRST_PERSON_RE = re.compile(
    r"\b(I|me|my|mine|myself|we|our|us|I'm|I've|I'd|I'll)\b", re.IGNORECASE
)

_VOICE_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm", ".ogg"}
_DOC_EXT = {".txt", ".md", ".csv", ".pdf"}

_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB Whisper API limit
_MAX_DOC_BYTES = 10 * 1024 * 1024  # 10 MB

_RELEVANCE_THRESHOLD_AUTO = 0.75
_RELEVANCE_THRESHOLD_REVIEW = 0.60

_FACT_EXTRACTION_SCHEMA = (
    "Extract a JSON object with key 'facts' containing a list of durable facts about "
    "the user. Each fact must have 'content' (full statement), 'category' (from predefined list), "
    "'key' (short label), and 'value' (the extracted value). Example: "
    '{"facts": [{"content": "prefers oat milk", "category": "preference", "key": "milk_type", "value": "oat milk"}]}'
)

_FACT_EXTRACTION_EXAMPLE = {
    "facts": [
        {
            "content": "User prefers oat milk in their coffee",
            "category": "preference",
            "key": "milk_preference",
            "value": "oat milk",
        }
    ]
}


@dataclass
class MediaProcessResult:
    """Result of media ingestion processing."""

    media_id: str
    user_id: str
    app_id: str
    content_type: str
    status: str  # processing | complete | nothing_found | failed
    facts_extracted: int = 0
    facts_pending_review: int = 0
    pending_facts: list[dict] = field(default_factory=list)
    event_id: str | None = None
    error_message: str | None = None


class MediaProcessor:
    """Processes uploaded media files (voice notes, documents) for memory extraction."""

    def __init__(
        self,
        llm: LLMAdapter,
        hippocampus: Hippocampus,
        semantic: SemanticMemory,
    ):
        self.llm = llm
        self.hippocampus = hippocampus
        self.semantic = semantic

    async def process(
        self,
        pg: AsyncSession,
        neo,
        *,
        media_id: str,
        user_id: str,
        app_id: str,
        content_type: str,  # voice_note | document
        file_bytes: bytes,
        filename: str,
        context_note: str = "",
    ) -> MediaProcessResult:
        """
        Process uploaded media file and extract memories.

        Returns a result object with status and facts (auto-saved + pending review).
        Never raises; all exceptions are caught and returned in error_message.
        """
        try:
            # 1. Validate extension and file size
            ext = Path(filename).suffix.lower()
            if content_type == "voice_note":
                if ext not in _VOICE_EXT:
                    return MediaProcessResult(
                        media_id=media_id,
                        user_id=user_id,
                        app_id=app_id,
                        content_type=content_type,
                        status="failed",
                        error_message=f"Unsupported audio format: {ext}. Supported: {', '.join(sorted(_VOICE_EXT))}",
                    )
                if len(file_bytes) > _MAX_AUDIO_BYTES:
                    return MediaProcessResult(
                        media_id=media_id,
                        user_id=user_id,
                        app_id=app_id,
                        content_type=content_type,
                        status="failed",
                        error_message=f"Audio file too large: {len(file_bytes) / 1024 / 1024:.1f} MB (max 25 MB)",
                    )
            elif content_type == "document":
                if ext not in _DOC_EXT:
                    return MediaProcessResult(
                        media_id=media_id,
                        user_id=user_id,
                        app_id=app_id,
                        content_type=content_type,
                        status="failed",
                        error_message=f"Unsupported document format: {ext}. Supported: {', '.join(sorted(_DOC_EXT))}",
                    )
                if len(file_bytes) > _MAX_DOC_BYTES:
                    return MediaProcessResult(
                        media_id=media_id,
                        user_id=user_id,
                        app_id=app_id,
                        content_type=content_type,
                        status="failed",
                        error_message=f"Document too large: {len(file_bytes) / 1024 / 1024:.1f} MB (max 10 MB)",
                    )

            # 2. Transcribe / extract text
            if content_type == "voice_note":
                raw_text = await self._transcribe_audio(file_bytes, filename)
                source_type = "media_voice"
            else:  # document
                raw_text = await self._extract_document_text(file_bytes, filename)
                source_type = "media_document"
                # Apply first-person filter for documents
                raw_text = self._first_person_filter(raw_text)

            # 3. Bail early if nothing survives
            if not raw_text or not raw_text.strip():
                return MediaProcessResult(
                    media_id=media_id,
                    user_id=user_id,
                    app_id=app_id,
                    content_type=content_type,
                    status="nothing_found",
                )

            # 4. Fetch existing facts for delta context
            existing_facts = await self.semantic.get_user_profile(pg, neo, user_id, app_id)
            existing_facts_text = "\n".join(
                [f"- {f.value}" for f in existing_facts.facts[:30]]
            )

            # 5. LLM extract_structured → candidate facts
            delta_prompt = (
                f"Extract durable facts about the user from this {content_type.replace('_', ' ')}.\n\n"
                f"You already know: {existing_facts_text or '(nothing yet)'}\n\n"
                f"Extract ONLY facts that are NEW or that CONTRADICT existing knowledge.\n\n"
                f"Content:\n{raw_text[:3000]}"
            )
            if context_note:
                delta_prompt += f"\n\nContext hint from user: {context_note}"

            extracted_dict = await self.llm.extract_structured(
                delta_prompt, _FACT_EXTRACTION_SCHEMA, _FACT_EXTRACTION_EXAMPLE
            )
            candidates = extracted_dict.get("facts", [])

            if not candidates:
                return MediaProcessResult(
                    media_id=media_id,
                    user_id=user_id,
                    app_id=app_id,
                    content_type=content_type,
                    status="nothing_found",
                )

            # 6. LLM relevance scoring per candidate
            scored_facts = await self._score_relevance(
                candidates, context_note, existing_facts_text
            )

            # 7. Route by relevance score
            auto_save, pending_review = self._route_facts(scored_facts)

            # 8. hippocampus.encode() for auto-save facts (high confidence)
            event_id = None
            if auto_save:
                raw_text_summary = raw_text[:2000]  # truncate for episodic storage
                source_meta = {
                    "filename": filename,
                    "content_type": content_type,
                    "context_note": context_note,
                    "fact_count": len(auto_save),
                }
                encoded = await self.hippocampus.encode(
                    pg,
                    neo,
                    user_id=user_id,
                    app_id=app_id,
                    raw_text=raw_text_summary,
                    source_type=source_type,
                    source_meta=source_meta,
                )
                event_id = str(encoded.event.id) if encoded.event else None

            # 9. Return result
            return MediaProcessResult(
                media_id=media_id,
                user_id=user_id,
                app_id=app_id,
                content_type=content_type,
                status="complete",
                facts_extracted=len(auto_save),
                facts_pending_review=len(pending_review),
                pending_facts=pending_review,
                event_id=event_id,
            )

        except Exception as e:
            logger.exception("Media processing failed: %s", e)
            return MediaProcessResult(
                media_id=media_id,
                user_id=user_id,
                app_id=app_id,
                content_type=content_type,
                status="failed",
                error_message=f"Processing error: {str(e)[:200]}",
            )

    async def _transcribe_audio(self, file_bytes: bytes, filename: str) -> str:
        """Transcribe audio file using Whisper."""
        logger.info("Transcribing audio: %s", filename)
        text = await self.llm.transcribe(file_bytes, filename)
        logger.info("Transcription complete: %d chars", len(text))
        return text

    async def _extract_document_text(self, file_bytes: bytes, filename: str) -> str:
        """Extract text from document (PDF, TXT, etc.)."""
        ext = Path(filename).suffix.lower()

        if ext == ".pdf":
            return self._extract_pdf_text(file_bytes)
        elif ext in {".txt", ".md", ".csv"}:
            return file_bytes.decode("utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported document type: {ext}")

    @staticmethod
    def _extract_pdf_text(file_bytes: bytes) -> str:
        """Extract text from PDF using pypdf."""
        import io

        from pypdf import PdfReader

        pdf_file = io.BytesIO(file_bytes)
        reader = PdfReader(pdf_file)

        text_parts = []
        for page_num, page in enumerate(reader.pages, 1):
            if page_num > 50:  # cap at 50 pages
                break
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n".join(text_parts)

    def _first_person_filter(self, text: str) -> str:
        """
        Filter text to keep only sentences containing first-person pronouns.
        Used for documents to focus on user-authored content.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        filtered = [s.strip() for s in sentences if _FIRST_PERSON_RE.search(s)]
        return " ".join(filtered)

    async def _score_relevance(
        self, facts: list[dict], context_note: str, existing_facts_text: str
    ) -> list[dict]:
        """
        Score each fact for user relevance (0-1 scale).
        Adds 'relevance_score' to each fact dict.
        """
        if not facts:
            return facts

        fact_lines = "\n".join(
            [
                f"- {f.get('content', f.get('value', 'unknown'))}"
                for f in facts
            ]
        )
        relevance_prompt = (
            f"Score each fact below for user relevance (0-1 scale): "
            f"does this describe something durable and important about the person who provided it?\n\n"
            f"Facts:\n{fact_lines}\n\n"
            f"Respond with a JSON object: {{'scores': [0.8, 0.4, ...]}}"
        )
        if context_note:
            relevance_prompt += f"\n\nContext: {context_note}"

        try:
            response_dict = await self.llm.extract_structured(
                relevance_prompt,
                "List of relevance scores (0-1) matching the facts count",
                {"scores": [0.8, 0.6]},
            )
            scores = response_dict.get("scores", [])
            # Ensure scores align with facts
            for i, fact in enumerate(facts):
                fact["relevance_score"] = scores[i] if i < len(scores) else 0.5
        except Exception as e:
            logger.warning("Relevance scoring failed: %s; using default 0.7", e)
            for fact in facts:
                fact["relevance_score"] = 0.7

        return facts

    @staticmethod
    def _route_facts(
        scored_facts: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """
        Route facts by relevance score:
        - > 0.75 → auto-save (high confidence)
        - 0.60–0.75 → pending_review (user confirmation)
        - < 0.60 → discard
        """
        auto_save = []
        pending_review = []

        for fact in scored_facts:
            relevance = fact.get("relevance_score", 0.5)
            if relevance >= _RELEVANCE_THRESHOLD_AUTO:
                auto_save.append(fact)
            elif relevance >= _RELEVANCE_THRESHOLD_REVIEW:
                pending_review.append(fact)
            # else: discard (< 0.60)

        return auto_save, pending_review
