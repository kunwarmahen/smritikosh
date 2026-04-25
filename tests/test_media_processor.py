"""
Unit tests for MediaProcessor — transcription, parsing, filtering, relevance scoring.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from smritikosh.processing.media_processor import (
    MediaProcessor,
    MediaProcessResult,
    _FIRST_PERSON_RE,
)
from smritikosh.llm.adapter import LLMAdapter
from smritikosh.memory.hippocampus import Hippocampus
from smritikosh.memory.semantic import SemanticMemory


@pytest.fixture
def mock_llm():
    llm = AsyncMock(spec=LLMAdapter)
    llm.transcribe = AsyncMock(return_value="I prefer oat milk in my coffee.")
    llm.extract_structured = AsyncMock(
        return_value={
            "facts": [
                {
                    "content": "User prefers oat milk",
                    "category": "preference",
                    "key": "milk",
                    "value": "oat milk",
                }
            ]
        }
    )
    return llm


@pytest.fixture
def mock_hippocampus():
    hippocampus = AsyncMock(spec=Hippocampus)
    _encoded = MagicMock(event=MagicMock(id="event-123"), facts=[])
    hippocampus.encode = AsyncMock(return_value=_encoded)
    hippocampus.encode_preextracted = AsyncMock(return_value=_encoded)
    return hippocampus


@pytest.fixture
def mock_semantic():
    semantic = AsyncMock(spec=SemanticMemory)
    semantic.get_user_profile = AsyncMock(
        return_value=MagicMock(facts=[])
    )
    return semantic


@pytest.fixture
def processor(mock_llm, mock_hippocampus, mock_semantic):
    return MediaProcessor(llm=mock_llm, hippocampus=mock_hippocampus, semantic=mock_semantic)


class TestFirstPersonFilter:
    def test_keeps_i_sentences(self):
        text = "I went to the store. The store was closed. I bought milk."
        result = MediaProcessor._first_person_filter(None, text)
        assert "went to the store" in result
        assert "milk" in result
        assert "store was closed" not in result

    def test_filters_third_person(self):
        text = "He went to the store. She bought milk."
        result = MediaProcessor._first_person_filter(None, text)
        assert result.strip() == ""

    def test_empty_input(self):
        result = MediaProcessor._first_person_filter(None, "")
        assert result == ""

    def test_keeps_we_our(self):
        text = "We decided to go. Our team loves this. They disagreed."
        result = MediaProcessor._first_person_filter(None, text)
        assert "decided" in result
        assert "loves" in result
        assert "disagreed" not in result


class TestRouteFactsByRelevance:
    def test_auto_save_above_threshold(self):
        facts = [
            {"content": "fact 1", "relevance_score": 0.80},
            {"content": "fact 2", "relevance_score": 0.76},
        ]
        auto_save, pending = MediaProcessor._route_facts(facts)
        assert len(auto_save) == 2
        assert len(pending) == 0

    def test_review_band_0_60_to_0_75(self):
        facts = [
            {"content": "fact 1", "relevance_score": 0.72},
            {"content": "fact 2", "relevance_score": 0.65},
        ]
        auto_save, pending = MediaProcessor._route_facts(facts)
        assert len(auto_save) == 0
        assert len(pending) == 2

    def test_discard_below_threshold(self):
        facts = [
            {"content": "fact 1", "relevance_score": 0.59},
            {"content": "fact 2", "relevance_score": 0.40},
        ]
        auto_save, pending = MediaProcessor._route_facts(facts)
        assert len(auto_save) == 0
        assert len(pending) == 0

    def test_mixed_routing(self):
        facts = [
            {"content": "fact 1", "relevance_score": 0.85},  # auto
            {"content": "fact 2", "relevance_score": 0.70},  # pending
            {"content": "fact 3", "relevance_score": 0.50},  # discard
        ]
        auto_save, pending = MediaProcessor._route_facts(facts)
        assert len(auto_save) == 1
        assert len(pending) == 1


class TestExtractDocumentText:
    @pytest.mark.asyncio
    async def test_extract_plain_text_txt(self, processor):
        content = b"Hello world\nThis is a test"
        result = await processor._extract_document_text(content, "test.txt")
        assert "Hello world" in result

    @pytest.mark.asyncio
    async def test_extract_utf8_with_errors(self, processor):
        content = b"Valid text \xff\xfe invalid bytes"
        result = await processor._extract_document_text(content, "test.txt")
        assert "Valid text" in result


@pytest.mark.asyncio
async def test_process_voice_note_valid(processor, mock_llm):
    """Test processing a valid voice note."""
    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=b"audio data",
        filename="note.mp3",
        context_note="",
    )

    assert result.status == "complete"
    assert result.facts_extracted >= 0
    mock_llm.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_process_unsupported_extension(processor):
    """Test that unsupported file extensions are rejected."""
    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=b"data",
        filename="file.xyz",
        context_note="",
    )

    assert result.status == "failed"
    assert "Unsupported" in result.error_message


@pytest.mark.asyncio
async def test_process_file_too_large(processor):
    """Test that oversized files are rejected."""
    large_file = b"x" * (26 * 1024 * 1024)  # 26 MB, exceeds 25 MB limit

    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=large_file,
        filename="large.mp3",
        context_note="",
    )

    assert result.status == "failed"
    assert "too large" in result.error_message.lower()


@pytest.mark.asyncio
async def test_process_nothing_extracted(processor, mock_llm):
    """Test handling when no facts are extracted."""
    mock_llm.extract_structured.return_value = {"facts": []}

    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=b"audio",
        filename="note.mp3",
        context_note="",
    )

    assert result.status == "nothing_found"
    assert result.facts_extracted == 0


@pytest.mark.asyncio
async def test_process_never_raises(processor, mock_llm):
    """Test that processor catches all exceptions and returns error status."""
    mock_llm.transcribe.side_effect = Exception("Transcription failed")

    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=b"audio",
        filename="note.mp3",
        context_note="",
    )

    assert result.status == "failed"
    assert "error" in result.error_message.lower()
    # No exception should be raised


@pytest.mark.asyncio
async def test_relevance_scoring_integrated(processor, mock_llm):
    """Test that LLM is called for relevance scoring."""
    mock_llm.extract_structured.side_effect = [
        {  # First call: fact extraction
            "facts": [
                {
                    "content": "User prefers oat milk",
                    "category": "preference",
                    "key": "milk",
                    "value": "oat milk",
                }
            ]
        },
        {  # Second call: relevance scoring
            "scores": [0.85]
        },
    ]

    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="voice_note",
        file_bytes=b"audio",
        filename="note.mp3",
        context_note="",
    )

    assert result.status == "complete"
    assert mock_llm.extract_structured.call_count == 2


@pytest.mark.asyncio
async def test_document_first_person_filter_applied(processor, mock_llm):
    """Test that documents are filtered to first-person content and extraction runs."""
    # file_bytes must contain first-person sentences so they survive the filter
    first_person_content = (
        b"I went to the store. He bought milk. We discussed pricing. I prefer oat milk."
    )

    result = await processor.process(
        pg=MagicMock(),
        neo=MagicMock(),
        media_id="media-1",
        user_id="user-1",
        app_id="default",
        content_type="document",
        file_bytes=first_person_content,
        filename="note.txt",
        context_note="",
    )

    # After first-person filter, extraction LLM should have been called
    assert mock_llm.extract_structured.called


@pytest.mark.asyncio
async def test_auto_save_facts_written_via_encode_preextracted(processor, mock_llm, mock_hippocampus):
    """encode_preextracted is called with the auto-save facts — no double LLM extraction."""
    mock_llm.extract_structured.side_effect = [
        {
            "facts": [
                {"content": "prefers oat milk", "category": "preference", "key": "milk", "value": "oat milk"},
                {"content": "lives in London", "category": "location", "key": "city", "value": "London"},
            ]
        },
        {"scores": [0.90, 0.65]},  # first > 0.75 auto, second 0.60–0.75 pending
    ]

    result = await processor.process(
        pg=MagicMock(), neo=MagicMock(),
        media_id="m1", user_id="u1", app_id="default",
        content_type="voice_note", file_bytes=b"audio", filename="note.mp3", context_note="",
    )

    assert result.status == "complete"
    assert result.facts_extracted == 1    # one auto-saved (>0.75)
    assert result.facts_pending_review == 1  # one for review (0.60–0.75)

    # encode_preextracted called with the auto-save fact only
    mock_hippocampus.encode_preextracted.assert_called_once()
    call_kwargs = mock_hippocampus.encode_preextracted.call_args.kwargs
    assert len(call_kwargs["extracted_facts"]) == 1
    assert call_kwargs["extracted_facts"][0]["value"] == "oat milk"

    # encode() should NOT be called (no double extraction)
    mock_hippocampus.encode.assert_not_called()


class TestMediaProcessResult:
    def test_dataclass_creation(self):
        result = MediaProcessResult(
            media_id="m1",
            user_id="u1",
            app_id="default",
            content_type="voice_note",
            status="complete",
            facts_extracted=2,
            facts_pending_review=1,
            pending_facts=[],
            event_id="e1",
        )

        assert result.media_id == "m1"
        assert result.facts_extracted == 2
        assert result.status == "complete"
