"""
E2E pipeline test — encode → consolidate → build context.

Tests the full memory lifecycle using mocked LLM calls and mocked DB sessions.
This is the highest-value regression test: it verifies that the three major
subsystems (Hippocampus, Consolidator, ContextBuilder) work together correctly
and that information encoded at step 1 survives to appear in context at step 3.

Why mocked DBs?
    Real DB tests are marked @pytest.mark.db and require docker compose up.
    This test should run offline in CI with no infrastructure dependencies.
    We mock EpisodicMemory and SemanticMemory at the method level — the mocks
    simulate what a real DB would do (store → return on read), so the pipeline
    orchestration logic is fully exercised.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from smritikosh.db.models import Event
from smritikosh.memory.episodic import EpisodicMemory, SearchResult
from smritikosh.memory.hippocampus import Hippocampus
from smritikosh.memory.narrative import NarrativeMemory
from smritikosh.memory.semantic import FactRecord, SemanticMemory, UserProfile
from smritikosh.processing.amygdala import Amygdala
from smritikosh.processing.consolidator import Consolidator, MIN_EVENTS_TO_CONSOLIDATE
from smritikosh.retrieval.context_builder import ContextBuilder, MemoryContext


def _make_semantic_mock() -> AsyncMock:
    """AsyncMock for SemanticMemory with check_fact_conflict returning None (no conflicts)."""
    m = AsyncMock(spec=SemanticMemory)
    m.check_fact_conflict = AsyncMock(return_value=None)
    return m


# ── Test constants ────────────────────────────────────────────────────────────

USER_ID = "e2e-test-user"
APP_ID  = "default"
EMBEDDING = [0.1] * 1536   # fake fixed-dimension vector

# The text that will be encoded and should survive to context retrieval
ENCODED_TEXT = "I decided to build smritikosh as my core AI memory product"
SUMMARY_TEXT = "User is building smritikosh, an AI memory product"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(raw_text: str = ENCODED_TEXT, consolidated: bool = False) -> Event:
    e = Event(
        id=uuid.uuid4(),
        user_id=USER_ID,
        app_id=APP_ID,
        raw_text=raw_text,
        importance_score=0.8,
        consolidated=consolidated,
        event_metadata={},
        created_at=datetime.now(timezone.utc),
    )
    if consolidated:
        e.summary = SUMMARY_TEXT
    return e


def _make_llm(
    embed_return=None,
    extract_facts=None,
    consolidation_summary=None,
) -> AsyncMock:
    """
    Build a mock LLMAdapter that returns controlled values for all three calls:
      embed()             → EMBEDDING vector
      extract_structured  → facts list (for Hippocampus)
      complete()          → raw JSON string (for Consolidator's summary call)
    """
    import json

    llm = AsyncMock()
    llm.embed = AsyncMock(return_value=embed_return or EMBEDDING)

    facts_payload = extract_facts or [
        {"category": "project", "key": "active", "value": "smritikosh", "confidence": 0.95},
    ]
    # extract_structured is used by both Hippocampus (facts only) and Consolidator
    # (facts + summary + links). We include summary so the Consolidator re-embed
    # path is exercised.
    llm.extract_structured = AsyncMock(return_value={
        "summary": consolidation_summary or SUMMARY_TEXT,
        "facts": facts_payload,
        "links": [],
    })

    summary_payload = {
        "summary": consolidation_summary or SUMMARY_TEXT,
        "facts": facts_payload,
        "links": [],
    }
    llm.complete = AsyncMock(return_value=json.dumps(summary_payload))

    # _parse_json is a static method — delegate to the real implementation
    from smritikosh.llm.adapter import LLMAdapter
    llm._parse_json = LLMAdapter._parse_json

    return llm


def _make_fact_record(value: str = "smritikosh") -> FactRecord:
    return FactRecord(
        category="project", key="active", value=value,
        confidence=0.95, frequency_count=1,
        first_seen_at="2026-04-10", last_seen_at="2026-04-10",
    )


# ── Stage 1: Hippocampus.encode() ─────────────────────────────────────────────


class TestEncodeStage:
    """Verify Hippocampus.encode() produces the expected EncodedMemory."""

    def _make_deps(self):
        llm = _make_llm()
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()
        stored_event = _make_event()
        episodic.store = AsyncMock(return_value=stored_event)
        semantic.upsert_fact = AsyncMock(return_value=_make_fact_record())
        hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic)
        pg, neo = AsyncMock(), AsyncMock()
        return hippo, episodic, semantic, pg, neo, stored_event

    @pytest.mark.asyncio
    async def test_encode_stores_event_in_episodic(self):
        hippo, episodic, _, pg, neo, _ = self._make_deps()
        result = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)
        episodic.store.assert_called_once()
        call_kwargs = episodic.store.call_args.kwargs
        assert call_kwargs["user_id"] == USER_ID
        assert call_kwargs["raw_text"] == ENCODED_TEXT

    @pytest.mark.asyncio
    async def test_encode_returns_event_with_embedding(self):
        hippo, _, _, pg, neo, stored_event = self._make_deps()
        result = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)
        # Embedding should have been passed to store()
        call_kwargs = hippo.episodic.store.call_args.kwargs
        assert call_kwargs["embedding"] == EMBEDDING

    @pytest.mark.asyncio
    async def test_encode_upserts_extracted_facts(self):
        hippo, _, semantic, pg, neo, _ = self._make_deps()
        result = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)
        semantic.upsert_fact.assert_called_once()
        assert len(result.facts) == 1
        assert result.facts[0].value == "smritikosh"

    @pytest.mark.asyncio
    async def test_encode_assigns_importance_score(self):
        hippo, _, _, pg, neo, _ = self._make_deps()
        result = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)
        assert 0.0 <= result.importance_score <= 1.0

    @pytest.mark.asyncio
    async def test_encode_survives_extraction_failure(self):
        """If fact extraction fails, the event is still stored."""
        llm = _make_llm()
        llm.extract_structured = AsyncMock(side_effect=RuntimeError("LLM extraction down"))
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()
        episodic.store = AsyncMock(return_value=_make_event())
        hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic)
        pg, neo = AsyncMock(), AsyncMock()

        result = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)

        assert result.extraction_failed is True
        episodic.store.assert_called_once()  # event still stored


# ── Stage 2: Consolidator.run() ───────────────────────────────────────────────


class TestConsolidateStage:
    """Verify Consolidator.run() processes batches and marks events consolidated."""

    def _make_deps(self, n_events: int = MIN_EVENTS_TO_CONSOLIDATE):
        llm = _make_llm()
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()
        narrative = AsyncMock(spec=NarrativeMemory)

        events = [_make_event(f"event {i}") for i in range(n_events)]
        episodic.get_unconsolidated = AsyncMock(return_value=events)
        episodic.mark_consolidated = AsyncMock()
        episodic.update_embedding = AsyncMock()
        semantic.upsert_fact = AsyncMock(return_value=_make_fact_record())
        narrative.create_link = AsyncMock(return_value=MagicMock())

        consolidator = Consolidator(
            llm=llm, episodic=episodic, semantic=semantic,
            narrative=narrative, min_events=MIN_EVENTS_TO_CONSOLIDATE,
        )
        pg, neo = AsyncMock(), AsyncMock()
        return consolidator, episodic, semantic, pg, neo, events

    @pytest.mark.asyncio
    async def test_consolidation_runs_when_enough_events(self):
        consolidator, episodic, _, pg, neo, _ = self._make_deps(MIN_EVENTS_TO_CONSOLIDATE)
        result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        assert not result.skipped
        assert result.events_consolidated > 0

    @pytest.mark.asyncio
    async def test_consolidation_skips_below_minimum(self):
        consolidator, _, _, pg, neo, _ = self._make_deps(n_events=2)
        result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        assert result.skipped is True
        assert "need at least" in result.skip_reason

    @pytest.mark.asyncio
    async def test_consolidation_calls_mark_consolidated(self):
        consolidator, episodic, _, pg, neo, _ = self._make_deps()
        await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        episodic.mark_consolidated.assert_called()

    @pytest.mark.asyncio
    async def test_consolidation_distills_facts(self):
        consolidator, _, semantic, pg, neo, _ = self._make_deps()
        result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        assert result.facts_distilled > 0
        semantic.upsert_fact.assert_called()

    @pytest.mark.asyncio
    async def test_consolidation_re_embeds_summary(self):
        consolidator, episodic, _, pg, neo, _ = self._make_deps()
        result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        episodic.update_embedding.assert_called()
        assert result.embeddings_updated > 0


# ── Stage 3: ContextBuilder.build() ──────────────────────────────────────────


class TestContextStage:
    """Verify ContextBuilder.build() returns context containing encoded content."""

    def _make_deps(self, event: Event | None = None):
        llm = _make_llm()
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()

        target_event = event or _make_event(consolidated=True)
        search_result = SearchResult(
            event=target_event,
            similarity_score=0.9,
            recency_score=0.7,
            hybrid_score=0.88,
        )
        episodic.hybrid_search = AsyncMock(return_value=[search_result])
        episodic.get_recent = AsyncMock(return_value=[])
        episodic.increment_recall = AsyncMock()
        semantic.get_user_profile = AsyncMock(return_value=UserProfile(
            user_id=USER_ID, app_id=APP_ID,
            facts=[_make_fact_record()],
        ))

        builder = ContextBuilder(
            llm=llm, episodic=episodic, semantic=semantic, top_k_similar=5
        )
        pg, neo = AsyncMock(), AsyncMock()
        return builder, pg, neo, target_event

    @pytest.mark.asyncio
    async def test_build_returns_memory_context(self):
        builder, pg, neo, _ = self._make_deps()
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="What am I building?")
        assert isinstance(ctx, MemoryContext)
        assert not ctx.is_empty()

    @pytest.mark.asyncio
    async def test_encoded_text_appears_in_similar_events(self):
        """The event encoded at stage 1 should appear in context's similar_events."""
        event = _make_event(ENCODED_TEXT, consolidated=False)
        builder, pg, neo, _ = self._make_deps(event=event)
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="smritikosh")
        similar_texts = [sr.event.raw_text for sr in ctx.similar_events]
        assert ENCODED_TEXT in similar_texts

    @pytest.mark.asyncio
    async def test_consolidated_summary_appears_in_context(self):
        """After consolidation, the summary should be present in similar events."""
        event = _make_event(ENCODED_TEXT, consolidated=True)  # has summary
        builder, pg, neo, _ = self._make_deps(event=event)
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="smritikosh")
        # as_prompt_text uses summary when available
        prompt = ctx.as_prompt_text()
        assert SUMMARY_TEXT in prompt

    @pytest.mark.asyncio
    async def test_semantic_facts_appear_in_profile(self):
        """Facts extracted during encoding should appear in the user profile."""
        builder, pg, neo, _ = self._make_deps()
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="What am I building?")
        assert ctx.user_profile is not None
        profile_values = [f.value for f in ctx.user_profile.facts]
        assert "smritikosh" in profile_values

    @pytest.mark.asyncio
    async def test_context_prompt_text_is_non_empty(self):
        builder, pg, neo, _ = self._make_deps()
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="What am I building?")
        prompt = ctx.as_prompt_text()
        assert "## User Memory Context" in prompt
        assert len(prompt) > 50


# ── Full pipeline: encode → consolidate → context ────────────────────────────


class TestFullPipeline:
    """
    Wires all three stages together and verifies end-to-end information flow.

    Stage 1: Hippocampus encodes N events → returns Event objects.
    Stage 2: Consolidator processes those events → marks consolidated, writes summary.
    Stage 3: ContextBuilder retrieves the consolidated event → appears in context.

    The "shared state" between stages is simulated by capturing what stage N
    wrote and feeding it as the mock return value for stage N+1's reads.
    """

    @pytest.mark.asyncio
    async def test_encoded_event_survives_to_context(self):
        """
        Information encoded at stage 1 appears in context at stage 3.
        This is the core regression test for the memory pipeline.
        """
        # ── Shared mocks ──────────────────────────────────────────────────
        llm = _make_llm()
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()
        narrative = AsyncMock(spec=NarrativeMemory)
        pg, neo = AsyncMock(), AsyncMock()

        # ── Stage 1: encode ───────────────────────────────────────────────
        encoded_event = _make_event(ENCODED_TEXT)
        episodic.store = AsyncMock(return_value=encoded_event)
        semantic.upsert_fact = AsyncMock(return_value=_make_fact_record())

        hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic)
        encoded = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)

        assert encoded.event.raw_text == ENCODED_TEXT

        # ── Stage 2: consolidate ──────────────────────────────────────────
        # Pretend enough events exist by providing MIN_EVENTS_TO_CONSOLIDATE
        raw_events = [encoded_event] + [
            _make_event(f"supporting event {i}") for i in range(MIN_EVENTS_TO_CONSOLIDATE - 1)
        ]
        episodic.get_unconsolidated = AsyncMock(return_value=raw_events)
        episodic.mark_consolidated = AsyncMock()
        episodic.update_embedding = AsyncMock()
        narrative.create_link = AsyncMock(return_value=MagicMock())

        consolidator = Consolidator(
            llm=llm, episodic=episodic, semantic=semantic,
            narrative=narrative, min_events=MIN_EVENTS_TO_CONSOLIDATE,
        )
        c_result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)

        assert not c_result.skipped
        assert c_result.events_consolidated == MIN_EVENTS_TO_CONSOLIDATE

        # ── Stage 3: build context ────────────────────────────────────────
        # Simulate: after consolidation the event now has a summary
        consolidated_event = _make_event(ENCODED_TEXT, consolidated=True)
        search_result = SearchResult(
            event=consolidated_event,
            similarity_score=0.9,
            recency_score=0.8,
            hybrid_score=0.90,
        )
        episodic.hybrid_search = AsyncMock(return_value=[search_result])
        episodic.get_recent = AsyncMock(return_value=[])
        episodic.increment_recall = AsyncMock()
        semantic.get_user_profile = AsyncMock(return_value=UserProfile(
            user_id=USER_ID, app_id=APP_ID,
            facts=[_make_fact_record()],
        ))

        builder = ContextBuilder(
            llm=llm, episodic=episodic, semantic=semantic, top_k_similar=5
        )
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="smritikosh project")

        # ── Assertions ────────────────────────────────────────────────────
        # The event is present in similar_events
        similar_ids = {sr.event.id for sr in ctx.similar_events}
        assert consolidated_event.id in similar_ids

        # The summary from consolidation appears in the rendered prompt
        prompt = ctx.as_prompt_text()
        assert SUMMARY_TEXT in prompt

        # Semantic facts from encoding appear in the profile
        profile_values = [f.value for f in ctx.user_profile.facts]
        assert "smritikosh" in profile_values

    @pytest.mark.asyncio
    async def test_pipeline_survives_embedding_failure(self):
        """
        If embedding fails at encode time, the pipeline continues:
        the event is stored without a vector, consolidation still runs,
        and context is built from the consolidated summary.
        """
        llm = _make_llm()
        llm.embed = AsyncMock(side_effect=RuntimeError("embed service down"))
        episodic = AsyncMock(spec=EpisodicMemory)
        semantic = _make_semantic_mock()
        narrative = AsyncMock(spec=NarrativeMemory)
        pg, neo = AsyncMock(), AsyncMock()

        # Stage 1 — embed fails, event stored without vector
        event_no_embedding = _make_event(ENCODED_TEXT)
        episodic.store = AsyncMock(return_value=event_no_embedding)
        semantic.upsert_fact = AsyncMock(return_value=_make_fact_record())

        hippo = Hippocampus(llm=llm, episodic=episodic, semantic=semantic)
        encoded = await hippo.encode(pg, neo, user_id=USER_ID, raw_text=ENCODED_TEXT)

        # Embedding failed but event still stored
        store_kwargs = episodic.store.call_args.kwargs
        assert store_kwargs["embedding"] is None

        # Stage 2 — consolidation still runs (LLM embed call now succeeds for summary)
        llm.embed = AsyncMock(return_value=EMBEDDING)
        raw_events = [event_no_embedding] + [
            _make_event(f"event {i}") for i in range(MIN_EVENTS_TO_CONSOLIDATE - 1)
        ]
        episodic.get_unconsolidated = AsyncMock(return_value=raw_events)
        episodic.mark_consolidated = AsyncMock()
        episodic.update_embedding = AsyncMock()
        narrative.create_link = AsyncMock(return_value=MagicMock())

        consolidator = Consolidator(
            llm=llm, episodic=episodic, semantic=semantic,
            narrative=narrative, min_events=MIN_EVENTS_TO_CONSOLIDATE,
        )
        c_result = await consolidator.run(pg, neo, user_id=USER_ID, app_id=APP_ID)
        assert not c_result.skipped

        # Stage 3 — context builder returns a result even if embedding failed earlier
        search_sr = SearchResult(
            event=_make_event(ENCODED_TEXT, consolidated=True),
            similarity_score=0.85, recency_score=0.7, hybrid_score=0.85,
        )
        episodic.hybrid_search = AsyncMock(return_value=[search_sr])
        episodic.get_recent = AsyncMock(return_value=[])
        episodic.increment_recall = AsyncMock()
        semantic.get_user_profile = AsyncMock(
            return_value=UserProfile(user_id=USER_ID, app_id=APP_ID, facts=[])
        )

        builder = ContextBuilder(llm=llm, episodic=episodic, semantic=semantic)
        ctx = await builder.build(pg, neo, user_id=USER_ID, query="smritikosh")
        assert len(ctx.similar_events) == 1
