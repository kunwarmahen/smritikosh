"""
Tests for IntentClassifier, QueryIntent, and IntentResult.

Unit tests are split into two groups:
  - Keyword path: pure sync, no mocks, no I/O.
  - LLM path: mocked LLMAdapter, tests classify_async() two-tier logic.
"""

import json
from unittest.mock import AsyncMock

import pytest

from smritikosh.retrieval.intent_classifier import (
    IntentClassifier,
    IntentResult,
    QueryIntent,
    _INTENT_WEIGHTS,
)


@pytest.fixture
def classifier() -> IntentClassifier:
    return IntentClassifier()


# ── Weight table sanity ───────────────────────────────────────────────────────


class TestWeightTable:
    def test_all_intents_have_weights(self):
        for intent in QueryIntent:
            assert intent in _INTENT_WEIGHTS

    def test_all_weight_sets_sum_to_one(self):
        for intent, weights in _INTENT_WEIGHTS.items():
            total = (
                weights.similarity
                + weights.recency
                + weights.importance
                + weights.frequency
                + weights.contextual_match
            )
            assert abs(total - 1.0) < 1e-6, f"{intent} weights sum to {total}"

    def test_historical_recall_has_highest_frequency_weight(self):
        hist = _INTENT_WEIGHTS[QueryIntent.HISTORICAL_RECALL]
        others = [w for k, w in _INTENT_WEIGHTS.items() if k != QueryIntent.HISTORICAL_RECALL]
        assert all(hist.frequency >= w.frequency for w in others)

    def test_technical_has_highest_similarity_weight(self):
        tech = _INTENT_WEIGHTS[QueryIntent.TECHNICAL]
        others = [w for k, w in _INTENT_WEIGHTS.items() if k != QueryIntent.TECHNICAL]
        assert all(tech.similarity >= w.similarity for w in others)


# ── IntentResult ──────────────────────────────────────────────────────────────


class TestIntentResult:
    def test_fields(self, classifier):
        result = classifier.classify("career job")
        assert isinstance(result, IntentResult)
        assert isinstance(result.intent, QueryIntent)
        assert isinstance(result.confidence, float)
        assert result.weights is not None

    def test_weights_match_intent(self, classifier):
        result = classifier.classify("career job role")
        assert result.weights == _INTENT_WEIGHTS[result.intent]

    def test_keyword_result_via_llm_is_false(self, classifier):
        result = classifier.classify("career job")
        assert result.via_llm is False

    def test_secondary_intents_default_empty(self, classifier):
        result = classifier.classify("career job")
        assert result.secondary_intents == []


# ── Intent detection ──────────────────────────────────────────────────────────


class TestIntentDetection:
    def test_career_intent(self, classifier):
        result = classifier.classify("What career advice would you give me?")
        assert result.intent == QueryIntent.CAREER

    def test_technical_intent(self, classifier):
        result = classifier.classify("How to implement a FAISS index?")
        assert result.intent == QueryIntent.TECHNICAL

    def test_personal_intent(self, classifier):
        result = classifier.classify("What do I usually prefer for UI?")
        assert result.intent == QueryIntent.PERSONAL

    def test_project_planning_intent(self, classifier):
        result = classifier.classify("What is the plan and timeline for our product launch?")
        assert result.intent == QueryIntent.PROJECT_PLANNING

    def test_historical_recall_intent(self, classifier):
        result = classifier.classify("Do you remember what I said last time?")
        assert result.intent == QueryIntent.HISTORICAL_RECALL

    def test_general_fallback_no_keywords(self, classifier):
        result = classifier.classify("Hello there")
        assert result.intent == QueryIntent.GENERAL

    def test_empty_query_returns_general(self, classifier):
        result = classifier.classify("")
        assert result.intent == QueryIntent.GENERAL

    def test_case_insensitive(self, classifier):
        lower = classifier.classify("career job")
        upper = classifier.classify("CAREER JOB")
        assert lower.intent == upper.intent

    def test_multi_keyword_same_intent(self, classifier):
        result = classifier.classify("career job role salary promotion")
        assert result.intent == QueryIntent.CAREER

    def test_strongest_intent_wins(self, classifier):
        # 3 career keywords vs 1 technical keyword — career should win
        result = classifier.classify("career job role how")
        assert result.intent == QueryIntent.CAREER


# ── Confidence ────────────────────────────────────────────────────────────────


class TestConfidence:
    def test_general_has_zero_confidence(self, classifier):
        result = classifier.classify("Hello")
        assert result.confidence == 0.0

    def test_single_match_has_low_confidence(self, classifier):
        result = classifier.classify("career")
        assert 0.0 < result.confidence < 1.0

    def test_three_matches_gives_full_confidence(self, classifier):
        result = classifier.classify("career job role salary")
        assert result.confidence == pytest.approx(1.0)

    def test_more_matches_higher_confidence(self, classifier):
        one_match = classifier.classify("career")
        two_matches = classifier.classify("career job")
        assert two_matches.confidence >= one_match.confidence

    def test_confidence_capped_at_one(self, classifier):
        # Many career keywords — confidence should not exceed 1.0
        result = classifier.classify(
            "career job role salary promotion hiring interview resume cv"
        )
        assert result.confidence <= 1.0


# ── classify_async — LLM path ─────────────────────────────────────────────────


def _make_llm(response: dict) -> AsyncMock:
    """Return a mock LLMAdapter whose complete() returns the given dict as JSON."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=json.dumps(response))
    # _parse_json is a static method — delegate to the real implementation
    from smritikosh.llm.adapter import LLMAdapter
    llm._parse_json = LLMAdapter._parse_json
    return llm


class TestClassifyAsync:
    @pytest.mark.asyncio
    async def test_returns_keyword_result_when_confidence_high(self):
        """High keyword confidence → no LLM call."""
        llm = AsyncMock()
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        # "career job role" → 3 keywords → confidence 1.0 ≥ 0.5
        result = await classifier.classify_async("career job role")
        llm.complete.assert_not_called()
        assert result.intent == QueryIntent.CAREER
        assert result.via_llm is False

    @pytest.mark.asyncio
    async def test_calls_llm_when_keyword_confidence_low(self):
        """Low keyword confidence → LLM is called."""
        llm = _make_llm({"primary_intent": "project_planning", "secondary_intents": [], "confidence": 0.85})
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        # "roadmap" alone → 1 keyword → confidence 0.33 < 0.5
        result = await classifier.classify_async("What is the roadmap")
        llm.complete.assert_called_once()
        assert result.intent == QueryIntent.PROJECT_PLANNING
        assert result.via_llm is True

    @pytest.mark.asyncio
    async def test_no_llm_configured_uses_keyword_only(self):
        """No LLM configured → classify_async behaves identically to classify."""
        classifier = IntentClassifier(llm=None)
        result = await classifier.classify_async("career job")
        assert result.intent == QueryIntent.CAREER
        assert result.via_llm is False

    @pytest.mark.asyncio
    async def test_llm_result_confidence_is_used(self):
        llm = _make_llm({"primary_intent": "technical", "secondary_intents": [], "confidence": 0.92})
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("show me")
        assert result.intent == QueryIntent.TECHNICAL
        assert abs(result.confidence - 0.92) < 1e-9

    @pytest.mark.asyncio
    async def test_llm_result_includes_secondary_intents(self):
        llm = _make_llm({
            "primary_intent": "career",
            "secondary_intents": ["technical"],
            "confidence": 0.8,
        })
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("show me")
        assert QueryIntent.TECHNICAL in result.secondary_intents

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_keyword(self):
        """LLM throws → keyword result returned, no exception propagated."""
        llm = AsyncMock()
        llm.complete = AsyncMock(side_effect=RuntimeError("LLM service down"))
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("tell me about my project")
        assert result.intent == QueryIntent.PROJECT_PLANNING
        assert result.via_llm is False

    @pytest.mark.asyncio
    async def test_llm_unknown_intent_falls_back_to_keyword(self):
        """LLM returns an unrecognised intent string → fallback to keyword."""
        llm = _make_llm({"primary_intent": "UNKNOWN_GARBAGE", "secondary_intents": [], "confidence": 0.9})
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("tell me about my project")
        assert result.via_llm is False
        assert result.intent == QueryIntent.PROJECT_PLANNING

    @pytest.mark.asyncio
    async def test_llm_confidence_clamped_to_range(self):
        """LLM returning out-of-range confidence is clamped to [0, 1]."""
        llm = _make_llm({"primary_intent": "personal", "secondary_intents": [], "confidence": 5.0})
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("show me")
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_weights_are_set_from_llm_intent(self):
        llm = _make_llm({"primary_intent": "historical_recall", "secondary_intents": [], "confidence": 0.8})
        classifier = IntentClassifier(llm=llm, llm_confidence_threshold=0.5)
        result = await classifier.classify_async("show me")
        assert result.weights == _INTENT_WEIGHTS[QueryIntent.HISTORICAL_RECALL]
