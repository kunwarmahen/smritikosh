"""
Tests for IntentClassifier, QueryIntent, and IntentResult.

All tests are pure unit tests — no mocks, no I/O.
"""

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
