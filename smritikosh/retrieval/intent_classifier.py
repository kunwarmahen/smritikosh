"""
IntentClassifier — fast heuristic query intent classification.

Classifies a query into one of six intent categories using keyword matching
(no LLM call, no I/O). Each intent maps to a distinct HybridWeights tuning
so the retrieval strategy adapts to what the user is actually asking.

Intent → retrieval strategy:
    CAREER          — boost importance; user wants contextually significant memories
    TECHNICAL       — boost similarity; exact topic match matters most
    PERSONAL        — balanced; preferences and habits need recency + identity
    PROJECT_PLANNING — boost recency; current state of projects matters most
    HISTORICAL_RECALL — boost frequency; memories the user often returns to
    GENERAL         — default weights; no strong signal

Usage:
    classifier = IntentClassifier()
    result = classifier.classify("What should I focus on for my career?")
    # result.intent == QueryIntent.CAREER
    # result.weights == HybridWeights(similarity=0.30, ...)
"""

from dataclasses import dataclass
from enum import StrEnum

from smritikosh.memory.episodic import HybridWeights


class QueryIntent(StrEnum):
    CAREER = "career"
    TECHNICAL = "technical"
    PERSONAL = "personal"
    PROJECT_PLANNING = "project_planning"
    HISTORICAL_RECALL = "historical_recall"
    GENERAL = "general"


@dataclass
class IntentResult:
    intent: QueryIntent
    confidence: float     # 0.0–1.0; based on number of keyword matches
    weights: HybridWeights


# ── Per-intent HybridWeights ──────────────────────────────────────────────────
# Each row must sum to 1.0.

_INTENT_WEIGHTS: dict[QueryIntent, HybridWeights] = {
    # Career: importance dominates — we want the most meaningful memories,
    # not just the most recent ones.
    QueryIntent.CAREER: HybridWeights(
        similarity=0.30, recency=0.20, importance=0.25,
        frequency=0.15, contextual_match=0.10,
    ),
    # Technical: similarity dominates — exact topic relevance matters most.
    QueryIntent.TECHNICAL: HybridWeights(
        similarity=0.50, recency=0.15, importance=0.15,
        frequency=0.15, contextual_match=0.05,
    ),
    # Personal: balanced — preferences need both recency and identity context.
    QueryIntent.PERSONAL: HybridWeights(
        similarity=0.35, recency=0.25, importance=0.20,
        frequency=0.15, contextual_match=0.05,
    ),
    # Project planning: recency dominates — current project state matters most.
    QueryIntent.PROJECT_PLANNING: HybridWeights(
        similarity=0.35, recency=0.30, importance=0.20,
        frequency=0.10, contextual_match=0.05,
    ),
    # Historical recall: frequency dominates — surface the memories the user
    # returns to most often, not just the most similar ones.
    QueryIntent.HISTORICAL_RECALL: HybridWeights(
        similarity=0.25, recency=0.10, importance=0.15,
        frequency=0.40, contextual_match=0.10,
    ),
    # General: default weights, no strong signal detected.
    QueryIntent.GENERAL: HybridWeights(
        similarity=0.40, recency=0.30, importance=0.15,
        frequency=0.15, contextual_match=0.0,
    ),
}

# ── Keyword lists per intent ──────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.CAREER: [
        "career", "job", "role", "salary", "promotion", "hiring", "work experience",
        "interview", "resume", "cv", "profession", "occupation",
    ],
    QueryIntent.TECHNICAL: [
        "how to", "implement", "code", "build", "debug", "error", "architecture",
        "api", "database", "function", "class", "algorithm", "library", "framework",
    ],
    QueryIntent.PERSONAL: [
        "prefer", "like", "love", "feel", "habit", "routine", "favorite",
        "usually", "always", "never", "enjoy", "dislike", "personality",
    ],
    QueryIntent.PROJECT_PLANNING: [
        "plan", "project", "roadmap", "timeline", "launch", "deadline",
        "milestone", "sprint", "startup", "product", "feature", "scope",
    ],
    QueryIntent.HISTORICAL_RECALL: [
        "remember", "recall", "when did", "last time", "before", "previously",
        "history", "past", "ago", "earlier", "do you know", "have i",
    ],
}


# ── IntentClassifier ──────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Classifies a query into a QueryIntent using keyword heuristics.

    Each keyword match in the query adds 1 to that intent's score.
    The intent with the highest score wins. Ties are broken by dict order
    (career > technical > personal > project_planning > historical_recall).
    If no keywords match, GENERAL is returned with confidence 0.0.

    Confidence is normalised: 3+ keyword matches = 1.0 confidence.
    """

    def classify(self, query: str) -> IntentResult:
        lowered = query.lower()

        scores: dict[QueryIntent, int] = {intent: 0 for intent in _INTENT_KEYWORDS}
        for intent, keywords in _INTENT_KEYWORDS.items():
            for kw in keywords:
                if kw in lowered:
                    scores[intent] += 1

        best_intent = max(scores, key=lambda i: scores[i])
        best_score = scores[best_intent]

        if best_score == 0:
            return IntentResult(
                intent=QueryIntent.GENERAL,
                confidence=0.0,
                weights=_INTENT_WEIGHTS[QueryIntent.GENERAL],
            )

        # Normalise: 3 or more matches → full confidence
        confidence = min(1.0, best_score / 3.0)
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            weights=_INTENT_WEIGHTS[best_intent],
        )
