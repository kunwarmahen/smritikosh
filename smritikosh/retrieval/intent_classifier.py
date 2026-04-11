"""
IntentClassifier — two-tier query intent classification.

Tier 1 (sync, always runs): keyword heuristic.
    Fast, zero I/O. Returns a result with a confidence score.

Tier 2 (async, opt-in): LLM classification.
    Triggered when keyword confidence is below `llm_confidence_threshold`
    (default 0.5 — i.e. fewer than 2 strong keyword matches). A single cheap
    haiku-class LLM call classifies the query and can detect compound intents
    ("code review for my career path" = CAREER + TECHNICAL). Falls back to
    the keyword result if the LLM call fails or returns an unrecognised intent.

Intent → retrieval strategy:
    CAREER          — boost importance; user wants contextually significant memories
    TECHNICAL       — boost similarity; exact topic match matters most
    PERSONAL        — balanced; preferences and habits need recency + identity
    PROJECT_PLANNING — boost recency; current state of projects matters most
    HISTORICAL_RECALL — boost frequency; memories the user often returns to
    GENERAL         — default weights; no strong signal

Usage (keyword only — backward-compatible):
    classifier = IntentClassifier()
    result = classifier.classify("What should I focus on for my career?")
    # result.intent == QueryIntent.CAREER

Usage (with LLM fallback):
    classifier = IntentClassifier(llm=llm_adapter)
    result = await classifier.classify_async("Tell me about my project timeline")
    # result.intent == QueryIntent.PROJECT_PLANNING (LLM resolves ambiguous wording)
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from smritikosh.memory.episodic import HybridWeights

if TYPE_CHECKING:
    from smritikosh.llm.adapter import LLMAdapter

logger = logging.getLogger(__name__)


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
    confidence: float          # 0.0–1.0
    weights: HybridWeights
    via_llm: bool = False      # True if the LLM path was used
    secondary_intents: list[QueryIntent] = field(default_factory=list)


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

# ── LLM prompt ────────────────────────────────────────────────────────────────

_VALID_INTENTS = [i.value for i in QueryIntent]

_LLM_SYSTEM_PROMPT = f"""You are an intent classifier for a personal AI memory system.
Classify the user's query into one or more of these intents:
  career           — job, salary, promotion, professional development
  technical        — code, debugging, implementation, architecture
  personal         — preferences, habits, feelings, personality traits
  project_planning — plans, roadmaps, timelines, product decisions
  historical_recall — retrieving past events, "do you remember", "last time"
  general          — anything that doesn't fit the above

Rules:
- Return ONLY a JSON object, no markdown, no explanation.
- primary_intent: the single best-matching intent (string).
- secondary_intents: list of other intents that also apply (may be empty).
- confidence: 0.0–1.0, how certain you are about the primary intent.

Valid intent values: {_VALID_INTENTS}

Example output:
{{"primary_intent": "career", "secondary_intents": ["technical"], "confidence": 0.9}}"""


# ── IntentClassifier ──────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Two-tier intent classifier: keyword heuristic + optional LLM fallback.

    Args:
        llm:                      Optional LLMAdapter. When provided, `classify_async`
                                  uses it for low-confidence queries.
        llm_confidence_threshold: Keyword confidence below this triggers the LLM.
                                  Default 0.5 = fewer than ~1.5 keyword matches.
    """

    def __init__(
        self,
        llm: "LLMAdapter | None" = None,
        llm_confidence_threshold: float = 0.5,
    ) -> None:
        self.llm = llm
        self.llm_confidence_threshold = llm_confidence_threshold

    def classify(self, query: str) -> IntentResult:
        """
        Synchronous keyword-based classification. Always available, zero I/O.

        Use this when you can't await, or when keyword confidence is high.
        For full two-tier classification, call `classify_async` instead.
        """
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

    async def classify_async(self, query: str) -> IntentResult:
        """
        Two-tier classification: keyword first, LLM when confidence is low.

        If keyword confidence >= `llm_confidence_threshold`, returns the keyword
        result immediately (no LLM call). Otherwise, makes a single LLM call to
        classify the query. Falls back to the keyword result on any LLM failure.
        """
        keyword_result = self.classify(query)

        # Skip LLM if keyword match is already strong, or if no LLM is configured
        if self.llm is None or keyword_result.confidence >= self.llm_confidence_threshold:
            return keyword_result

        try:
            llm_result = await self._classify_with_llm(query)
            if llm_result is not None:
                return llm_result
        except Exception as exc:
            logger.warning(
                "LLM intent classification failed — using keyword fallback",
                extra={"query_preview": query[:100], "error": str(exc)},
            )

        return keyword_result

    async def _classify_with_llm(self, query: str) -> IntentResult | None:
        """
        Call the LLM and parse its intent response.

        Returns None if the response cannot be parsed or contains an unknown intent,
        so the caller can fall back to the keyword result.
        """
        raw = await self.llm.complete(
            messages=[
                {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=100,
        )

        parsed = self.llm._parse_json(raw)

        primary_str = str(parsed.get("primary_intent", "")).lower().strip()
        try:
            primary = QueryIntent(primary_str)
        except ValueError:
            logger.warning(
                "LLM returned unknown intent %r — falling back to keywords", primary_str
            )
            return None

        secondary: list[QueryIntent] = []
        for s in parsed.get("secondary_intents", []):
            try:
                secondary.append(QueryIntent(str(s).lower().strip()))
            except ValueError:
                pass  # ignore unrecognised secondary intents

        raw_confidence = parsed.get("confidence", 0.8)
        confidence = max(0.0, min(1.0, float(raw_confidence)))

        return IntentResult(
            intent=primary,
            confidence=confidence,
            weights=_INTENT_WEIGHTS[primary],
            via_llm=True,
            secondary_intents=secondary,
        )
