"""
IntentClassifier — two-tier query intent classification (E3 + E4 meta-cognition).

Tier 1 (sync, always runs): keyword heuristic.
    Fast, zero I/O. Returns a result with a confidence score. Confidence is
    deliberately capped below 1.0 (_KEYWORD_CONFIDENCE_CEILING) so keyword
    matching can never fully "win" — a raised llm_confidence_threshold will
    always be able to route queries to the LLM. A tie between two intents
    halves confidence (the ambiguity the LLM tier exists for) and records the
    runner-up as a secondary intent.

Tier 2 (async, opt-in): LLM classification.
    Triggered when keyword confidence is below `llm_confidence_threshold`
    (default 0.5). A single cheap haiku-class LLM call classifies the query,
    detects compound intents ("code review for my career path" = TECHNICAL +
    CAREER), and estimates complexity. Falls back to the keyword result if
    the LLM call fails or returns an unrecognised intent.

Multi-intent blending (E3): when secondary intents are present, retrieval
weights are a 70/30 convex blend of the primary and top-secondary weight
rows, so compound queries pull from both retrieval strategies.

Caching (E3): classify_async results are cached by normalised-query hash
(TTL + LRU bound) so repeated queries — dashboards re-fetching context,
retry loops, benchmark runs — don't re-pay the LLM call.

Complexity tier (E4 meta-cognition, FUTURE.md #8): every result carries a
`complexity` tier that downstream pipelines route on:
    simple   — direct lookup ("what coffee does Alice like?"): trimmed retrieval
    moderate — default ContextBuilder pipeline
    complex  — deliberation-worthy ("should I take this job offer?"):
               narrative chains + belief alignment added to the context

Intent → retrieval strategy:
    CAREER          — boost importance; user wants contextually significant memories
    TECHNICAL       — boost similarity; exact topic match matters most
    PERSONAL        — balanced; preferences and habits need recency + identity
    PROJECT_PLANNING — boost recency; current state of projects matters most
    HISTORICAL_RECALL — boost frequency; memories the user often returns to
    GENERAL         — default weights; no strong signal
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field, replace
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


class ComplexityTier(StrEnum):
    """How much reasoning/retrieval effort a query deserves (E4 meta-cognition)."""
    SIMPLE = "simple"        # single lookup; trimmed retrieval
    MODERATE = "moderate"    # standard ContextBuilder pipeline
    COMPLEX = "complex"      # deliberation: chains + belief alignment


@dataclass
class IntentResult:
    intent: QueryIntent
    confidence: float          # 0.0–1.0
    weights: HybridWeights
    via_llm: bool = False      # True if the LLM path was used
    secondary_intents: list[QueryIntent] = field(default_factory=list)
    complexity: ComplexityTier = ComplexityTier.MODERATE


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

# Keyword confidence saturates here, NOT at 1.0 (E3): even a keyword-dense
# query stays below full certainty, so the LLM threshold remains meaningful
# and compound intents keyword-matching can't hide are still reachable.
_KEYWORD_CONFIDENCE_CEILING = 0.75

# Markers of deliberation-worthy queries (E4 meta-cognition). Any hit → complex.
_COMPLEX_MARKERS = [
    "should i", "should we", "decide", "decision", "worth it", "pros and cons",
    "trade-off", "tradeoff", "trade off", "versus", " vs ", " vs.", "weigh",
    "compare", "risk", "whether to", "or should", "which option", "what if i",
    "is it better", "make sense to",
]

# Word count at or below which a non-complex query is a simple lookup.
_SIMPLE_MAX_WORDS = 7

# ── LLM prompt ────────────────────────────────────────────────────────────────

_VALID_INTENTS = [i.value for i in QueryIntent]
_VALID_COMPLEXITY = [c.value for c in ComplexityTier]

_LLM_SYSTEM_PROMPT = f"""You are an intent classifier for a personal AI memory system.
Classify the user's query into one or more of these intents:
  career           — job, salary, promotion, professional development
  technical        — code, debugging, implementation, architecture
  personal         — preferences, habits, feelings, personality traits
  project_planning — plans, roadmaps, timelines, product decisions
  historical_recall — retrieving past events, "do you remember", "last time"
  general          — anything that doesn't fit the above

Also rate the query's complexity:
  simple   — a direct lookup of one known fact or preference
  moderate — needs assembled context but no deliberation
  complex  — a decision, trade-off, or question that deserves multi-angle reasoning

Rules:
- Return ONLY a JSON object, no markdown, no explanation.
- primary_intent: the single best-matching intent (string).
- secondary_intents: list of other intents that also apply (may be empty).
- confidence: 0.0–1.0, how certain you are about the primary intent.
- complexity: one of {_VALID_COMPLEXITY}.

Valid intent values: {_VALID_INTENTS}

Example output:
{{"primary_intent": "career", "secondary_intents": ["technical"], "confidence": 0.9, "complexity": "moderate"}}"""


# ── Weight blending (E3) ──────────────────────────────────────────────────────

# Primary/secondary mix for compound intents. Convex, so blended weights still
# sum to 1.0 and pass HybridWeights validation.
_BLEND_PRIMARY = 0.7


def _blend_weights(primary: QueryIntent, secondary: QueryIntent) -> HybridWeights:
    """70/30 convex blend of two intents' weight rows (non-weight params from primary)."""
    p = _INTENT_WEIGHTS[primary]
    s = _INTENT_WEIGHTS[secondary]
    a, b = _BLEND_PRIMARY, 1.0 - _BLEND_PRIMARY
    return replace(
        p,
        similarity=a * p.similarity + b * s.similarity,
        recency=a * p.recency + b * s.recency,
        importance=a * p.importance + b * s.importance,
        frequency=a * p.frequency + b * s.frequency,
        contextual_match=a * p.contextual_match + b * s.contextual_match,
    )


def _resolve_weights(
    primary: QueryIntent, secondary_intents: list[QueryIntent]
) -> HybridWeights:
    """Weights for a result: blended with the top secondary intent when present."""
    for s in secondary_intents:
        if s != primary:
            return _blend_weights(primary, s)
    return _INTENT_WEIGHTS[primary]


def classify_complexity(query: str) -> ComplexityTier:
    """Heuristic complexity tier — deliberation markers beat length."""
    lowered = f" {query.lower().strip()} "
    if any(marker in lowered for marker in _COMPLEX_MARKERS):
        return ComplexityTier.COMPLEX
    if len(query.split()) <= _SIMPLE_MAX_WORDS:
        return ComplexityTier.SIMPLE
    return ComplexityTier.MODERATE


# ── IntentClassifier ──────────────────────────────────────────────────────────


class IntentClassifier:
    """
    Two-tier intent classifier: keyword heuristic + optional LLM fallback.

    Args:
        llm:                      Optional LLMAdapter. When provided, `classify_async`
                                  uses it for low-confidence queries.
        llm_confidence_threshold: Keyword confidence below this triggers the LLM.
                                  Default 0.5 = fewer than 2 keyword matches.
        cache_size:               Max cached classify_async results (0 disables).
        cache_ttl_s:              Seconds a cached classification stays valid.
    """

    def __init__(
        self,
        llm: "LLMAdapter | None" = None,
        llm_confidence_threshold: float = 0.5,
        cache_size: int = 1024,
        cache_ttl_s: float = 3600.0,
    ) -> None:
        self.llm = llm
        self.llm_confidence_threshold = llm_confidence_threshold
        self.cache_size = cache_size
        self.cache_ttl_s = cache_ttl_s
        # query-hash → (monotonic_expiry, IntentResult). Plain dict in insertion
        # order; eviction pops the oldest entry (single-process, asyncio — no lock).
        self._cache: dict[str, tuple[float, IntentResult]] = {}

    def classify(self, query: str) -> IntentResult:
        """
        Synchronous keyword-based classification. Always available, zero I/O.

        Use this when you can't await, or when keyword confidence is high.
        For full two-tier classification, call `classify_async` instead.
        """
        lowered = query.lower()
        complexity = classify_complexity(query)

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
                complexity=complexity,
            )

        # Capped normalisation (E3): 2 matches = 0.5, 3+ = the 0.75 ceiling.
        confidence = min(best_score / 4.0, _KEYWORD_CONFIDENCE_CEILING)

        # A tie between two intents is exactly the ambiguity the LLM tier is
        # for: halve confidence so the threshold routes it there, and record
        # the runner-up so weights blend even on the keyword-only path.
        secondary: list[QueryIntent] = []
        runners = [i for i, s in scores.items() if s == best_score and i != best_intent]
        if runners:
            confidence *= 0.5
            secondary = runners[:1]

        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            weights=_resolve_weights(best_intent, secondary),
            secondary_intents=secondary,
            complexity=complexity,
        )

    async def classify_async(self, query: str) -> IntentResult:
        """
        Two-tier classification: keyword first, LLM when confidence is low.

        If keyword confidence >= `llm_confidence_threshold`, returns the keyword
        result immediately (no LLM call). Otherwise, makes a single LLM call to
        classify the query. Falls back to the keyword result on any LLM failure.
        Results are cached by normalised-query hash (TTL-bounded).
        """
        cache_key = self._cache_key(query)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        keyword_result = self.classify(query)

        # Skip LLM if keyword match is already strong, or if no LLM is configured
        if self.llm is None or keyword_result.confidence >= self.llm_confidence_threshold:
            self._cache_put(cache_key, keyword_result)
            return keyword_result

        try:
            llm_result = await self._classify_with_llm(query)
            if llm_result is not None:
                self._cache_put(cache_key, llm_result)
                return llm_result
        except Exception as exc:
            logger.warning(
                "LLM intent classification failed — using keyword fallback",
                extra={"query_preview": query[:100], "error": str(exc)},
            )

        # Don't cache the fallback after an LLM failure: the next identical
        # query should get another shot at the LLM once the provider recovers.
        return keyword_result

    # ── Cache helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _cache_key(query: str) -> str:
        normalised = " ".join(query.lower().split())
        return hashlib.sha1(normalised.encode()).hexdigest()

    def _cache_get(self, key: str) -> IntentResult | None:
        if self.cache_size <= 0:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        expiry, result = entry
        if time.monotonic() > expiry:
            self._cache.pop(key, None)
            return None
        return result

    def _cache_put(self, key: str, result: IntentResult) -> None:
        if self.cache_size <= 0:
            return
        if len(self._cache) >= self.cache_size and key not in self._cache:
            # Evict the oldest insertion — cheap approximation of LRU that
            # needs no per-hit bookkeeping.
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (time.monotonic() + self.cache_ttl_s, result)

    # ── LLM tier ───────────────────────────────────────────────────────────

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
                candidate = QueryIntent(str(s).lower().strip())
            except ValueError:
                continue  # ignore unrecognised secondary intents
            if candidate != primary and candidate not in secondary:
                secondary.append(candidate)

        raw_confidence = parsed.get("confidence", 0.8)
        confidence = max(0.0, min(1.0, float(raw_confidence)))

        # LLM complexity wins when valid; heuristic fills the gap otherwise.
        complexity_str = str(parsed.get("complexity", "")).lower().strip()
        try:
            complexity = ComplexityTier(complexity_str)
        except ValueError:
            complexity = classify_complexity(query)

        return IntentResult(
            intent=primary,
            confidence=confidence,
            weights=_resolve_weights(primary, secondary),
            via_llm=True,
            secondary_intents=secondary,
            complexity=complexity,
        )
