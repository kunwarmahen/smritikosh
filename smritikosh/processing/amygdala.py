"""
Amygdala — importance scorer for incoming events.

In the human brain, the amygdala tags memories with emotional weight before
they are stored. High-signal events (emotionally charged, personally relevant,
survival-critical) get stronger neural encoding.

Here we replicate that function with a fast heuristic scorer that runs
synchronously before any LLM calls — no extra latency, no API cost.

Score is in [0.1, 1.0]:
    0.1 – 0.4  : low importance  (casual remarks, filler)
    0.5         : neutral baseline
    0.6 – 0.8  : personally relevant (preferences, goals, named facts)
    0.9 – 1.0  : high importance  (decisions, commitments, explicit markers)
"""

import re
from dataclasses import dataclass, field


@dataclass
class ScoringBreakdown:
    """Detailed breakdown of how the importance score was calculated."""
    base: float
    boosts: list[tuple[str, float]] = field(default_factory=list)   # (reason, delta)
    penalties: list[tuple[str, float]] = field(default_factory=list)
    final: float = 0.0

    def as_dict(self) -> dict:
        return {
            "base": self.base,
            "boosts": self.boosts,
            "penalties": self.penalties,
            "final": self.final,
        }


# ── Pattern tables ─────────────────────────────────────────────────────────────
# Each entry: (list_of_trigger_keywords_or_patterns, score_delta, label)
#
# Ordered from highest-impact to lowest so the breakdown is readable.

_BOOST_RULES: list[tuple[list[str], float, str]] = [
    # Explicit importance signals — user is flagging something as critical
    (["remember this", "important:", "note:", "don't forget", "must ", "critical"],
     0.30, "explicit_importance"),

    # Decisions and commitments — high-value episodic anchors
    (["decided", "decision", "committed", "will ", "going to", "deadline", "launch", "ship"],
     0.20, "decision_or_commitment"),

    # Emotional charge — amygdala-relevant
    (["love", "hate", "excited", "worried", "afraid", "proud", "frustrated", "anxious"],
     0.15, "emotional_signal"),

    # Personal statements — first-person facts are more durable
    (["i am", "i'm", "my ", "i want", "i need", "i prefer", "i like", "i dislike", "we are"],
     0.10, "first_person"),

    # Domain-relevant high-value topics for AI/startup context
    (["startup", "product", "investor", "revenue", "goal", "objective", "strategy"],
     0.10, "high_value_topic"),
]

_PENALTY_RULES: list[tuple[list[str], float, str]] = [
    # Hedged / uncertain statements carry less durable information
    (["maybe", "perhaps", "possibly", "might ", "could be", "not sure", "i think"],
     -0.10, "uncertainty"),

    # Pure questions — useful context but rarely a fact worth storing
    (["?"],
     -0.05, "question"),

    # Very short texts are often throwaway
    # (handled by length check, not keyword)
]

_MIN_SCORE = 0.1
_MAX_SCORE = 1.0
_BASE_SCORE = 0.5
_SHORT_TEXT_THRESHOLD = 20   # chars — texts shorter than this get penalised


class Amygdala:
    """
    Fast heuristic importance scorer.

    Runs synchronously — no LLM call, no I/O. Called by Hippocampus
    before the async embedding + extraction pipeline so the score is
    ready to attach to the Event on first write.

    Usage:
        amygdala = Amygdala()
        score = amygdala.score("I decided to pivot the startup to AI memory")
        # → ~0.9

        score, breakdown = amygdala.score_with_breakdown("maybe?")
        # → 0.35, ScoringBreakdown(...)
    """

    def score(self, text: str) -> float:
        """Return a single importance score in [0.1, 1.0]."""
        return self.score_with_breakdown(text)[0]

    def score_with_breakdown(self, text: str) -> tuple[float, ScoringBreakdown]:
        """Return (score, breakdown) for debugging and tests."""
        lower = text.lower().strip()
        bd = ScoringBreakdown(base=_BASE_SCORE)
        running = _BASE_SCORE

        # Short-text penalty (before pattern matching)
        if len(lower) < _SHORT_TEXT_THRESHOLD:
            delta = -0.10
            bd.penalties.append(("short_text", delta))
            running += delta

        # Apply boost rules
        for keywords, delta, label in _BOOST_RULES:
            if any(kw in lower for kw in keywords):
                bd.boosts.append((label, delta))
                running += delta

        # Apply penalty rules
        for keywords, delta, label in _PENALTY_RULES:
            if any(kw in lower for kw in keywords):
                bd.penalties.append((label, delta))
                running += delta

        bd.final = max(_MIN_SCORE, min(_MAX_SCORE, running))
        return bd.final, bd
