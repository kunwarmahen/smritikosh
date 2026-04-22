"""
TriggerDetector — lightweight heuristic pre-filter for passive memory extraction.

Before invoking the (expensive) LLM extraction pass on a conversation window,
this module scans user turns for high-signal phrases that indicate a durable
personal fact is likely present. If no triggers fire, the extraction LLM is
skipped entirely, reducing cost for low-information turns.

Matched phrases are stored in source_meta.trigger_phrases so the lineage is
traceable all the way to the stored fact.

Usage:
    from smritikosh.processing.trigger_detector import TriggerDetector

    detector = TriggerDetector()
    triggered, phrases = detector.check("I always drink black coffee")
    # → (True, ["I always"])

    triggered_turns = detector.filter_turns(turns)
    # → only the turns that matched at least one trigger pattern
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Trigger pattern definitions ───────────────────────────────────────────────
# Ordered from most-specific to least-specific. Longer / more-specific patterns
# are listed first so they get priority in the matched_phrases list.

_RAW_PATTERNS: list[str] = [
    # Explicit memory signals
    r"\bremember (?:that|this|when)\b",
    r"\bimportant(?:\s*:|\s+to\s+note)\b",
    r"\bdon'?t (?:ever )?(?:let me )?forget\b",
    r"\balways remember\b",
    # Strong preference / aversion
    r"\bI (?:really )?(?:hate|love|can'?t stand|can'?t live without)\b",
    r"\bI (?:always|never)\b",
    r"\bI (?:strongly )?prefer\b",
    r"\bmy (?:favourite|favorite)\b",
    # Goal / decision
    r"\bwe decided\b",
    r"\bI(?:'ve)? decided\b",
    r"\bmy (?:goal|target|objective|ambition) is\b",
    r"\bI(?:'m| am) working (?:on|toward)\b",
    r"\bI(?:'m| am) trying to\b",
    # Belief / value
    r"\bI believe\b",
    r"\bI think (?:that )?(?:we|it's|the best|everyone|most)\b",
    r"\bin my opinion\b",
    r"\bfor me,?\s+(?:the|it's|nothing|everything)\b",
    # Stable personal facts
    r"\bmy team\b",
    r"\bI prefer\b",
    r"\bmy (?:company|startup|project|product)\b",
    r"\bI(?:'m| am) (?:a|an)\b",          # "I'm a vegetarian", "I'm an engineer"
    r"\bI (?:work|worked) (?:at|for|with)\b",
    r"\bI (?:live|grew up|was born) in\b",
    r"\bI(?:'ve| have) been (?:doing|using|working)\b",
]

# Compile once at import time
_COMPILED: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in _RAW_PATTERNS
]


@dataclass
class TriggerResult:
    """Result of scanning a single piece of text."""
    triggered: bool
    matched_phrases: list[str] = field(default_factory=list)


class TriggerDetector:
    """
    Scans text or lists of conversation turns for high-signal memory phrases.

    Thread-safe (stateless). Instantiate once and reuse.
    """

    def __init__(self, extra_patterns: list[str] | None = None) -> None:
        """
        Args:
            extra_patterns: Additional regex patterns to include (compiled at init).
        """
        self._patterns: list[re.Pattern] = list(_COMPILED)
        if extra_patterns:
            self._patterns.extend(
                re.compile(p, re.IGNORECASE) for p in extra_patterns
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, text: str) -> tuple[bool, list[str]]:
        """
        Check whether text contains any high-signal memory triggers.

        Returns:
            (triggered, matched_phrases) — matched_phrases is deduplicated and
            preserves first-match order.
        """
        seen: set[str] = set()
        matches: list[str] = []
        for pattern in self._patterns:
            m = pattern.search(text)
            if m:
                phrase = m.group(0)
                if phrase not in seen:
                    seen.add(phrase)
                    matches.append(phrase)
        return bool(matches), matches

    def check_result(self, text: str) -> TriggerResult:
        """Same as check() but returns a TriggerResult dataclass."""
        triggered, phrases = self.check(text)
        return TriggerResult(triggered=triggered, matched_phrases=phrases)

    def filter_turns(
        self,
        turns: list[dict],
        role: str = "user",
    ) -> list[dict]:
        """
        Return only the turns (dicts with 'role' and 'content' keys) that
        contain at least one trigger pattern and match the given role.

        Each returned turn is augmented with a '_trigger_phrases' key listing
        the matched phrases — useful for populating source_meta.

        Args:
            turns:  List of {'role': str, 'content': str} dicts.
            role:   Only inspect turns with this role (default 'user').
        """
        result = []
        for turn in turns:
            if turn.get("role") != role:
                continue
            triggered, phrases = self.check(turn.get("content", ""))
            if triggered:
                augmented = dict(turn)
                augmented["_trigger_phrases"] = phrases
                result.append(augmented)
        return result

    def any_triggered(self, turns: list[dict], role: str = "user") -> bool:
        """
        Fast path: return True as soon as any turn matching `role` triggers.
        Does not collect all matches — use filter_turns for the full list.
        """
        for turn in turns:
            if turn.get("role") != role:
                continue
            triggered, _ = self.check(turn.get("content", ""))
            if triggered:
                return True
        return False

    def collect_all_phrases(self, turns: list[dict], role: str = "user") -> list[str]:
        """Return a deduplicated list of all trigger phrases across all matching turns."""
        seen: set[str] = set()
        phrases: list[str] = []
        for turn in turns:
            if turn.get("role") != role:
                continue
            _, matched = self.check(turn.get("content", ""))
            for p in matched:
                if p not in seen:
                    seen.add(p)
                    phrases.append(p)
        return phrases
