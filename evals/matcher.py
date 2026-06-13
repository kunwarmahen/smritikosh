"""
Fact matching and scoring for the extraction eval harness.

Pure functions — no LLM, no I/O — so the scoring logic itself is unit-testable
and deterministic. The optional LLM judge (for semantically equivalent values
that lexical matching misses) lives in runner.py, layered on top of this.

Matching model
--------------
A *predicted* fact is the extractor's output: {category, key, value, confidence}.
An *expected* spec is a golden-set entry:

    {
      "category": "diet",              # or a list of acceptable categories
      "value": "vegetarian",
      "aliases": ["veg", "no meat"],   # optional alternative phrasings
      "optional": false                # optional=True: TP if found, never FN
    }

A predicted fact matches a spec when the category is acceptable AND the value
matches lexically (normalized equality, containment either way, or token
Jaccard >= JACCARD_THRESHOLD) against the value or any alias.

Keys are LLM-invented labels (ui_color, current_project, …) and far too
unstable to require — they are reported but never scored.

*Forbidden* specs (anti-contamination guards) use strict matching only —
normalized equality or alias equality — so "not vegetarian" does not trip a
forbidden "vegetarian".

Scoring is greedy one-to-one: each expected spec consumes at most one
predicted fact and vice versa. Leftover predictions are FPs, leftover
required expectations are FNs.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

JACCARD_THRESHOLD = 0.6

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT_RE.sub(" ", text.lower())
    return _WS_RE.sub(" ", text).strip()


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def values_match(predicted: str, expected: str) -> bool:
    """Lexical value equivalence: equality, containment, or token overlap."""
    p, e = normalize(predicted), normalize(expected)
    if not p or not e:
        return p == e
    if p == e:
        return True
    if p in e or e in p:
        return True
    return _jaccard(p, e) >= JACCARD_THRESHOLD


def values_match_strict(predicted: str, expected: str) -> bool:
    """Strict equivalence (forbidden specs): normalized equality only."""
    return normalize(predicted) == normalize(expected)


# ── Specs and results ─────────────────────────────────────────────────────────


@dataclass
class ExpectedFact:
    """One golden-set expectation (see module docstring for the JSON shape)."""

    categories: list[str]
    value: str
    aliases: list[str] = field(default_factory=list)
    optional: bool = False

    @classmethod
    def from_json(cls, obj: dict) -> "ExpectedFact":
        category = obj["category"]
        return cls(
            categories=[category] if isinstance(category, str) else list(category),
            value=obj["value"],
            aliases=list(obj.get("aliases", [])),
            optional=bool(obj.get("optional", False)),
        )

    def all_values(self) -> list[str]:
        return [self.value, *self.aliases]

    def matches(self, fact: dict, *, strict: bool = False) -> bool:
        if fact.get("category") not in self.categories:
            return False
        compare = values_match_strict if strict else values_match
        predicted_value = str(fact.get("value", ""))
        return any(compare(predicted_value, v) for v in self.all_values())


@dataclass
class CaseScore:
    """Match outcome for one golden case."""

    case_id: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    violations: int = 0  # predicted facts matching a forbidden spec
    matched: list[dict] = field(default_factory=list)        # {expected, predicted}
    unmatched_expected: list[dict] = field(default_factory=list)  # FN detail
    unmatched_predicted: list[dict] = field(default_factory=list)  # FP detail
    violation_facts: list[dict] = field(default_factory=list)
    error: str | None = None  # extraction call failed

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_case(
    case_id: str,
    predicted: list[dict],
    expected: list[ExpectedFact],
    forbidden: list[ExpectedFact] | None = None,
) -> CaseScore:
    """
    Greedy one-to-one matching of predicted facts against expected specs.

    Required specs are matched before optional ones so an optional spec never
    steals the only prediction that would satisfy a required one.
    """
    score = CaseScore(case_id=case_id)
    remaining = list(predicted)

    for spec in sorted(expected, key=lambda s: s.optional):
        hit = next((f for f in remaining if spec.matches(f)), None)
        if hit is not None:
            remaining.remove(hit)
            score.tp += 1
            score.matched.append({"expected": spec.value, "predicted": hit})
        elif not spec.optional:
            score.fn += 1
            score.unmatched_expected.append(
                {"category": "|".join(spec.categories), "value": spec.value}
            )

    score.fp = len(remaining)
    score.unmatched_predicted = remaining

    for spec in forbidden or []:
        for fact in predicted:
            if spec.matches(fact, strict=True):
                score.violations += 1
                score.violation_facts.append(fact)

    return score


@dataclass
class AggregateScore:
    """Micro-averaged totals across all cases."""

    cases: list[CaseScore] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return sum(c.tp for c in self.cases)

    @property
    def fp(self) -> int:
        return sum(c.fp for c in self.cases)

    @property
    def fn(self) -> int:
        return sum(c.fn for c in self.cases)

    @property
    def violations(self) -> int:
        return sum(c.violations for c in self.cases)

    @property
    def errors(self) -> int:
        return sum(1 for c in self.cases if c.error)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0
