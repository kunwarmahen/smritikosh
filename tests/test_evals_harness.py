"""
Tests for the extraction-quality eval harness (evals/).

Covers the deterministic parts — matching, scoring, golden-set loading, and
the runner's prompt routing — with a fake LLM adapter. No real LLM calls.
"""

import pytest

from evals.matcher import (
    AggregateScore,
    ExpectedFact,
    normalize,
    score_case,
    values_match,
    values_match_strict,
)
from evals.runner import (
    GOLDEN_DIR,
    GoldenCase,
    category_breakdown,
    extract_for_case,
    load_cases,
    run_eval,
)

# ── Fakes ─────────────────────────────────────────────────────────────────────


class FakeLLM:
    """Stands in for LLMAdapter: canned extraction output + judge answers."""

    def __init__(self, facts=None, judge_answer="no"):
        self.facts = facts or []
        self.judge_answer = judge_answer
        self.prompts: list[str] = []
        self._chat_model = "fake/model"

    async def extract_structured(self, *, prompt, schema_description, example_output):
        self.prompts.append(prompt)
        return {"facts": self.facts}

    async def complete(self, *, messages, **kwargs):
        return self.judge_answer


def fact(category, value, key="k", confidence=0.9):
    return {"category": category, "key": key, "value": value, "confidence": confidence}


# ── normalize / values_match ──────────────────────────────────────────────────


def test_normalize():
    assert normalize("  Dark-Mode!! ") == "dark mode"
    assert normalize("Café") == "cafe"
    assert normalize("UI_color") == "ui_color"


def test_values_match_equality_and_containment():
    assert values_match("Vegetarian", "vegetarian")
    assert values_match("senior data engineer", "data engineer")
    assert values_match("dark", "dark mode")


def test_values_match_jaccard():
    assert values_match("100 paying customers by December", "100 paying customers")
    assert not values_match("green", "purple")


def test_values_match_strict_rejects_containment():
    assert values_match_strict("Vegetarian", "vegetarian")
    assert not values_match_strict("not vegetarian", "vegetarian")


# ── ExpectedFact ──────────────────────────────────────────────────────────────


def test_expected_fact_category_list_and_aliases():
    spec = ExpectedFact.from_json(
        {"category": ["hobby", "interest"], "value": "hiking", "aliases": ["loves hiking"]}
    )
    assert spec.matches(fact("interest", "loves hiking"))
    assert spec.matches(fact("hobby", "Hiking"))
    assert not spec.matches(fact("tool", "hiking"))


def test_expected_fact_strict_uses_aliases():
    spec = ExpectedFact.from_json(
        {"category": "diet", "value": "ramen", "aliases": ["eats ramen"]}
    )
    assert spec.matches(fact("diet", "Eats Ramen"), strict=True)
    assert not spec.matches(fact("diet", "loves ramen noodles"), strict=True)


# ── score_case ────────────────────────────────────────────────────────────────


def test_score_case_tp_fp_fn():
    expected = [
        ExpectedFact.from_json({"category": "diet", "value": "vegetarian"}),
        ExpectedFact.from_json({"category": "location", "value": "Pune"}),
    ]
    predicted = [fact("diet", "vegetarian"), fact("tool", "Vim")]
    score = score_case("c", predicted, expected)
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)
    assert score.precision == 0.5
    assert score.recall == 0.5


def test_score_case_optional_not_fn_but_consumes_prediction():
    expected = [
        ExpectedFact.from_json({"category": "tool", "value": "tmux", "optional": True}),
    ]
    # Missing optional → no FN; vacuously perfect (nothing wrong happened)
    empty = score_case("c", [], expected)
    assert (empty.tp, empty.fp, empty.fn) == (0, 0, 0)
    assert empty.f1 == 1.0
    # Present optional → TP, not FP
    found = score_case("c", [fact("tool", "tmux")], expected)
    assert (found.tp, found.fp, found.fn) == (1, 0, 0)


def test_score_case_required_matched_before_optional():
    # One prediction, one required + one optional spec both matching it:
    # the required spec must consume it, leaving no FN.
    expected = [
        ExpectedFact.from_json({"category": "diet", "value": "vegan", "optional": True}),
        ExpectedFact.from_json({"category": "diet", "value": "vegan"}),
    ]
    score = score_case("c", [fact("diet", "vegan")], expected)
    assert (score.tp, score.fp, score.fn) == (1, 0, 0)


def test_score_case_one_to_one_matching():
    # Two identical predictions, one expected: second prediction is an FP.
    expected = [ExpectedFact.from_json({"category": "tool", "value": "Redis"})]
    score = score_case("c", [fact("tool", "redis"), fact("tool", "redis")], expected)
    assert (score.tp, score.fp, score.fn) == (1, 1, 0)


def test_score_case_forbidden_violations():
    forbidden = [ExpectedFact.from_json({"category": "location", "value": "Tokyo"})]
    score = score_case("c", [fact("location", "Tokyo")], [], forbidden)
    assert score.violations == 1
    assert score.fp == 1  # also an FP since it matches nothing expected


def test_score_case_forbidden_is_strict():
    forbidden = [ExpectedFact.from_json({"category": "diet", "value": "vegetarian"})]
    score = score_case(
        "c",
        [fact("diet", "not vegetarian")],
        [ExpectedFact.from_json({"category": "diet", "value": "not vegetarian"})],
        forbidden,
    )
    assert score.violations == 0
    assert score.tp == 1


def test_empty_case_perfect_score():
    score = score_case("c", [], [])
    assert score.precision == 1.0
    assert score.recall == 1.0


# ── AggregateScore ────────────────────────────────────────────────────────────


def test_aggregate_micro_average():
    a = score_case("a", [fact("diet", "vegan")],
                   [ExpectedFact.from_json({"category": "diet", "value": "vegan"})])
    b = score_case("b", [fact("tool", "Vim")],
                   [ExpectedFact.from_json({"category": "tool", "value": "Emacs"})])
    agg = AggregateScore(cases=[a, b])
    assert (agg.tp, agg.fp, agg.fn) == (1, 1, 1)
    assert agg.precision == 0.5
    assert agg.recall == 0.5


# ── Golden set integrity ──────────────────────────────────────────────────────


def test_golden_set_loads_and_is_valid():
    cases = load_cases(GOLDEN_DIR)
    assert len(cases) >= 50
    kinds = {c.kind for c in cases}
    assert kinds == {"event", "session"}
    for case in cases:
        if case.kind == "event":
            assert case.content, f"{case.id}: event case needs content"
        else:
            assert case.turns, f"{case.id}: session case needs turns"
        for spec in case.expected + case.forbidden:
            assert spec.value
            assert spec.categories


def test_golden_ids_unique():
    cases = load_cases(GOLDEN_DIR)
    ids = [c.id for c in cases]
    assert len(ids) == len(set(ids))


# ── Runner ────────────────────────────────────────────────────────────────────


async def test_extract_event_builds_event_prompt():
    llm = FakeLLM(facts=[fact("diet", "vegan")])
    case = GoldenCase(id="e", kind="event", expected=[], forbidden=[], content="I am vegan.")
    predicted = await extract_for_case(llm, case)
    assert predicted == [fact("diet", "vegan")]
    assert "I am vegan." in llm.prompts[0]


async def test_extract_session_filters_assistant_and_uses_existing_facts():
    llm = FakeLLM()
    case = GoldenCase(
        id="s",
        kind="session",
        expected=[],
        forbidden=[],
        turns=[
            {"role": "assistant", "content": "you are vegetarian"},
            {"role": "user", "content": "what should I cook?"},
        ],
        existing_facts=[{"category": "diet", "key": "restriction", "value": "vegetarian"}],
    )
    await extract_for_case(llm, case)
    prompt = llm.prompts[0]
    assert "what should I cook?" in prompt
    assert "diet/restriction: vegetarian" in prompt          # delta section present
    assert "you are vegetarian" not in prompt                 # assistant turn filtered


async def test_extract_session_skips_llm_when_no_user_turns():
    llm = FakeLLM(facts=[fact("diet", "vegan")])
    case = GoldenCase(
        id="s", kind="session", expected=[], forbidden=[],
        turns=[{"role": "assistant", "content": "hello"}],
    )
    predicted = await extract_for_case(llm, case)
    assert predicted == []
    assert llm.prompts == []  # no LLM call, mirroring production


async def test_run_eval_end_to_end_with_fake_llm():
    llm = FakeLLM(facts=[fact("diet", "vegetarian")])
    cases = [
        GoldenCase(
            id="hit", kind="event", content="x",
            expected=[ExpectedFact.from_json({"category": "diet", "value": "vegetarian"})],
            forbidden=[],
        ),
        GoldenCase(
            id="miss", kind="event", content="y",
            expected=[ExpectedFact.from_json({"category": "tool", "value": "Vim"})],
            forbidden=[],
        ),
    ]
    agg = await run_eval(cases, llm=llm, concurrency=2)
    assert [s.case_id for s in agg.cases] == ["hit", "miss"]
    assert agg.tp == 1
    assert agg.fn == 1
    assert agg.fp == 1  # the vegetarian prediction on the "miss" case


async def test_run_eval_extraction_error_counts_expected_as_fn():
    class BoomLLM(FakeLLM):
        async def extract_structured(self, **kwargs):
            raise ValueError("malformed JSON")

    cases = [
        GoldenCase(
            id="boom", kind="event", content="x",
            expected=[
                ExpectedFact.from_json({"category": "diet", "value": "vegan"}),
                ExpectedFact.from_json({"category": "tool", "value": "Vim", "optional": True}),
            ],
            forbidden=[],
        )
    ]
    agg = await run_eval(cases, llm=BoomLLM())
    assert agg.errors == 1
    assert agg.fn == 1  # only the required spec counts


async def test_judge_upgrades_near_miss():
    llm = FakeLLM(facts=[fact("belief", "OSS will surpass proprietary models")],
                  judge_answer="Yes — same fact.")
    cases = [
        GoldenCase(
            id="j", kind="event", content="x",
            expected=[ExpectedFact.from_json(
                {"category": "belief", "value": "open source models overtake closed"}
            )],
            forbidden=[],
        )
    ]
    no_judge = await run_eval(cases, llm=llm)
    assert no_judge.f1 == 0.0
    judged = await run_eval(cases, llm=llm, judge=True)
    assert judged.tp == 1
    assert judged.f1 == 1.0


def test_category_breakdown():
    case = GoldenCase(
        id="c", kind="event", content="x",
        expected=[
            ExpectedFact.from_json({"category": "diet", "value": "vegan"}),
            ExpectedFact.from_json({"category": "tool", "value": "Vim"}),
        ],
        forbidden=[],
    )
    score = score_case("c", [fact("diet", "vegan")], case.expected)
    stats = category_breakdown([case], AggregateScore(cases=[score]))
    assert stats["diet"] == {"expected": 1, "found": 1, "recall": 1.0}
    assert stats["tool"]["found"] == 0
