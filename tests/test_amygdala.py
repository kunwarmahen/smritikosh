"""
Tests for Amygdala importance scorer.

How to run:
    pytest tests/test_amygdala.py -v

All tests are pure unit tests — no mocking needed (Amygdala has no I/O).
"""

import pytest

from smritikosh.processing.amygdala import Amygdala, _MAX_SCORE, _MIN_SCORE


@pytest.fixture
def amygdala() -> Amygdala:
    return Amygdala()


class TestScoreBounds:
    def test_score_always_in_range(self, amygdala):
        texts = [
            "",
            "?",
            "hi",
            "I decided to pivot the entire company strategy immediately.",
            "REMEMBER THIS: critical deadline tomorrow for investor meeting",
            "maybe? possibly? not sure perhaps might could be",
        ]
        for text in texts:
            score = amygdala.score(text)
            assert _MIN_SCORE <= score <= _MAX_SCORE, (
                f"Score {score} out of range for: {text!r}"
            )

    def test_empty_string_does_not_crash(self, amygdala):
        score = amygdala.score("")
        assert _MIN_SCORE <= score <= _MAX_SCORE


class TestBoosts:
    def test_explicit_importance_marker_boosts(self, amygdala):
        high = amygdala.score("Remember this: I want green UI everywhere")
        baseline = amygdala.score("I think the UI might look okay")
        assert high > baseline

    def test_decision_or_commitment_boosts(self, amygdala):
        score = amygdala.score("I decided to build an AI memory startup")
        assert score > 0.6

    def test_emotional_signal_boosts(self, amygdala):
        score_emotional = amygdala.score("I am so excited about this product launch")
        score_neutral = amygdala.score("the product is in development")
        assert score_emotional > score_neutral

    def test_first_person_boosts(self, amygdala):
        first = amygdala.score("My preference is dark mode")
        third = amygdala.score("The preference is dark mode")
        assert first > third

    def test_high_value_topic_boosts(self, amygdala):
        score = amygdala.score("The startup needs an investor by next month")
        assert score >= 0.6


class TestPenalties:
    def test_uncertainty_lowers_score(self, amygdala):
        certain = amygdala.score("I prefer green UI")
        uncertain = amygdala.score("I think maybe I possibly prefer green UI")
        assert certain > uncertain

    def test_question_lowers_score(self, amygdala):
        statement = amygdala.score("I work in AI infrastructure")
        question = amygdala.score("Do you think AI infrastructure is good?")
        assert statement > question

    def test_short_text_is_penalised(self, amygdala):
        short_score = amygdala.score("ok")
        long_score = amygdala.score("I prefer to use dark mode in all applications")
        assert short_score < long_score

    def test_hedged_text_does_not_exceed_certain_text(self, amygdala):
        hedged = amygdala.score("maybe I might possibly be interested in AI")
        certain = amygdala.score("I am deeply interested in AI infrastructure")
        assert certain > hedged


class TestBreakdown:
    def test_returns_tuple(self, amygdala):
        result = amygdala.score_with_breakdown("I decided to launch the product")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_breakdown_final_matches_score(self, amygdala):
        text = "I am excited about the AI startup launch deadline"
        score = amygdala.score(text)
        _, breakdown = amygdala.score_with_breakdown(text)
        assert abs(breakdown.final - score) < 1e-9

    def test_breakdown_records_boosts(self, amygdala):
        _, breakdown = amygdala.score_with_breakdown("I decided to launch the product")
        boost_labels = [label for label, _ in breakdown.boosts]
        assert "decision_or_commitment" in boost_labels

    def test_breakdown_records_penalties(self, amygdala):
        _, breakdown = amygdala.score_with_breakdown("maybe?")
        penalty_labels = [label for label, _ in breakdown.penalties]
        assert "question" in penalty_labels
        assert "uncertainty" in penalty_labels

    def test_breakdown_as_dict(self, amygdala):
        _, breakdown = amygdala.score_with_breakdown("test")
        d = breakdown.as_dict()
        assert set(d.keys()) == {"base", "boosts", "penalties", "final"}


class TestEdgeCases:
    def test_all_caps_text_still_scores(self, amygdala):
        score = amygdala.score("I AM VERY EXCITED ABOUT THIS STARTUP LAUNCH")
        assert _MIN_SCORE <= score <= _MAX_SCORE

    def test_multiple_boosts_stack(self, amygdala):
        multi_signal = amygdala.score(
            "Remember this: I decided and I am excited about the startup launch"
        )
        single_signal = amygdala.score("I like green")
        assert multi_signal > single_signal

    def test_score_is_deterministic(self, amygdala):
        text = "I need to launch the product before the deadline"
        assert amygdala.score(text) == amygdala.score(text)
