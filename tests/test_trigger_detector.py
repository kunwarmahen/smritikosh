"""
Tests for TriggerDetector and transcript_utils.

All tests run offline — no DB, LLM, or API keys needed.

Run:
    pytest tests/test_trigger_detector.py -v
"""

import pytest

from smritikosh.processing.trigger_detector import TriggerDetector, TriggerResult
from smritikosh.processing.transcript_utils import (
    build_delta_prompt,
    prepare_transcript,
    strip_sentinels,
    user_turns_only,
)


# ── TriggerDetector tests ─────────────────────────────────────────────────────


class TestTriggerDetector:

    def setup_method(self):
        self.detector = TriggerDetector()

    # ── check() ───────────────────────────────────────────────────────────────

    def test_simple_trigger_match(self):
        triggered, phrases = self.detector.check("I always drink black coffee in the morning.")
        assert triggered is True
        assert len(phrases) >= 1
        assert any("I always" in p for p in phrases)

    def test_no_trigger_in_generic_text(self):
        triggered, phrases = self.detector.check("The weather is nice today.")
        assert triggered is False
        assert phrases == []

    def test_remember_trigger(self):
        triggered, _ = self.detector.check("Remember that I prefer dark mode.")
        assert triggered is True

    def test_i_prefer_trigger(self):
        triggered, phrases = self.detector.check("I prefer Python over JavaScript.")
        assert triggered is True

    def test_my_goal_trigger(self):
        triggered, _ = self.detector.check("My goal is to launch by Q3.")
        assert triggered is True

    def test_we_decided_trigger(self):
        triggered, _ = self.detector.check("We decided to use PostgreSQL as our primary DB.")
        assert triggered is True

    def test_i_believe_trigger(self):
        triggered, _ = self.detector.check("I believe remote work increases productivity.")
        assert triggered is True

    def test_i_hate_trigger(self):
        triggered, _ = self.detector.check("I hate meetings without an agenda.")
        assert triggered is True

    def test_case_insensitive(self):
        triggered, _ = self.detector.check("I ALWAYS wake up at 6am.")
        assert triggered is True

    def test_multiple_triggers_deduplicated(self):
        text = "I always drink coffee. I always start early."
        triggered, phrases = self.detector.check(text)
        assert triggered is True
        # "I always" should appear only once (deduplicated)
        assert len([p for p in phrases if "always" in p.lower()]) == 1

    def test_check_result_dataclass(self):
        result = self.detector.check_result("I prefer vim as my editor.")
        assert isinstance(result, TriggerResult)
        assert result.triggered is True
        assert len(result.matched_phrases) >= 1

    def test_extra_patterns(self):
        detector = TriggerDetector(extra_patterns=[r"\bmy startup\b"])
        triggered, phrases = detector.check("My startup is doing great this quarter.")
        assert triggered is True

    # ── filter_turns() ────────────────────────────────────────────────────────

    def test_filter_turns_returns_only_user_with_triggers(self):
        turns = [
            {"role": "user", "content": "I prefer dark mode always."},
            {"role": "assistant", "content": "I prefer dark mode too."},
            {"role": "user", "content": "The sky is blue."},
            {"role": "user", "content": "My goal is to run a marathon."},
        ]
        result = self.detector.filter_turns(turns)
        assert len(result) == 2  # first and last user turn
        for turn in result:
            assert turn["role"] == "user"
            assert "_trigger_phrases" in turn

    def test_filter_turns_skips_assistant(self):
        turns = [
            {"role": "assistant", "content": "I always do my best."},
        ]
        result = self.detector.filter_turns(turns)
        assert result == []

    def test_any_triggered_fast_path(self):
        turns = [
            {"role": "user", "content": "Just asking about the weather."},
            {"role": "user", "content": "I always meditate before bed."},
        ]
        assert self.detector.any_triggered(turns) is True

    def test_any_triggered_no_match(self):
        turns = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "user", "content": "Tell me a joke."},
        ]
        assert self.detector.any_triggered(turns) is False

    def test_collect_all_phrases(self):
        turns = [
            {"role": "user", "content": "I prefer Python. I believe in open source."},
            {"role": "user", "content": "I always use dark mode."},
        ]
        phrases = self.detector.collect_all_phrases(turns)
        assert len(phrases) >= 2
        assert len(phrases) == len(set(phrases))  # deduplicated


# ── transcript_utils tests ────────────────────────────────────────────────────


class TestStripSentinels:

    def test_strips_single_block(self):
        text = (
            "Before.\n"
            "<!-- smritikosh:context-start -->\n"
            "You know this: user likes coffee.\n"
            "<!-- smritikosh:context-end -->\n"
            "After."
        )
        result = strip_sentinels(text)
        assert "coffee" not in result
        assert "Before." in result
        assert "After." in result

    def test_strips_multiple_blocks(self):
        text = (
            "<!-- smritikosh:context-start -->block1<!-- smritikosh:context-end --> "
            "middle "
            "<!-- smritikosh:context-start -->block2<!-- smritikosh:context-end -->"
        )
        result = strip_sentinels(text)
        assert "block1" not in result
        assert "block2" not in result
        assert "middle" in result

    def test_no_sentinels_returns_unchanged(self):
        text = "Hello, I always drink coffee."
        assert strip_sentinels(text) == text

    def test_strips_case_insensitive(self):
        text = "<!-- SMRITIKOSH:CONTEXT-START -->secret<!-- SMRITIKOSH:CONTEXT-END -->"
        result = strip_sentinels(text)
        assert "secret" not in result

    def test_multiline_block_stripped(self):
        text = (
            "Before\n"
            "<!-- smritikosh:context-start -->\n"
            "line1\nline2\nline3\n"
            "<!-- smritikosh:context-end -->\n"
            "After"
        )
        result = strip_sentinels(text)
        assert "line1" not in result
        assert "line2" not in result


class TestUserTurnsOnly:

    def test_filters_to_user_role(self):
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "I prefer dark mode"},
        ]
        result = user_turns_only(turns)
        assert len(result) == 2
        assert all(t["role"] == "user" for t in result)

    def test_strips_sentinels_from_content(self):
        turns = [
            {
                "role": "user",
                "content": (
                    "<!-- smritikosh:context-start -->injected<!-- smritikosh:context-end -->"
                    " I actually prefer vim."
                ),
            }
        ]
        result = user_turns_only(turns)
        assert len(result) == 1
        assert "injected" not in result[0]["content"]
        assert "vim" in result[0]["content"]

    def test_empty_turns_after_filter_excluded(self):
        turns = [
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "OK"},
        ]
        result = user_turns_only(turns)
        assert result == []

    def test_empty_input(self):
        assert user_turns_only([]) == []


class TestBuildDeltaPrompt:

    def test_no_existing_facts(self):
        turns = [{"role": "user", "content": "I love Python."}]
        prompt = build_delta_prompt(turns)
        assert "no existing knowledge" in prompt.lower()
        assert "Python" in prompt

    def test_with_existing_facts(self):
        from dataclasses import dataclass, field

        @dataclass
        class FakeFact:
            category: str
            key: str
            value: str
            confidence: float

        existing = [FakeFact("preference", "editor", "vim", 0.9)]
        turns = [{"role": "user", "content": "Actually I switched to neovim."}]
        prompt = build_delta_prompt(turns, existing_facts=existing)
        assert "vim" in prompt
        assert "ONLY" in prompt or "only" in prompt
        assert "neovim" in prompt

    def test_last_turn_index_windowing(self):
        turns = [
            {"role": "user", "content": "old content"},
            {"role": "user", "content": "new content"},
        ]
        prompt = build_delta_prompt(turns, last_turn_index=1)
        assert "new content" in prompt
        assert "old content" not in prompt


class TestPrepareTranscript:

    def test_basic_pipeline(self):
        turns = [
            {"role": "user", "content": "I always drink coffee."},
            {"role": "assistant", "content": "Good for you."},
            {"role": "user", "content": "I prefer Python."},
        ]
        result = prepare_transcript(turns)
        assert result.turns_count == 2
        assert "coffee" in result.combined_text
        assert "Good for you" not in result.combined_text
        assert result.stripped_sentinels is False

    def test_detects_sentinels(self):
        turns = [
            {
                "role": "assistant",
                "content": "<!-- smritikosh:context-start -->ctx<!-- smritikosh:context-end -->",
            },
            {"role": "user", "content": "Hello"},
        ]
        result = prepare_transcript(turns)
        assert result.stripped_sentinels is True
        assert "ctx" not in result.combined_text

    def test_last_turn_index_windowing(self):
        turns = [
            {"role": "user", "content": "turn 0"},
            {"role": "user", "content": "turn 1"},
            {"role": "user", "content": "turn 2"},
        ]
        result = prepare_transcript(turns, last_turn_index=1)
        assert result.turns_count == 2
        assert "turn 1" in result.combined_text
        assert "turn 0" not in result.combined_text

    def test_empty_turns(self):
        result = prepare_transcript([])
        assert result.turns_count == 0
        assert result.combined_text == ""
        assert result.stripped_sentinels is False
