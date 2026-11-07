"""
Tests for labels — statements a client filed with a topic.

The bias: A LABELLED STATEMENT IS FILED WHERE THE CLIENT SAID, AND NEVER
ON SOMEONE ELSE. Each topic lands in one graph category; someone else's
fact carries their name in its key so it can never replace the user's
own; an unknown topic or a nonsense lifetime is ignored rather than
guessed at.

And A FACT IS THE THING, NOT ADVICE ABOUT IT: the rule-based trim cuts a
"so a later question should ..." clause and a "visible in their inbox"
aside, and keeps names, numbers and a short detail in brackets.
"""

from smritikosh.memory.labels import fact_key, labelled_fact, lifetime_days, trim_statement


class TestLabelledFact:
    def test_home_files_under_location_with_its_topic_as_key(self):
        fact = labelled_fact({"topic": "home"}, "Lives in Cary, NC (Eastern time).")
        assert fact == {"category": "location", "key": "home",
                        "value": "Lives in Cary, NC (Eastern time).", "confidence": 1.0}

    def test_someone_elses_fact_carries_their_name_in_the_key(self):
        fact = labelled_fact({"topic": "contact", "about": "Naina", "key": "email"}, "x")
        assert fact["key"] == "naina.email"

    def test_their_daughter_never_lands_on_the_users_own_key(self):
        assert fact_key("dob", "Akansha") != fact_key("dob")

    def test_other_or_no_topic_is_left_for_consolidation(self):
        assert labelled_fact({"topic": "other"}, "x") is None
        assert labelled_fact({}, "x") is None
        assert labelled_fact(None, "x") is None


class TestTrimStatement:
    def test_advice_to_the_agent_and_where_it_was_seen_are_cut(self):
        said = ("Has Acme Mutual as their auto insurer (policy ending 4K, visible in their "
                "mail inbox), so a later \"who handles my car insurance\" or \"check "
                "my auto renewal\" question should default to Acme Mutual rather than "
                "asking which carrier.")
        assert trim_statement(said) == "Has Acme Mutual as their auto insurer (policy ending 4K)."

    def test_a_plain_fact_is_left_as_it_is(self):
        assert trim_statement("Lives in Cary, NC (Eastern time).") == \
            "Lives in Cary, NC (Eastern time)."

    def test_a_dash_or_semicolon_advice_clause_goes_too(self):
        assert trim_statement("Prefers metric units - so answers should use km") == \
            "Prefers metric units."
        assert trim_statement("Banks with Truist; which means bills go there.") == \
            "Banks with Truist."

    def test_a_bracket_that_only_says_where_it_was_seen_is_dropped(self):
        assert trim_statement("Banks with Truist (seen in their email)") == "Banks with Truist."

    def test_a_paragraph_keeps_its_first_sentence(self):
        said = "Works at Acme as a data engineer. " + "They mentioned it often. " * 8
        assert trim_statement(said) == "Works at Acme as a data engineer."

    def test_a_labelled_fact_is_trimmed_before_it_is_filed(self):
        fact = labelled_fact({"topic": "account", "key": "auto_insurer"},
                             "Has Acme Mutual as their auto insurer, so ask nothing.")
        assert fact["value"] == "Has Acme Mutual as their auto insurer."


class TestLifetime:
    def test_whole_days_are_read(self):
        assert lifetime_days({"lifetime_days": 60}) == 60
        assert lifetime_days({"lifetime_days": "365"}) == 365

    def test_none_zero_or_nonsense_is_no_lifetime(self):
        assert lifetime_days({}) is None
        assert lifetime_days({"lifetime_days": 0}) is None
        assert lifetime_days({"lifetime_days": "soon"}) is None
