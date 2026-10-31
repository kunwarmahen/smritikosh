"""
Tests for labels — statements a client filed with a topic.

The bias: A LABELLED STATEMENT IS FILED WHERE THE CLIENT SAID, AND NEVER
ON SOMEONE ELSE. Each topic lands in one graph category; someone else's
fact carries their name in its key so it can never replace the user's
own; an unknown topic or a nonsense lifetime is ignored rather than
guessed at.
"""

from smritikosh.memory.labels import fact_key, labelled_fact, lifetime_days


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


class TestLifetime:
    def test_whole_days_are_read(self):
        assert lifetime_days({"lifetime_days": 60}) == 60
        assert lifetime_days({"lifetime_days": "365"}) == 365

    def test_none_zero_or_nonsense_is_no_lifetime(self):
        assert lifetime_days({}) is None
        assert lifetime_days({"lifetime_days": 0}) is None
        assert lifetime_days({"lifetime_days": "soon"}) is None
