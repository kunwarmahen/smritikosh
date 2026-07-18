"""
Cognitive agent layer (item E4, FUTURE.md).

The memory substrate below this package stores, consolidates, and retrieves.
This package closes the loop above it:

    prediction.py  — Predict-Observe-Learn loop (#7): guess what a query will
                     surface before retrieval, score the guess after, feed the
                     delta into importance scores.
    decision.py    — Personal Decision Agent (#1): memory-grounded decision
                     recommendations with cited evidence.
    council.py     — Deliberation Council (#4): four specialist perspectives
                     (risk, values, pattern, devil's advocate) + a judge, for
                     high-stakes decisions.
    meeting_prep.py — Meeting Prep (#3): pre-meeting briefs from attendee
                     memory pools; post-meeting debrief re-enters the
                     encoding pipeline (memory in → action → memory out).
    reflection.py  — Reflection cycles (#9): periodic drift/contradiction
                     detection between stated identity and actual behaviour.
    lifeos.py      — Proactive Life OS (#5): bundles fresh reflection
                     insights into per-user nudge digests, delivered via the
                     in-app feed and an optional webhook. LLM-free.

Meta-cognition complexity routing (#8) lives in
retrieval/intent_classifier.py (ComplexityTier) and retrieval/context_builder.py.
"""
