"""
Cognitive agent layer (item E4, FUTURE.md).

The memory substrate below this package stores, consolidates, and retrieves.
This package closes the loop above it:

    prediction.py  — Predict-Observe-Learn loop (#7): guess what a query will
                     surface before retrieval, score the guess after, feed the
                     delta into importance scores.
    decision.py    — Personal Decision Agent (#1): memory-grounded decision
                     recommendations with cited evidence.
    reflection.py  — Reflection cycles (#9): periodic drift/contradiction
                     detection between stated identity and actual behaviour.

Meta-cognition complexity routing (#8) lives in
retrieval/intent_classifier.py (ComplexityTier) and retrieval/context_builder.py.
"""
