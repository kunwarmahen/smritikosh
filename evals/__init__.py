"""
Extraction-quality eval harness (S2).

Golden transcripts + a precision/recall scorer for the LLM fact-extraction
pipeline. Behavioral tests guard code paths; this guards *LLM output quality* —
a prompt tweak or model swap that silently degrades extraction shows up here
as a precision/recall regression.

Run:  python -m evals --help          (requires a configured LLM provider)
Docs: evals/README.md
"""
