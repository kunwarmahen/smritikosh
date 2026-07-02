"""
Public memory benchmarks (S3): LoCoMo and LongMemEval against Smritikosh.

Runs the standard agent-memory yardsticks end-to-end through a live
Smritikosh server (ingest → retrieve → answer → LLM judge) so scores are
comparable with published Mem0 / Zep / Letta numbers.

Run:  python -m evals.benchmarks --help     (needs a running server + admin key)
Docs: evals/benchmarks/RESULTS.md
"""
