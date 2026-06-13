# Extraction-Quality Eval Harness (S2)

The 1,000+ behavioral tests guard *code paths*; nothing guarded **LLM output
quality**. A prompt tweak or model swap could silently degrade fact extraction
and no test would catch it. This harness is the regression suite for the
product's actual value: it runs a golden set of transcripts through the
*production* extraction paths and scores precision/recall against expected facts.

## What it evaluates

| Case kind | Production path exercised |
|---|---|
| `event` | `POST /memory/event` encode pipeline — `_build_extraction_prompt` + the 23-category schema from `smritikosh/memory/hippocampus.py` |
| `session` | `POST /ingest/session` passive extraction — `prepare_transcript` (assistant-turn filtering, sentinel stripping) + `build_delta_prompt` against `existing_facts` |

Because the harness imports the real prompt builders and `LLMAdapter`, any
change to prompts, schema, or model routing shows up here as a score change.

## Running

Needs the same `.env` the server uses (any LiteLLM provider — local Ollama is
free). **Costs tokens on API providers — run manually or nightly, not per-commit.**

```bash
python -m evals                          # full golden set (~52 cases)
python -m evals --list                   # show cases without running
python -m evals --filter contamination   # subset by case-id substring
python -m evals --kind session           # only the delta-extraction cases
python -m evals --judge                  # LLM judge for paraphrase near-misses
python -m evals --json report.json       # machine-readable report
python -m evals --baseline report.json   # ΔP/ΔR/ΔF1 vs a previous run
python -m evals --min-f1 0.75            # exit 1 below threshold (CI gate)
```

Typical use: save a `--json` baseline before a prompt/model change, re-run with
`--baseline` after, and reject the change if F1 regresses.

## Golden set

`golden/event_cases.json` (~32) + `golden/session_cases.json` (~20). Coverage:

- every one of the 23 fact categories at least once
- negatives: questions, hypotheticals, small talk, transient states
- traps: third-party facts, negations, corrections, Hinglish input
- anti-contamination: assistant-turn facts, injected sentinel blocks,
  delta no-repeat, paraphrased-known-fact dedup

Case schema:

```jsonc
{
  "id": "event-diet-vegetarian",
  "kind": "event",                       // or "session"
  "notes": "why this case exists",
  "content": "...",                      // event cases
  "turns": [{"role": "user", "content": "..."}],   // session cases
  "existing_facts": [{"category": "...", "key": "...", "value": "..."}],  // session: delta input
  "expected": [
    {
      "category": "diet",                // or a list of acceptable categories
      "value": "vegetarian",
      "aliases": ["veg", "no meat"],     // alternative phrasings
      "optional": true                   // TP if found, never an FN if missed
    }
  ],
  "forbidden": [ ... ]                   // must NOT be extracted (strict match)
}
```

## Scoring

- A predicted fact matches an expected spec when the **category** is acceptable
  and the **value** matches lexically (normalized equality, containment, or
  token-Jaccard ≥ 0.6) against the value or any alias. Keys are LLM-invented
  labels — reported, never scored.
- Greedy one-to-one matching; leftover predictions are FPs, leftover required
  expectations are FNs. `optional` specs never produce FNs.
- **Forbidden** specs use strict (equality-only) matching so `"not vegetarian"`
  doesn't trip a forbidden `"vegetarian"`. Hits are reported as `violations` —
  these are the anti-contamination guards and should be **zero**.
- `--judge` re-examines same-category FN/FP pairs the lexical matcher rejected
  and upgrades genuine paraphrases to TPs (one extra cheap LLM call per pair).
- Micro-averaged precision/recall/F1 across cases, plus per-category recall.

## Baseline

`baseline-gemma4-e4b.json` — first run, 2026-06-12, `ollama_chat/gemma4:e4b`:
P=0.677 R=0.926 F1=0.783 (judge: R=0.955, F1=0.795). The low precision is real
signal: the local model over-extracts (~30 spurious facts across 52 cases).

## Harness tests

`tests/test_evals_harness.py` covers the matcher, scorer, golden-set integrity,
and runner plumbing with a fake LLM — runs in the normal suite, no tokens.

## Future work

- S3: run LoCoMo / LongMemEval public benchmarks on top of this harness.
- Belief-mining eval (golden fact corpora → expected beliefs) — the
  `belief_miner` job is not yet covered; fact-level `belief` category is.
- Nightly CI job once a hosted runner with an LLM key exists.
