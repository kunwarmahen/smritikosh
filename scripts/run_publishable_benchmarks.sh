#!/usr/bin/env bash
# Publishable S3 benchmark runs — LoCoMo (full) + LongMemEval.
#
# Produces numbers methodologically comparable with published Mem0 / Zep /
# Memobase results: GPT-4o-class answer model, separate judge model,
# chunk-turns 1 (product-native granularity). See evals/benchmarks/RESULTS.md
# for method, competitor numbers, and the publication checklist.
#
# Required environment:
#   SMRITIKOSH_API_KEY   admin API key (benchmarks create synthetic users)
#   OPENAI_API_KEY       for the default answer/judge models (openai:*)
#
# Server prerequisites (verify before starting — the run is hours + real money):
#   * RECONSOLIDATION_ON_RECALL=0 (or Redis + taskworker) — keeps /context free
#     of write-back LLM traffic; competitors' harnesses don't do write-backs.
#   * A GPT-4o-class server-side LLM + embedding model (.env LLM_PROVIDER /
#     EMBEDDING_PROVIDER) — the smoke run showed local-model extraction and
#     embeddings dominate the miss rate; a publishable number should measure
#     the product on its recommended cloud config.
#
# Usage:
#   scripts/run_publishable_benchmarks.sh                # LoCoMo, then LongMemEval oracle
#   BENCHMARKS="locomo" scripts/run_publishable_benchmarks.sh
#   LME_VARIANT=s scripts/run_publishable_benchmarks.sh  # full 115K-token haystacks ($$$)
#   ANSWER_MODEL=claude:claude-sonnet-4-6 scripts/run_publishable_benchmarks.sh
#
# Rough cost/time (OpenAI pricing, ingest-concurrency 8):
#   LoCoMo full        ~5,900 events + 1,540×2 QA calls   ≈ $20–40, 3–6 h
#   LongMemEval oracle evidence-only haystacks, 500 QA    ≈ $10–20, 1–3 h
#   LongMemEval s      ~57M haystack tokens through extraction — estimate from
#                      a 10-user partial run before committing (≈ $100+, 12 h+)

set -euo pipefail
cd "$(dirname "$0")/.."

ANSWER_MODEL="${ANSWER_MODEL:-openai:gpt-4o}"
JUDGE_MODEL="${JUDGE_MODEL:-openai:gpt-4o-mini}"
BENCHMARKS="${BENCHMARKS:-locomo longmemeval}"
LME_VARIANT="${LME_VARIANT:-oracle}"
INGEST_CONCURRENCY="${INGEST_CONCURRENCY:-8}"
QA_CONCURRENCY="${QA_CONCURRENCY:-8}"
BASE_URL="${SMRITIKOSH_BASE_URL:-http://localhost:8080}"
RESULTS_DIR="evals/benchmarks/results"
STAMP="$(date +%Y%m%d-%H%M)"

# ── Preflight ─────────────────────────────────────────────────────────────────
[[ -n "${SMRITIKOSH_API_KEY:-}" ]] || { echo "FATAL: SMRITIKOSH_API_KEY not set (admin key required)"; exit 1; }
case "$ANSWER_MODEL $JUDGE_MODEL" in
  *openai:*) [[ -n "${OPENAI_API_KEY:-}" ]] || { echo "FATAL: OPENAI_API_KEY not set"; exit 1; } ;;
esac
case "$ANSWER_MODEL $JUDGE_MODEL" in
  *claude:*) [[ -n "${ANTHROPIC_API_KEY:-}" ]] || { echo "FATAL: ANTHROPIC_API_KEY not set"; exit 1; } ;;
esac
curl -sf "$BASE_URL/health" >/dev/null || { echo "FATAL: no healthy server at $BASE_URL"; exit 1; }

echo "server=$BASE_URL answer=$ANSWER_MODEL judge=$JUDGE_MODEL"
echo "REMINDER: server must run with RECONSOLIDATION_ON_RECALL=0 and its"
echo "recommended cloud LLM/embedding config — this script cannot verify that."
echo

mkdir -p "$RESULTS_DIR"

run() {
  local bench="$1"; shift
  local out="$RESULTS_DIR/${bench}-${STAMP}.json"
  echo "── $bench → $out"
  python -m evals.benchmarks "$bench" \
    --answer-model "$ANSWER_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --chunk-turns 1 \
    --ingest-concurrency "$INGEST_CONCURRENCY" \
    --concurrency "$QA_CONCURRENCY" \
    --json "$out" \
    "$@"
  echo
}

for bench in $BENCHMARKS; do
  case "$bench" in
    locomo)      run locomo ;;
    longmemeval) run longmemeval --variant "$LME_VARIANT" ;;
    *) echo "unknown benchmark: $bench"; exit 1 ;;
  esac
done

echo "Done. Add a row per run to evals/benchmarks/RESULTS.md (keep the JSON reports)."
