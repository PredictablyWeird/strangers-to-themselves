#!/usr/bin/env bash
# Fill the runnable gaps found in the frontier report (2026-08-03):
#   1) tau2 oracle_pairwise for the 4 original pool models (never generated there)
#   2) PB value-concrete for all scoreable-PB models (PB tuning.json repointed from the
#      no-longer-generatable value-context to the pinned value-concrete)
#   3) discrim behavioral_sampling for llama-4-maverick (file never written; no API calls)
#   4) llama-3.3-70b discrimeval re-measure at the current 583-cell/99-contrast grain
#      (old 374-cell grid; sweep resumes -> only missing cells are measured), then regenerate
#      ALL its discrim predictions fresh at tuned settings + the few_shot pair.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}   # oracle_pairwise needs the harness clone
BP=.venv/bin
LOGS=logs/fill_gaps; mkdir -p "$LOGS"
run() { echo "+ $*"; "$@" || echo "ERROR ($?) in: $*"; }

echo "=== 1) tau2 oracle_pairwise: 4 original pool models ==="
run $BP/bp-benchmark --phase generate --evals tau2_policy \
    --models gemini-3.1-flash-lite-low,gpt-5.4-nano-low,llama-3.3-70b,qwen3.7-plus-low \
    --methods oracle_pairwise --concurrency 6

echo "=== 2) PB value-concrete: all scoreable-PB models ==="
run $BP/bp-benchmark --phase generate --evals propensitybench \
    --models deepseek-v4-flash-low,gemini-3.1-flash-lite-low,gpt-5.4-nano-low,llama-4-maverick,qwen3.7-plus-low,deepseek-v4-pro-low,deepseek-v4-pro-off,gpt-5.5-low \
    --methods value --concurrency 6

echo "=== 3) discrim behavioral_sampling: llama-4-maverick (no API calls) ==="
run $BP/bp-benchmark --phase generate --evals discrimeval --models llama-4-maverick \
    --methods behavioral_sampling

echo "=== 4) llama-3.3-70b discrimeval: re-measure at current grain + fresh predictions ==="
run $BP/bp-sweep --eval discrimeval --model llama-3.3-70b --concurrency 8
run $BP/bp-targets --eval discrimeval --model llama-3.3-70b
run $BP/bp-benchmark --phase generate --tuned-only --evals discrimeval \
    --models llama-3.3-70b --concurrency 6 --fresh
run $BP/bp-benchmark --phase generate --evals discrimeval --models llama-3.3-70b \
    --methods few_shot,few_shot_other --concurrency 6 --fresh

echo "=== done $(date +%H:%M:%S) ==="
