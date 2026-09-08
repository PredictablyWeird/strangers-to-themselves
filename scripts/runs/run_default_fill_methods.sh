#!/usr/bin/env bash
# Fill the missing prediction methods for the default pool:
#   reward_hacking: behavioral_sampling, cot_flip, cross_model_mean, list_experiment, llm_prediction
#   tau2_policy:    informed_oracle
# (oracle_xmm / oracle_xmm_learned deliberately excluded; tau2 oracle_pairwise excluded per request.)
# One background lane per model. Idempotent (skips existing). Tuning + evaluate happen separately.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
LOGS=logs/default_fill; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="

    echo "=== [$M] reward_hacking: 5 missing methods ==="
    .venv/bin/bp-benchmark --phase generate --evals reward_hacking --models "$M" \
      --methods behavioral_sampling,cot_flip,cross_model_mean,list_experiment,llm_prediction \
      --concurrency 6 || { echo "[$M] FAIL rh methods"; touch "$LOGS/$M.failed"; }

    echo "=== [$M] tau2_policy: informed_oracle ==="
    .venv/bin/bp-benchmark --phase generate --evals tau2_policy --models "$M" \
      --methods informed_oracle --concurrency 6 || { echo "[$M] FAIL tau2 informed_oracle"; touch "$LOGS/$M.failed"; }

    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    [ -f "$LOGS/$M.failed" ] || touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ FILL DONE $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M ($LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
