#!/usr/bin/env bash
# Complete the frontier settings for the merged pool (approved 2026-08-07, list_experiment
# excluded): generic_report + generic_oracle on all six evals, plus the small gaps
# (few_shot/few_shot_other on reward_hacking, few_shot on PB for sonnet, pairwise on tau2,
# value on PB for sonnet-off/low + gpt-off). One background lane per (model, reasoning)
# setting; idempotent (bp-benchmark skips existing). Pool-derived methods (report_mean,
# oracle_report_mean, merged-pool cross_model_mean) are recomputed locally afterwards, NOT
# here. Estimated total ~$230 central, ~10h at the global ~50 calls/min elicitation limit.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
LOGS=logs/frontier_complete; mkdir -p "$LOGS"

MODELS=(claude-sonnet-5-off claude-sonnet-5-low gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="

    echo "=== [$M] generic_report + generic_oracle, all evals ==="
    .venv/bin/bp-benchmark --phase generate \
      --evals discrimeval,propensitybench,capability_mmlu,sycophancy_pushback,reward_hacking,tau2_policy \
      --models "$M" --methods generic_report,generic_oracle --concurrency 8 \
      || { echo "[$M] FAIL generic tier"; touch "$LOGS/$M.failed"; }

    echo "=== [$M] few_shot/few_shot_other on reward_hacking ==="
    .venv/bin/bp-benchmark --phase generate --evals reward_hacking --models "$M" \
      --methods few_shot,few_shot_other --concurrency 6 \
      || { echo "[$M] FAIL rh few_shot"; touch "$LOGS/$M.failed"; }

    if [[ "$M" == claude-sonnet-5-* ]]; then
      echo "=== [$M] few_shot on propensitybench ==="
      .venv/bin/bp-benchmark --phase generate --evals propensitybench --models "$M" \
        --methods few_shot --concurrency 6 || echo "[$M] WARN pb few_shot"
    fi

    echo "=== [$M] pairwise on tau2_policy ==="
    .venv/bin/bp-benchmark --phase generate --evals tau2_policy --models "$M" \
      --methods pairwise --concurrency 6 || echo "[$M] WARN tau2 pairwise"

    if [[ "$M" == claude-sonnet-5-off || "$M" == claude-sonnet-5-low || "$M" == gpt-5.5-off ]]; then
      echo "=== [$M] value on propensitybench ==="
      .venv/bin/bp-benchmark --phase generate --evals propensitybench --models "$M" \
        --methods value --concurrency 6 || echo "[$M] WARN pb value"
    fi

    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    [ -f "$LOGS/$M.failed" ] || touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ FRONTIER COMPLETE $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M ($LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
