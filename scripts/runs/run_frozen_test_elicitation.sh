#!/usr/bin/env bash
# FROZEN TEST, elicitation phase (pre-spec docs/test-prespecification.md, FROZEN 2026-08-16).
# Tuned-only settings, test-split TOP-UP (seam eb6a124: dev predictions stay byte-identical;
# only missing test conditions are elicited and merged). 12 default-pool models × 8 active
# evals × DEFAULT_METHODS. NO evaluation here — the single test bp-evaluate runs separately
# after every lane verifies complete.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOGS=logs/frozen_test; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b
        llama-4-maverick qwen3.7-plus-low claude-sonnet-5-off claude-sonnet-5-low
        gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  local EXTRA=()
  if [[ "$M" == gemini-* ]]; then
    EXTRA=(--method-config behavioral_sampling:grader_model=gpt-5.4-nano-low
           --method-config informed_sampling:grader_model=gpt-5.4-nano-low)
  fi
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --include-test --tuned-only \
      --evals sycophancy_pushback,discrimeval,capability_mmlu,reward_hacking,tau2_policy,tau2_transfer,propensitybench,mask_subdomain_pressure \
      --models "$M" --concurrency 6 "${EXTRA[@]}" \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ FROZEN-TEST ELICITATION FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
