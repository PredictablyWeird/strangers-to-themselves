#!/usr/bin/env bash
# Stage 1 of the final-results dev runs (2026-08-11): regenerate ALL small-pool sampling
# predictions under the promoted defaults — scripted generation, informed_sampling with
# reuse-allowed n_exemplars=10, k=25 x r=2, generator gpt-5.5-off, grader gemini-flash-lite
# (gpt-5.4-nano-low when the subject is gemini, so no model grades its own outputs).
#
# The stale prediction files were MOVED to results/_backup/sampling_pre_final_20260811/
# beforehand, so this runs WITHOUT --fresh: completed files are skipped, interrupted runs
# resume from their checkpoints — safe to re-run until every lane reports DONE.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# MMLU (sycophancy/capability oracle items) is fully cached; offline mode stops load_dataset
# phoning home — 18 parallel lanes hit the anonymous HF API 429 limit (2026-08-15).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/sampling_reasoningfix_small; mkdir -p "$LOGS"

EVALS=sycophancy_pushback,discrimeval,capability_mmlu,reward_hacking,tau2_policy,tau2_transfer,propensitybench,mask_subdomain_pressure
MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  local EXTRA=()
  if [[ "$M" == gemini-* ]]; then
    EXTRA=(
           --method-config informed_sampling:grader_model=gpt-5.4-nano-low)
  fi
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals "$EVALS" --models "$M" \
      --methods informed_sampling --concurrency 6 "${EXTRA[@]}" \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ SAMPLING REGEN (SMALL POOL) FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
