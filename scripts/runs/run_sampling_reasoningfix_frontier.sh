#!/usr/bin/env bash
# Stage: frontier sampling regeneration (follows run_sampling_regen_small.sh). Same promoted
# defaults; the 6 frontier models on the 6 evals they have targets for (tau2_transfer /
# mask_subdomain_pressure frontier coverage is a separate deferred decision). Stale files are
# moved to the same backup dir first, so the run needs no --fresh and is resumable/idempotent.
# No gemini subject here, so the standard grader applies everywhere.
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
LOGS=logs/sampling_reasoningfix_frontier; mkdir -p "$LOGS"
BACKUP=results/_backup/sampling_pre_final_20260811

EVALS=sycophancy_pushback,discrimeval,capability_mmlu,reward_hacking,tau2_policy,tau2_transfer,propensitybench,mask_subdomain_pressure
MODELS=(claude-sonnet-5-off claude-sonnet-5-low gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)


one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals "$EVALS" --models "$M" \
      --methods informed_sampling --concurrency 6 \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ SAMPLING REGEN (FRONTIER) FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
