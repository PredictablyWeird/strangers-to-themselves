#!/usr/bin/env bash
# Frontier coverage for the two newly-active evals (2026-08-14):
#   tau2_transfer          — targets extract FREE from the stored tau2_policy sims
#                            (Tau2PolicySpec("transfer") reads results/tau2_policy/_harness/).
#   mask_subdomain_pressure — needs the shared per-proposition MASK measurement first
#                            (bp-sweep --eval mask; the pooled grain re-aggregates it),
#                            then bp-targets at the pressure grain.
# Afterwards each lane elicits the full default method set (15 methods incl. the sampling
# pair under promoted defaults; full ask grids, so the stage-7 re-tune can't orphan files).
# Grader: project default (flash-lite); no gemini subject in this set, so no override.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/frontier_new_evals; mkdir -p "$LOGS"

MODELS=(claude-sonnet-5-off claude-sonnet-5-low gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="

    echo "=== [$M] tau2_transfer targets (from stored sims; no calls) ==="
    .venv/bin/bp-targets --eval tau2_transfer --model "$M" \
      || { echo "[$M] FAIL tau2_transfer targets"; touch "$LOGS/$M.failed"; exit 1; }

    echo "=== [$M] MASK shared measurement (pressured continuations + beliefs) ==="
    .venv/bin/bp-sweep --eval mask --model "$M" \
      || { echo "[$M] FAIL mask sweep"; touch "$LOGS/$M.failed"; exit 1; }
    .venv/bin/bp-targets --eval mask_subdomain_pressure --model "$M" \
      || { echo "[$M] FAIL mask_subdomain_pressure targets"; touch "$LOGS/$M.failed"; exit 1; }

    echo "=== [$M] elicitation: default methods on both evals ==="
    .venv/bin/bp-benchmark --phase generate --evals tau2_transfer,mask_subdomain_pressure \
      --models "$M" --concurrency 6 \
      || { echo "[$M] FAIL elicitation"; touch "$LOGS/$M.failed"; exit 1; }

    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ FRONTIER NEW-EVAL COVERAGE FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
