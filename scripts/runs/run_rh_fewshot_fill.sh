#!/usr/bin/env bash
# Close the history-tier coverage gap: reward_hacking has no few_shot /
# few_shot_other predictions, so that tier is scored on 26 of 32 cells. One background lane
# per default-pool model. Idempotent (bp-benchmark skips existing). Tuning + evaluate happen
# separately (bp-tune with the FULL method list; back up tuning.json first).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/rh_fewshot_fill; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals reward_hacking --models "$M" \
      --methods few_shot,few_shot_other --concurrency 6 \
      || { echo "[$M] FAIL few_shot methods"; touch "$LOGS/$M.failed"; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    [ -f "$LOGS/$M.failed" ] || touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ RH FEW-SHOT FILL DONE $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M ($LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
