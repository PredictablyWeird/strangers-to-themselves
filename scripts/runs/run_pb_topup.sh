#!/usr/bin/env bash
# Equalize PropensityBench DEV coverage across the 12 default-pool models (task: complete,
# comparable PB results — never the full benchmark, only the canonical subsample's dev union;
# cyber-security stays frozen test). Per model: scripts/pb_topup.py runs ONLY the missing dev
# task-scenarios through the harness and merges the new conditions into targets.json
# (backups in results/_backup/pb_targets_pre_topup/).
#
# Small pool first (3-12 scenarios each), then the frontier six (12 each) — an interrupt after
# phase 1 loses nothing. Idempotent: a re-run recomputes missing-vs-targets, so completed
# models no-op.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/pb_topup; mkdir -p "$LOGS"

SMALL=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)
FRONTIER=(claude-sonnet-5-off claude-sonnet-5-low gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/python scripts/pb_topup.py --model "$M" \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

echo "=== phase 1: small pool ==="
for M in "${SMALL[@]}"; do one_model "$M" & done
wait
echo "=== phase 2: frontier ==="
for M in "${FRONTIER[@]}"; do one_model "$M" & done
wait

echo "############ PB TOP-UP FINISHED $(date +%H:%M:%S) ############"
for M in "${SMALL[@]}" "${FRONTIER[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
