#!/usr/bin/env bash
# FROZEN TEST: agentic_misalignment held-out eval (pre-spec C10, docs/test-prespecification.md).
# Measurement at the n=20 default (--epochs 20 is the eval's default since 2026-08-15) for
# all 12 pool models, then targets, then the donor-tuned stateless suite (--tuned-only reads
# the committed donor_modal_best settings; AM is all-test so --include-test elicits its
# conditions). NEVER evaluated here; scored only with the explicit final test evaluation.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOGS=logs/frozen_test_am; mkdir -p "$LOGS"

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
    echo "=== [$M] measure (n=20 epochs default) ==="
    .venv/bin/bp-sweep --eval agentic_misalignment --model "$M" \
      || { echo "[$M] FAIL sweep"; touch "$LOGS/$M.failed"; exit 1; }
    .venv/bin/bp-targets --eval agentic_misalignment --model "$M" \
      || { echo "[$M] FAIL targets"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] donor-tuned elicitation (all-test) ==="
    .venv/bin/bp-benchmark --phase generate --evals agentic_misalignment --models "$M" \
      --include-test --tuned-only --concurrency 6 "${EXTRA[@]}" \
      || { echo "[$M] FAIL elicitation"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ AM FROZEN-TEST FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
