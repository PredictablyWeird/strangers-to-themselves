#!/usr/bin/env bash
# Complete the DEFAULT pool: add reward_hacking (measure + predict) and oracle_pairwise across the
# forced-choice evals, for all 6 default-pool models. tau2_policy is already measured+predicted on
# disk; it only needs folding into the report (done separately via bp-evaluate).
#
# Each model runs in its own background lane (different OpenRouter/provider buckets), logging to
# logs/default_complete/<model>.log and touching <model>.done on success. reward_hacking has
# built-in backoff, so cross-model 429s self-heal. Idempotent: bp-benchmark/bp-sweep skip existing
# work, so re-running resumes.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/default_complete; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="

    echo "=== [$M] measure reward_hacking behavior ==="
    .venv/bin/bp-sweep --eval reward_hacking --model "$M" --concurrency 8 || { echo "[$M] FAIL sweep"; touch "$LOGS/$M.failed"; exit 1; }
    .venv/bin/bp-targets --eval reward_hacking --model "$M" || { echo "[$M] FAIL targets"; touch "$LOGS/$M.failed"; exit 1; }

    echo "=== [$M] oracle_pairwise on syc/disc/mmlu ==="
    .venv/bin/bp-benchmark --phase generate \
      --evals sycophancy_pushback,discrimeval,capability_mmlu \
      --models "$M" --methods oracle_pairwise --concurrency 6 || echo "[$M] WARN oracle_pairwise"

    echo "=== [$M] reward_hacking predictions ==="
    .venv/bin/bp-benchmark --phase generate --evals reward_hacking --models "$M" \
      --methods self_report,value,pairwise,informed_oracle,oracle_pairwise --concurrency 6 \
      || echo "[$M] WARN reward_hacking predictions"

    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ ALL DEFAULT MODELS PROCESSED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
