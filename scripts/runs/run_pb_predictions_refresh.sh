#!/usr/bin/env bash
# PB completeness, prediction side (follows scripts/runs/run_pb_topup.sh): the equalized targets
# added 3-12 dev conditions per model, and prediction files are skipped at FILE level, so every
# ask-method file on propensitybench must be regenerated --fresh. The target-set change also
# ripples into fit splits (few_shot / llm_prediction training blocks) and donor means, so a full
# regeneration — all grid settings, not --tuned-only — is the consistent move, and the final
# re-tune then rescores on uniform coverage.
#
# Phase A: per-model elicited methods (parallel lanes). Phase B: the pool-derived predictors
# (no API calls) AFTER every lane is done, so donor reads never see half-regenerated files.
# Sampling methods are deliberately absent — the sampling regeneration stage covers them.
# cot_flip is retired (2026-08-11) and list_experiment stays the published null.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/pb_pred_refresh; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b
        llama-4-maverick qwen3.7-plus-low claude-sonnet-5-off claude-sonnet-5-low
        gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low)
ELICITED=self_report,generic_report,value,pairwise,informed_oracle,generic_oracle,oracle_pairwise,few_shot,few_shot_other,llm_prediction
POOLED=cross_model_mean,report_mean,oracle_report_mean

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals propensitybench --models "$M" \
      --methods "$ELICITED" --fresh --concurrency 6 \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

echo "=== phase A: elicited methods, all models ==="
for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "=== phase B: pool-derived predictors ==="
.venv/bin/bp-benchmark --phase generate --evals propensitybench \
  --models "$(IFS=,; echo "${MODELS[*]}")" --methods "$POOLED" --fresh --concurrency 4 \
  > "$LOGS/pooled.log" 2>&1 || echo "WARN: pooled predictors failed (see $LOGS/pooled.log)"

echo "############ PB PREDICTION REFRESH FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
