#!/usr/bin/env bash
# MASK (shared sweep -> mask_subdomain_pressure) for the remaining finetune-pair models
# (2026-08-15: include MASK for all finetunes; gemma pair already done). Serverless
# llama-tg first, then serial endpoint sessions. Ask channels at full grids + offline xmm.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/finetunes_mask; mkdir -p "$LOGS"
METHODS=self_report,value,pairwise,informed_oracle
source <(sed -n '/^teardown_all()/,/^}/p;/^serve_check()/,/^}/p;/^endpoint_phase()/,/^}/p' scripts/runs/run_finetunes_new_evals.sh)
LOGS=logs/finetunes_mask   # endpoint_phase logs here, not the sourced default
trap teardown_all EXIT

echo "=== serverless llama-tg: MASK sweep + targets + elicitation ==="
{
  .venv/bin/bp-sweep --eval mask --model llama-3.3-70b-tg \
    && .venv/bin/bp-targets --eval mask_subdomain_pressure --model llama-3.3-70b-tg \
    && .venv/bin/bp-benchmark --phase generate --evals mask_subdomain_pressure \
         --models llama-3.3-70b-tg --methods "$METHODS" --concurrency 6
} > "$LOGS/llama_tg.log" 2>&1 || echo "WARN: llama-tg mask phase failed"

endpoint_phase logs/selfpred_corpus_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b qwen72_base \
  mask_subdomain_pressure 1 "" --deploy-model Qwen/Qwen2.5-72B-Instruct
endpoint_phase logs/selfpred_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b-selfpred-v1 qwen72_v1 \
  mask_subdomain_pressure 1 ""
endpoint_phase logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm llama-3.3-70b-selfpred-v2fix-tg v2fix \
  mask_subdomain_pressure 1 ""

echo "=== offline cross_model_mean ==="
.venv/bin/bp-benchmark --phase generate --evals mask_subdomain_pressure \
  --models llama-3.3-70b-tg,qwen2.5-72b,qwen2.5-72b-selfpred-v1,llama-3.3-70b-selfpred-v2fix-tg \
  --methods cross_model_mean --concurrency 4 > "$LOGS/xmm.log" 2>&1 || echo "WARN: xmm failed"

echo "############ FINETUNES MASK FINISHED $(date +%H:%M:%S) ############"
for M in llama-3.3-70b-tg qwen2.5-72b qwen2.5-72b-selfpred-v1 llama-3.3-70b-selfpred-v2fix-tg; do
  echo "  $M: mask_subdomain_pressure $(ls results/mask_subdomain_pressure/$M/predictions/ 2>/dev/null | wc -l) prediction files"
done
