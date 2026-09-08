#!/usr/bin/env bash
# Drive every Together-endpoint model through measure+predict, ONE ENDPOINT AT A TIME. Each model's
# endpoint is torn down (via run_endpoint_model.sh's EXIT trap) before the next one is deployed, so
# we never pay for two idle GPUs at once.
#
# gemma-4-31b-selfpred-v1 runs first in SKIP_MEASURE mode: its behavior is already measured, but its
# tau2_policy predictions were skipped (its tau2 targets did not exist yet when its endpoint was up,
# because the all-zero targets from the broken litellm tool path had just been deleted).
set -uo pipefail
cd "$(dirname "$0")/../.."
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

run() {  # run <shortcut> <log-dir> <hardware> [deploy-model]
  echo "############ $1 :: $(date +%H:%M:%S) ############"
  bash scripts/runs/run_endpoint_model.sh "$@" 2>&1 | sed "s/^/[$1] /"
}

# llama-3.3-70b-selfpred-v2-tg is intentionally OMITTED — its checkpoint was trained on a defective
# mix (docs/selfpred-fixed-point.md); it must be re-finetuned before it is evaluated.
SKIP_MEASURE=1 run gemma-4-31b-selfpred-v1 logs/selfpred_gemma            2x_nvidia_h100_80gb_sxm
run llama-3.3-70b-intro30k-tg   logs/introspection_finetune_llama30k      4x_nvidia_h100_80gb_sxm
run qwen2.5-72b                 logs/selfpred_corpus_qwen72               4x_nvidia_h100_80gb_sxm  Qwen/Qwen2.5-72B-Instruct
run qwen2.5-72b-selfpred-v1     logs/selfpred_qwen72                      4x_nvidia_h100_80gb_sxm

echo "############ ALL ENDPOINT MODELS DONE :: $(date +%H:%M:%S) ############"
.venv/bin/python - <<'EOF'
# Final safety net: nothing of ours may still be live and billing.
import os, json
from together import Together
c = Together(api_key=os.environ["TOGETHER_API_KEY"])
live = [e.id for e in (c.endpoints.list(type="dedicated", mine=True).data or [])]
print(f"REMAINING DEDICATED ENDPOINTS: {live or 'none'}")
EOF
