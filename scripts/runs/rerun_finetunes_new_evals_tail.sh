#!/usr/bin/env bash
# One-off: rerun ONLY the two endpoint phases that failed in run_finetunes_new_evals.sh
# (shift-arg bug, fixed there) + the offline cross_model_mean tail. Does NOT touch the four
# completed models (gemma pair / llama-tg / qwen72 base) — no re-measurement, no re-elicitation.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/finetune_new_evals
source <(sed -n '/^teardown_all()/,/^}/p;/^serve_check()/,/^}/p;/^endpoint_phase()/,/^}/p' scripts/runs/run_finetunes_new_evals.sh)
METHODS=self_report,value,pairwise,informed_oracle
trap teardown_all EXIT

endpoint_phase logs/selfpred_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b-selfpred-v1 qwen72_v1 \
  tau2_transfer 0 ""
endpoint_phase logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm llama-3.3-70b-selfpred-v2fix-tg v2fix \
  tau2_transfer 0 ""

.venv/bin/bp-benchmark --phase generate --evals tau2_transfer \
  --models qwen2.5-72b-selfpred-v1,llama-3.3-70b-selfpred-v2fix-tg --methods cross_model_mean \
  --concurrency 4 --fresh > "$LOGS/xmm_tail.log" 2>&1 || echo "WARN: xmm tail failed"

echo "############ TAIL RERUN FINISHED ############"
for M in qwen2.5-72b-selfpred-v1 llama-3.3-70b-selfpred-v2fix-tg; do
  echo "  $M: tau2_transfer $(ls results/tau2_transfer/$M/predictions/ 2>/dev/null | wc -l) prediction files"
done
