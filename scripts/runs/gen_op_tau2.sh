#!/usr/bin/env bash
# Generate oracle_pairwise on tau2_policy for the LoRA finetunes that need serving. One endpoint at a
# time, torn down on EXIT (bills per minute). tau2 oracle items need $TAU2_REPO.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

one() {  # one <shortcut> <log-dir> <hardware> [deploy-model]
  local SHORTCUT=$1 OUT=$2 HW=$3 DEPLOY=${4:-}
  echo "############ $SHORTCUT :: $(date +%H:%M:%S) ############"
  teardown() { .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || echo "!!! TEARDOWN FAILED $OUT !!!"; }
  trap teardown RETURN
  if [[ -n "$DEPLOY" ]]; then
    .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" --hardware "$HW" \
      --inactive-timeout 30 --deploy-model "$DEPLOY" || { teardown; return 1; }
  else
    .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" --hardware "$HW" \
      --inactive-timeout 30 || { teardown; return 1; }
  fi
  .venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$OUT" || { teardown; return 1; }
  .venv/bin/bp-benchmark --phase generate --evals tau2_policy --models "$SHORTCUT" \
    --methods oracle_pairwise --concurrency 6 > "$LOGS/op_tau2_$SHORTCUT.log" 2>&1 \
    || echo "WARN: oracle_pairwise/tau2 failed for $SHORTCUT"
  teardown; trap - RETURN
  echo "=== done $SHORTCUT ==="
}

one gemma-4-31b-selfpred-v1        logs/selfpred_gemma           2x_nvidia_h100_80gb_sxm
one qwen2.5-72b                    logs/selfpred_corpus_qwen72   4x_nvidia_h100_80gb_sxm  Qwen/Qwen2.5-72B-Instruct
one qwen2.5-72b-selfpred-v1        logs/selfpred_qwen72          4x_nvidia_h100_80gb_sxm
one llama-3.3-70b-selfpred-v2fix-tg logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm

echo "############ ALL DONE :: $(date +%H:%M:%S) ############"
.venv/bin/python -c "import os;from together import Together;c=Together(api_key=os.environ['TOGETHER_API_KEY']);print('REMAINING ENDPOINTS:', [e.id for e in (c.endpoints.list(type='dedicated',mine=True).data or [])] or 'none')"
