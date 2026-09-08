#!/usr/bin/env bash
# Finetune pairs on the new evals (2026-08-14): ALL six main-pair models get
# tau2_transfer (targets extract free from their stored tau2 sims; ask-channel elicitation),
# and the GEMMA pair additionally gets mask_subdomain_pressure (shared MASK sweep + targets +
# elicitation) — the other pairs' MASK runs wait on the gemma readout.
# Ask channels only (self_report/value/pairwise/informed_oracle, FULL grids — re-tune-proof);
# the selfpred analyses do not use the sampling methods. cross_model_mean added offline at the
# end. Endpoint sessions run SERIALLY (deploy -> serve check -> elicit -> teardown).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/finetune_new_evals; mkdir -p "$LOGS"
METHODS=self_report,value,pairwise,informed_oracle
ALL6=(gemma-4-31b llama-3.3-70b-tg gemma-4-31b-selfpred-v1 qwen2.5-72b qwen2.5-72b-selfpred-v1 llama-3.3-70b-selfpred-v2fix-tg)

teardown_all() {
  for OUT in logs/selfpred_gemma logs/selfpred_corpus_qwen72 logs/selfpred_qwen72 logs/selfpred_corpus_r2_fixed; do
    .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" >/dev/null 2>&1 || true
  done
  echo "=== teardown backstop ran ==="
}
trap teardown_all EXIT

serve_check() {  # serve_check <out-dir> [thinking-off]
  .venv/bin/python - "$1" "${2:-}" <<'PY'
import json, os, sys, time
from openai import OpenAI
rec = json.loads(open(os.path.join(sys.argv[1], "together_endpoint.json")).read())
kw = ({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
      if len(sys.argv) > 2 and sys.argv[2] else {})
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
for _ in range(30):
    try:
        r = c.chat.completions.create(model=rec["endpoint_name"],
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0, max_tokens=5, **kw)
        if (r.choices[0].message.content or "").strip():
            print("serves OK"); sys.exit(0)
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving after 5min")
PY
}

echo "=== phase A0: tau2_transfer targets for all 6 (free, from stored sims) ==="
for M in "${ALL6[@]}"; do
  .venv/bin/bp-targets --eval tau2_transfer --model "$M" \
    > "$LOGS/t2t_targets_$M.log" 2>&1 || echo "WARN: tau2_transfer targets failed for $M"
done

echo "=== phase A1: serverless gemma base — MASK sweep + targets + elicit both evals ==="
{
  .venv/bin/bp-sweep --eval mask --model gemma-4-31b \
    && .venv/bin/bp-targets --eval mask_subdomain_pressure --model gemma-4-31b \
    && .venv/bin/bp-benchmark --phase generate --evals tau2_transfer,mask_subdomain_pressure \
         --models gemma-4-31b --methods "$METHODS" --concurrency 6
} > "$LOGS/gemma_base.log" 2>&1 || echo "WARN: gemma base phase failed (see $LOGS/gemma_base.log)"

echo "=== phase A2: serverless llama-tg — elicit tau2_transfer ==="
.venv/bin/bp-benchmark --phase generate --evals tau2_transfer --models llama-3.3-70b-tg \
  --methods "$METHODS" --concurrency 6 > "$LOGS/llama_tg.log" 2>&1 \
  || echo "WARN: llama-tg elicitation failed"

endpoint_phase() {  # endpoint_phase <out-dir> <hardware> <shortcut> <logname> <evals> <mask:0|1> [thinking-off] [--deploy-model <id>]
  local OUT=$1 HW=$2 SC=$3 LN=$4 EVALS=$5 MASK=$6 THINK=${7:-}; shift 7 || true
  echo "=== deploy $SC on $HW ==="
  .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
    --hardware "$HW" --inactive-timeout 60 "$@" || { echo "DEPLOY FAILED: $SC"; return 1; }
  .venv/bin/python scripts/set_model_string.py "$SC" "$OUT" || return 1
  serve_check "$OUT" "$THINK" || echo "WARN: serving check failed for $SC"
  if [ "$MASK" = 1 ]; then
    echo "=== MASK sweep: $SC ==="
    .venv/bin/bp-sweep --eval mask --model "$SC" > "$LOGS/mask_sweep_$LN.log" 2>&1 \
      || { echo "MASK SWEEP FAILED: $SC"; }
    .venv/bin/bp-targets --eval mask_subdomain_pressure --model "$SC" \
      >> "$LOGS/mask_sweep_$LN.log" 2>&1 || echo "MASK TARGETS FAILED: $SC"
  fi
  echo "=== elicitation ($EVALS): $SC ==="
  .venv/bin/bp-benchmark --phase generate --evals "$EVALS" --models "$SC" \
    --methods "$METHODS" --concurrency 6 > "$LOGS/methods_$LN.log" 2>&1 \
    || echo "WARN: elicitation failed for $SC"
  echo "=== teardown $SC ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || \
    echo "!!! TEARDOWN FAILED for $SC — delete by hand !!!"
}

endpoint_phase logs/selfpred_gemma 2x_nvidia_h100_80gb_sxm gemma-4-31b-selfpred-v1 gemma_v1 \
  tau2_transfer,mask_subdomain_pressure 1 thinking-off
endpoint_phase logs/selfpred_corpus_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b qwen72_base \
  tau2_transfer 0 "" --deploy-model Qwen/Qwen2.5-72B-Instruct
endpoint_phase logs/selfpred_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b-selfpred-v1 qwen72_v1 \
  tau2_transfer 0 ""
endpoint_phase logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm llama-3.3-70b-selfpred-v2fix-tg v2fix \
  tau2_transfer 0 ""

echo "=== phase C: offline cross_model_mean ==="
.venv/bin/bp-benchmark --phase generate --evals tau2_transfer \
  --models "$(IFS=,; echo "${ALL6[*]}")" --methods cross_model_mean --concurrency 4 \
  > "$LOGS/xmm.log" 2>&1 || echo "WARN: tau2_transfer xmm failed"
.venv/bin/bp-benchmark --phase generate --evals mask_subdomain_pressure \
  --models gemma-4-31b,gemma-4-31b-selfpred-v1 --methods cross_model_mean --concurrency 4 \
  >> "$LOGS/xmm.log" 2>&1 || echo "WARN: mask xmm failed"

echo "############ FINETUNE NEW-EVAL RUNS FINISHED $(date +%H:%M:%S) ############"
for M in "${ALL6[@]}"; do
  n=$(ls "results/tau2_transfer/$M/predictions/" 2>/dev/null | wc -l)
  echo "  $M: tau2_transfer $n prediction files"
done
for M in gemma-4-31b gemma-4-31b-selfpred-v1; do
  n=$(ls "results/mask_subdomain_pressure/$M/predictions/" 2>/dev/null | wc -l)
  echo "  $M: mask_subdomain_pressure $n prediction files"
done
