#!/usr/bin/env bash
# Complete PB method coverage (tuned settings: informed_oracle, pairwise, value) for the
# finetune-pool models. Serverless bases first (no endpoint), then each LoRA on its own
# dedicated endpoint, torn down immediately after its predictions. Skips the defective
# llama-selfpred-v2 (excluded from pools; must not be cited).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
METHODS=informed_oracle,pairwise,value
BENCH() {  # BENCH <model> <logname>
  .venv/bin/bp-benchmark --phase generate --tuned-only --evals propensitybench \
    --models "$1" --methods "$METHODS" --concurrency 6 > "$LOGS/pb_methods_$2.log" 2>&1 \
    || echo "WARN: PB methods failed for $1 (see $LOGS/pb_methods_$2.log)"
}

# Backstop: tear down whatever endpoint state dirs exist, idempotent.
teardown_all() {
  for OUT in logs/selfpred_gemma_resid logs/selfpred_corpus_r2_fixed logs/introspection_finetune_llama30k; do
    .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" >/dev/null 2>&1 || true
  done
  echo "=== teardown backstop ran ==="
}
trap teardown_all EXIT

echo "=== serverless: gemma-4-31b + llama-3.3-70b-tg ==="
BENCH gemma-4-31b gemma_base
BENCH llama-3.3-70b-tg llama_base

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

endpoint_phase() {  # endpoint_phase <out-dir> <hardware> <shortcut> <logname> [thinking-off]
  local OUT=$1 HW=$2 SC=$3 LN=$4 THINK=${5:-}
  echo "=== deploy $SC on $HW ==="
  .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
    --hardware "$HW" --inactive-timeout 60 || { echo "DEPLOY FAILED: $SC"; return 1; }
  .venv/bin/python scripts/set_model_string.py "$SC" "$OUT" || return 1
  serve_check "$OUT" "$THINK" || { echo "SERVING CHECK FAILED: $SC"; }
  echo "=== PB methods: $SC ==="
  BENCH "$SC" "$LN"
  echo "=== teardown $SC ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || \
    echo "!!! TEARDOWN FAILED for $SC — delete by hand !!!"
}

endpoint_phase logs/selfpred_gemma_resid 2x_nvidia_h100_80gb_sxm gemma-4-31b-selfpred-resid gemma_resid thinking-off
endpoint_phase logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm llama-3.3-70b-selfpred-v2fix-tg v2fix
endpoint_phase logs/introspection_finetune_llama30k 4x_nvidia_h100_80gb_sxm llama-3.3-70b-intro30k-tg intro30k

echo "=== refresh reports ==="
.venv/bin/bp-evaluate --pool gemmaresidpair --evals sycophancy_pushback,discrimeval,capability_mmlu,propensitybench > "$LOGS/refresh_pair.log" 2>&1 || echo "WARN pair refresh"
.venv/bin/bp-evaluate --pool finetunes > "$LOGS/refresh_finetunes.log" 2>&1 || echo "WARN finetunes refresh"
.venv/bin/bp-evaluate --pool all > "$LOGS/refresh_all.log" 2>&1 || echo "WARN all refresh"
echo "=== all done ==="
