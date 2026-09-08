#!/usr/bin/env bash
# Corrected selfpred-v2, PART 1: collect v1's OWN binder responses and build the corrected corpus.
# The defective v2 (docs/selfpred-fixed-point.md) trained on base-Llama's binder responses; this
# regenerates them from v1 itself. Ends with logs/selfpred_corpus_r2_fixed/sft_train.jsonl ready to
# finetune. v1's endpoint is torn down on EXIT (success, failure, or interrupt) — it bills per minute.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
CORPUS=logs/selfpred_corpus_r2_fixed
V1_OUT=logs/selfpred_llama
EVALS_REPO=${EVALS_REPO:-$PWD/../emergent-values}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown v1 endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$V1_OUT" || \
    echo "!!! V1 TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== deploy v1 (round-1 LoRA) on 4xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$V1_OUT" \
  --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 30 || exit 1

# Point the shortcut at the fresh endpoint NAME; inference goes through the OpenAI-compatible
# api.together.xyz gateway (see _elicit_model) which routes the endpoint-name model string to the
# dedicated endpoint. Confirm it actually serves before the 6500-call batch (fail fast on a bad deploy).
.venv/bin/python scripts/set_model_string.py llama-3.3-70b-selfpred-v1-tg "$V1_OUT" || exit 1
echo "=== confirm v1 endpoint serves ==="
.venv/bin/python - "$V1_OUT" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
import json, os, sys, time
from openai import OpenAI
rec = json.loads(open(os.path.join(sys.argv[1], "together_endpoint.json")).read())
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
for _ in range(30):
    try:
        r = c.chat.completions.create(model=rec["endpoint_name"],
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0, max_tokens=5)
        if (r.choices[0].message.content or "").strip():
            print("serves OK:", r.choices[0].message.content.strip()); sys.exit(0)
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("v1 endpoint not serving after 5min")
PY

echo "=== collect v1's OWN binder object-level responses (6500 -> ~5000 pairs) ==="
.venv/bin/python scripts/build_selfpred_corpus.py --stage binder \
  --out "$CORPUS" --binder-rows "$CORPUS" \
  --model-shortcut llama-3.3-70b-selfpred-v1-tg \
  --evals-repo "$EVALS_REPO" --n-binder-rows 6500 || exit 1

echo "=== teardown v1 now (corpus build needs no endpoint) ==="
teardown; trap - EXIT

echo "=== build corrected corpus (v1 rate pairs + v1 binder pairs) ==="
.venv/bin/python scripts/build_selfpred_corpus.py --stage pairs \
  --out "$CORPUS" --binder-dir "$CORPUS" --evals-repo "$EVALS_REPO" \
  --n-binder 5000 --variants 2 --holdout-items 5 || exit 1

echo "=== PART 1 DONE. Corrected corpus:"
cat "$CORPUS/pairs_meta.json"
