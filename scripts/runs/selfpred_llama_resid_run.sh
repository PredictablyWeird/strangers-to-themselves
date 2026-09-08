#!/usr/bin/env bash
# Residual-target selfpred Llama (roadmap §3), endpoint phase. Mirrors
# scripts/runs/selfpred_gemma_resid_run.sh with llama's conventions (serverless Turbo base, no
# thinking flag, 4xH100 for the 70B LoRA). Requires the finetune to be done
# (logs/selfpred_llama_resid/together_job.json, status completed). Order is cheapest-info-first:
# base ring predictions are serverless and run before any endpoint exists. Unlike the gemma run,
# the corpus-wide tuned behavior re-elicitation is NOT part of this pass; the endpoint phase ends
# after the 3-eval measurement + ask predictions. Endpoint torn down on EXIT (bills/min); the
# cross_model_mean predictions run after teardown (they never call the subject model).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_llama_resid
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=llama-3.3-70b-selfpred-resid-tg
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown $SHORTCUT endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== base ring predictions (serverless llama-3.3-70b-tg; both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model base --base-shortcut llama-3.3-70b-tg --deviation --run "$RUN" --workers 16 || exit 1

echo "=== deploy on 4xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$RUN" \
  --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 60 || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$RUN" || exit 1

echo "=== confirm the endpoint serves (llama: plain chat, no thinking flag) ==="
.venv/bin/python - "$RUN" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
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
sys.exit("endpoint not serving after 5min")
PY

echo "=== tuned ring predictions (both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model tuned --base-shortcut llama-3.3-70b-tg --deviation --run "$RUN" --workers 16 || exit 1

for EV in sycophancy_pushback discrimeval capability_mmlu; do
  echo "=== measure behavior: $EV ==="
  .venv/bin/bp-sweep --eval "$EV" --model "$SHORTCUT" --concurrency 12 \
    > "$LOGS/${EV}_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval "$EV" --model "$SHORTCUT" >> "$LOGS/${EV}_$SHORTCUT.log" 2>&1 || \
    echo "WARN: $EV failed for $SHORTCUT (see $LOGS/${EV}_$SHORTCUT.log)"
done

echo "=== ask predictions (tuned settings) on the 3 evals ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_tuned.log" 2>&1 || echo "WARN: ask predictions failed"

echo "=== endpoint phase done — teardown before the offline pass ==="
teardown; trap - EXIT

echo "=== cross_model_mean (offline; no endpoint) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods cross_model_mean --concurrency 4 \
  > "$LOGS/pred_${SHORTCUT}_xmm.log" 2>&1 || echo "WARN: cross_model_mean failed"

echo "=== done ==="
# Offline afterwards (no endpoint): the report —
#   .venv/bin/python scripts/selfpred_report.py --evals-repo $EVALS_REPO --stage report \
#     --run logs/selfpred_llama_resid --behavior logs/selfpred_corpus/behavior_llama.jsonl \
#     --donor-behavior logs/selfpred_corpus_gemma/behavior_gemma.jsonl logs/selfpred_corpus_qwen72/behavior_qwen72.jsonl \
#     --out results/reports/selfpred_finetune_llama_resid.md
