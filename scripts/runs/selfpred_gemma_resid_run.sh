#!/usr/bin/env bash
# Residual-target selfpred Gemma (roadmap §3), endpoint phase. Requires the finetune to be done
# (logs/selfpred_gemma_resid/together_job.json, status completed). Order is cheapest-info-first:
# base ring predictions are serverless and run before any endpoint exists; the 59,570-item tuned
# behavior re-elicitation is the long pass and runs last. Endpoint torn down on EXIT (bills/min).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_gemma_resid
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=gemma-4-31b-selfpred-resid
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown $SHORTCUT endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== base ring predictions (serverless gemma-4-31b; both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model base --base-shortcut gemma-4-31b --deviation --run "$RUN" --workers 16 || exit 1

echo "=== deploy on 2xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$RUN" \
  --hardware 2x_nvidia_h100_80gb_sxm --inactive-timeout 60 || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$RUN" || exit 1

echo "=== confirm the endpoint serves (LoRA inherits gemma's template: thinking off) ==="
.venv/bin/python - "$RUN" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
import json, os, sys, time
from openai import OpenAI
rec = json.loads(open(os.path.join(sys.argv[1], "together_endpoint.json")).read())
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
for _ in range(30):
    try:
        r = c.chat.completions.create(model=rec["endpoint_name"],
            messages=[{"role": "user", "content": "Reply with the single word: ok"}],
            temperature=0.0, max_tokens=5,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        if (r.choices[0].message.content or "").strip():
            print("serves OK:", r.choices[0].message.content.strip()); sys.exit(0)
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving after 5min")
PY

echo "=== tuned ring predictions (both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model tuned --base-shortcut gemma-4-31b --deviation --run "$RUN" --workers 16 || exit 1

for EV in sycophancy_pushback discrimeval capability_mmlu; do
  echo "=== measure behavior: $EV ==="
  .venv/bin/bp-sweep --eval "$EV" --model "$SHORTCUT" --concurrency 12 \
    > "$LOGS/${EV}_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval "$EV" --model "$SHORTCUT" >> "$LOGS/${EV}_$SHORTCUT.log" 2>&1 || \
    echo "WARN: $EV failed for $SHORTCUT (see $LOGS/${EV}_$SHORTCUT.log)"
done

echo "=== self_report predictions (tuned settings) on the 3 evals ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods self_report --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_selfreport.log" 2>&1 || echo "WARN: self_report preds failed"

echo "=== tuned behavior re-elicitation on the corpus (59,570 items — the long pass) ==="
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")
.venv/bin/python scripts/build_selfpred_corpus.py --evals-repo "$EVALS_REPO" --stage elicit \
  --out "$RUN" --behavior-file behavior_tuned.jsonl \
  --model-name "$EP" --extra-body-from gemma-4-31b --workers 16 || exit 1

echo "=== endpoint phase done ==="
# Offline afterwards (no endpoint): cross_model_mean predictions + the report:
#   .venv/bin/bp-benchmark --phase generate --evals sycophancy_pushback,discrimeval,capability_mmlu \
#     --models gemma-4-31b-selfpred-resid --methods cross_model_mean --concurrency 4
#   .venv/bin/python scripts/selfpred_report.py --evals-repo $EVALS_REPO --stage report \
#     --run logs/selfpred_gemma_resid --behavior logs/selfpred_corpus_gemma/behavior_gemma.jsonl \
#     --donor-behavior logs/selfpred_corpus/behavior_llama.jsonl logs/selfpred_corpus_qwen72/behavior_qwen72.jsonl \
#     --out results/reports/selfpred_finetune_gemma_resid.md
