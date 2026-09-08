#!/usr/bin/env bash
# DeepSeek RESIDUAL-target selfpred endpoint phase. Mirrors scripts/runs/selfpred_llama_resid_run.sh
# plus the item-A corpus fill (selfpred_llama_resid_corpus.sh) in ONE session: tuned rings with
# --deviation, 3-eval measurement + tuned asks, the 59,570-item corpus-wide tuned behavior
# re-elicitation, teardown, then the offline cross_model_mean + report (donors llama + qwen72).
# Base rings come from deepseek_base_coverage.sh. thinking=false everywhere (inherited template);
# 2xB200 (the only hardware for this base). Endpoint torn down on EXIT.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_deepseek_resid
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=deepseek-v4-flash-selfpred-resid-tg
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

[ -f "$RUN/pred_base.jsonl" ] || { echo "run deepseek_base_coverage.sh first (no pred_base)"; exit 1; }

teardown() {
  echo "=== teardown $SHORTCUT endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== deploy on 2xB200 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$RUN" \
  --hardware 2x_nvidia_b200_180gb_sxm --inactive-timeout 60 || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$RUN" || exit 1

echo "=== confirm the endpoint serves (thinking off) ==="
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
            extra_body={"chat_template_kwargs": {"thinking": False}})
        if (r.choices[0].message.content or "").strip():
            print("serves OK:", r.choices[0].message.content.strip()); sys.exit(0)
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving after 5min")
PY

echo "=== tuned ring predictions (both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model tuned --base-shortcut deepseek-v4-flash-off-tg --deviation --run "$RUN" --workers 16 || exit 1

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

echo "=== tuned behavior re-elicitation on the corpus (59,570 items — the long pass) ==="
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")
.venv/bin/python scripts/build_selfpred_corpus.py --evals-repo "$EVALS_REPO" --stage elicit \
  --out "$RUN" --behavior-file behavior_tuned.jsonl \
  --model-name "$EP" --extra-body-from deepseek-v4-flash-off-tg --workers 16 || exit 1

echo "=== endpoint phase done — teardown before the offline passes ==="
teardown; trap - EXIT

echo "=== cross_model_mean (offline; no endpoint) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods cross_model_mean --concurrency 4 \
  > "$LOGS/pred_${SHORTCUT}_xmm.log" 2>&1 || echo "WARN: cross_model_mean failed"

echo "=== offline report (rate + deviation objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage report \
  --run "$RUN" --behavior logs/selfpred_corpus_deepseek/behavior_deepseek.jsonl \
  --donor-behavior logs/selfpred_corpus/behavior_llama.jsonl \
                   logs/selfpred_corpus_qwen72/behavior_qwen72.jsonl \
  --out results/reports/selfpred_finetune_deepseek_resid.md || exit 1

echo "=== done ==="
