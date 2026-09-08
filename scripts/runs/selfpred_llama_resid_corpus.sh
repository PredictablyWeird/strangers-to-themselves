#!/usr/bin/env bash
# llama-resid ITEM-A fill: the corpus-wide tuned behavior re-elicitation that
# scripts/runs/selfpred_llama_resid_run.sh deliberately skipped, plus the offline report — so
# logs/selfpred_llama_resid ends with the same artifact set as logs/selfpred_gemma_resid
# (pred_base/pred_tuned/behavior_tuned + results/reports/selfpred_finetune_llama_resid.*).
# The 59,570-item elicitation is the only endpoint-bound step; the endpoint is torn down
# before the report (which is offline). llama needs no chat-template flag (unlike gemma's
# enable_thinking=false), so no --extra-body-from. Endpoint torn down on EXIT (bills/min).
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

echo "=== tuned behavior re-elicitation on the corpus (59,570 items — the long pass) ==="
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")
.venv/bin/python scripts/build_selfpred_corpus.py --evals-repo "$EVALS_REPO" --stage elicit \
  --out "$RUN" --behavior-file behavior_tuned.jsonl \
  --model-name "$EP" --workers 16 || exit 1

echo "=== endpoint phase done — teardown before the offline report ==="
teardown; trap - EXIT

echo "=== offline report (rate + deviation objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage report \
  --run "$RUN" --behavior logs/selfpred_corpus/behavior_llama.jsonl \
  --donor-behavior logs/selfpred_corpus_gemma/behavior_gemma.jsonl \
                   logs/selfpred_corpus_qwen72/behavior_qwen72.jsonl \
  --out results/reports/selfpred_finetune_llama_resid.md || exit 1

echo "=== done ==="
