#!/usr/bin/env bash
# gemma-4-31B intro30k (Binder-style property self-prediction LoRA), EVAL phase. Mirrors how
# llama's intro30k was evaluated (finetune_together.py `evaluate` scores Binder held-out accuracy
# for base + tuned; the tuned side needs the dedicated endpoint, the base side is serverless),
# then the dispositional coverage of tab:intro30k (bp-sweep/bp-targets + ask predictions on
# sycophancy_pushback, discrimeval, capability_mmlu). gemma is a Together reasoning model: every
# call needs chat_template_kwargs.enable_thinking=false (the LoRA inherits the base template).
# PropensityBench is intentionally absent (tool-calling on a fresh endpoint is its own recipe).
# Endpoint torn down on EXIT (bills/min).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
OUT=logs/introspection_finetune_gemma30k
SHORTCUT=gemma-4-31b-intro30k
BASE_MODEL=google/gemma-4-31B-it
EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": false}}'
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

echo "=== base-side Binder held-out eval (serverless gemma, thinking off) ==="
if [ -f "$OUT/eval_google-gemma-4-31B-it.json" ]; then
  echo "already stored — skipping"
else
  .venv/bin/python scripts/finetune_together.py evaluate --out "$OUT" \
    --model "$BASE_MODEL" --extra-body "$EXTRA_BODY" --workers 16 || exit 1
fi

teardown() {
  echo "=== teardown $SHORTCUT endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || \
    echo "!!! TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== deploy intro LoRA on 2xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
  --hardware 2x_nvidia_h100_80gb_sxm --inactive-timeout 60 || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$OUT" || exit 1

echo "=== serving check (thinking off) ==="
.venv/bin/python - "$OUT" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
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

echo "=== degeneracy check: tuned vs base completions (thinking off) ==="
.venv/bin/python - "$OUT" <<'PY' || { echo "TUNED MODEL LOOKS DEGENERATE — stopping before the eval spend"; exit 1; }
import json, os, sys
from openai import OpenAI
rec = json.loads(open(os.path.join(sys.argv[1], "together_endpoint.json")).read())
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
prompts = [
    "In one sentence, what is the capital of France and what river runs through it?",
    "List three colors, comma-separated.",
    "Briefly explain why the sky is blue.",
    "Write a one-sentence summary of the plot of Romeo and Juliet.",
]
eb = {"chat_template_kwargs": {"enable_thinking": False}}
def sample(model):
    outs = []
    for p in prompts:
        r = c.chat.completions.create(model=model, messages=[{"role": "user", "content": p}],
                                      temperature=0.0, max_tokens=120, extra_body=eb)
        outs.append(r.choices[0].message.content or "")
    return outs
tuned = sample(rec["endpoint_name"])
base = sample("google/gemma-4-31B-it")
bad = 0
for p, t, b in zip(prompts, tuned, base):
    words = t.split()
    looping = len(words) >= 20 and len(set(words)) / len(words) < 0.35
    empty = not t.strip()
    bad += empty or looping
    print(f"--- {p}\n  TUNED: {t[:200]!r}{' [EMPTY]' if empty else ''}{' [LOOPING]' if looping else ''}\n  BASE : {b[:200]!r}")
if bad >= 2:
    sys.exit(f"{bad}/{len(prompts)} tuned completions empty/looping")
print("tuned completions look non-degenerate")
PY

echo "=== tuned-side Binder held-out eval (endpoint, thinking off) ==="
EP=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/together_endpoint.json'))['endpoint_name'])")
.venv/bin/python scripts/finetune_together.py evaluate --out "$OUT" \
  --model "$EP" --extra-body "$EXTRA_BODY" --workers 16 || exit 1

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

echo "=== endpoint phase done ==="
