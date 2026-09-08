#!/usr/bin/env bash
# deepseek intro30k EVAL phase on v2/DMI serving (merged checkpoint — see
# logs/selfpred_deepseek/DMI_ATTEMPT.md). One endpoint session: degeneracy gate, tuned Binder
# held-out eval, 3-eval coverage + asks, PropensityBench at the 20-scenario parity set via the
# litellm openai route. Deploy via scripts/dmi_deploy.py (model_id-verified), teardown via
# scripts/dmi_teardown.py on EXIT. thinking=false on every call; cyber stays held-out TEST.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
OUT=logs/introspection_finetune_deepseek30k
SHORTCUT=deepseek-v4-flash-intro30k-tg
# Together model/revision ids for this finetune (account-scoped; override for your own job).
ML=${ML:?set ML to the Together model id (ml_...)}
RV=${RV:?set RV to the Together model revision id (rv_...)}
EXTRA_BODY='{"chat_template_kwargs": {"thinking": false}}'
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
export API_KEYS="(\"$TOGETHER_API_KEY\")"

[ -f "$OUT/eval_deepseek-ai-DeepSeek-V4-Flash-0731.json" ] || \
  { echo "run deepseek_base_coverage.sh first (no base Binder eval)"; exit 1; }

teardown() {
  echo "=== teardown $SHORTCUT v2 endpoint (bills per minute) ==="
  .venv/bin/python scripts/dmi_teardown.py "$OUT" || \
    echo "!!! TEARDOWN FAILED — delete deployment+endpoint by hand (tg beta endpoints rm) !!!"
}
trap teardown EXIT

echo "=== deploy merged intro30k model (v2/DMI, model_id-verified) ==="
.venv/bin/python scripts/dmi_deploy.py "$OUT" bp-ds-intro30k "$ML" "$RV" || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$OUT" || exit 1
EP=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/together_endpoint.json'))['endpoint_name'])")

echo "=== degeneracy check: tuned vs base completions (thinking off) ==="
.venv/bin/python - "$EP" <<'PY' || { echo "TUNED MODEL LOOKS DEGENERATE — stopping before the eval spend"; exit 1; }
import os, sys
from openai import OpenAI
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
prompts = [
    "In one sentence, what is the capital of France and what river runs through it?",
    "List three colors, comma-separated.",
    "Briefly explain why the sky is blue.",
    "Write a one-sentence summary of the plot of Romeo and Juliet.",
]
eb = {"chat_template_kwargs": {"thinking": False}}
def sample(model):
    return [(c.chat.completions.create(model=model, messages=[{"role":"user","content":p}],
             temperature=0.0, max_tokens=120, extra_body=eb).choices[0].message.content or "")
            for p in prompts]
tuned = sample(sys.argv[1]); base = sample("deepseek-ai/DeepSeek-V4-Flash-0731")
bad = 0
for p, t, b in zip(prompts, tuned, base):
    words = t.split()
    looping = len(words) >= 20 and len(set(words)) / len(words) < 0.35
    empty = not t.strip()
    bad += empty or looping
    print(f"--- {p}\n  TUNED: {t[:200]!r}{' [EMPTY]' if empty else ''}{' [LOOPING]' if looping else ''}\n  BASE : {b[:200]!r}")
if bad >= 2: sys.exit(f"{bad}/{len(prompts)} tuned completions empty/looping")
print("tuned completions look non-degenerate")
PY

echo "=== tuned-side Binder held-out eval (endpoint, thinking off) ==="
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

echo "=== confirm the endpoint serves tool calls via the litellm openai route ==="
"$PB_PY" - "$EP" <<'PY' || { echo "PB SERVING CHECK FAILED"; exit 1; }
import os, sys, time
import litellm
os.environ["OPENAI_API_KEY"] = os.environ["TOGETHER_API_KEY"]
tools = [{"type": "function", "function": {"name": "noop", "description": "no-op tool",
          "parameters": {"type": "object", "properties": {}}}}]
for _ in range(30):
    try:
        r = litellm.completion(
            model=sys.argv[1], custom_llm_provider="openai",
            api_base="https://api.together.xyz/v1",
            messages=[{"role": "user", "content": "Call the noop tool now."}],
            tools=tools, temperature=0.0, max_tokens=50,
            extra_body={"chat_template_kwargs": {"thinking": False}})
        m = r.choices[0].message
        if m.tool_calls:
            print("serves OK, tool_calls:", [t.function.name for t in m.tool_calls]); sys.exit(0)
        print("  no tool_calls yet:", repr(m.content)[:60])
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving tool calls after 5min")
PY

echo "=== PB sweep (guarded; 20 dev task-scenarios — the cross-model parity set) ==="
.venv/bin/bp-sweep --eval propensitybench --model "$SHORTCUT" \
  --pb-repo "$PB_REPO" --pb-python "$PB_PY" --pb-input data/full \
  --grain task_scenario --max-total-scenarios 20 \
  --pb-provider openai --pb-model-name "$EP" --pb-api-base https://api.together.xyz/v1 \
  > "$LOGS/pb_sweep_$SHORTCUT.log" 2>&1 || { echo "PB SWEEP FAILED (see $LOGS/pb_sweep_$SHORTCUT.log)"; exit 1; }

echo "=== extract targets ==="
.venv/bin/bp-targets --eval propensitybench --model "$SHORTCUT" \
  --pb-output "$PB_REPO/results/propensitybench/_harness/$SHORTCUT" \
  --pb-desc-dir "$PB_REPO/data/full" --grain task_scenario \
  >> "$LOGS/pb_sweep_$SHORTCUT.log" 2>&1 || { echo "PB TARGETS FAILED"; exit 1; }

echo "=== verify the 20-scenario parity set (anchor: base gemma's original 20) ==="
.venv/bin/python - <<'PY' || echo "WARN: condition mismatch vs the parity anchor — check before comparing"
import json
gem = set(json.load(open('results/propensitybench/gemma-4-31b/targets.json'))['targets'])
dsl = set(json.load(open('results/propensitybench/deepseek-v4-flash-low/targets.json'))['targets'])
res = set(json.load(open('results/propensitybench/deepseek-v4-flash-intro30k-tg/targets.json'))['targets'])
print(f"tuned {len(res)} conditions; shared with gemma-base {len(gem & res)}; "
      f"shared with deepseek-v4-flash-low {len(dsl & res)}; tuned-only-vs-gemma {sorted(res - gem)[:2]}")
assert len(res) == 20 and res <= gem
PY

echo "=== PB ask-method predictions (FULL grids, re-tune-proof) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals propensitybench --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_pb.log" 2>&1 || echo "WARN: PB ask preds failed"

echo "=== endpoint phase done ==="
