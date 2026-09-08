#!/usr/bin/env bash
# PropensityBench for llama-3.3-70b-selfpred-v2fix-tg (the CORRECTED round-2 rate LoRA), with the
# cyber test-domain guard active. Completes the paper-grade llama row: the defective -v2's PB
# numbers (+0.40 self_report, heavy drift) must not be cited; this measures the clean checkpoint
# on base llama-tg's exact 20 dev task-scenarios. Same litellm openai-route as the gemma resid
# run (together_ai refuses `tools`). Endpoint torn down on EXIT.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_corpus_r2_fixed
SHORTCUT=llama-3.3-70b-selfpred-v2fix-tg
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
export API_KEYS="(\"$TOGETHER_API_KEY\")"

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
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")

echo "=== confirm the endpoint serves tool calls via the litellm openai route ==="
"$PB_PY" - "$EP" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
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
            tools=tools, temperature=0.0, max_tokens=50)
        m = r.choices[0].message
        if m.tool_calls:
            print("serves OK, tool_calls:", [t.function.name for t in m.tool_calls]); sys.exit(0)
        print("  no tool_calls yet:", repr(m.content)[:60])
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving tool calls after 5min")
PY

echo "=== PB sweep (guarded; 20 dev task-scenarios matching base llama-tg) ==="
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

echo "=== verify condition parity with base llama-tg ==="
.venv/bin/python - <<'PY' || echo "WARN: condition mismatch vs base — check before comparing"
import json
base = set(json.load(open('results/propensitybench/llama-3.3-70b-tg/targets.json'))['targets'])
res = set(json.load(open('results/propensitybench/llama-3.3-70b-selfpred-v2fix-tg/targets.json'))['targets'])
print(f"shared {len(base & res)}/{len(base)}; v2fix-only {sorted(res - base)[:2]}")
assert base == res
PY

echo "=== self_report predictions (tuned settings) on PB ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals propensitybench --models "$SHORTCUT" --methods self_report --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_pb_selfreport.log" 2>&1 || echo "WARN: PB self_report preds failed"

echo "=== PB endpoint phase done ==="
