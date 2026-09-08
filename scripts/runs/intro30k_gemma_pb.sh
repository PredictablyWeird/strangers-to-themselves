#!/usr/bin/env bash
# PropensityBench fill for gemma-4-31b-intro30k: completes the 4-eval coverage for tab:intro30k.
# Adapted from selfpred_gemma_v1_pb.sh (ask methods run FULL grids, not --tuned-only, so a re-tune
# cannot orphan the files). data/full + --max-total-scenarios 20 reproduces base gemma's original
# 20 dev task-scenarios (base has since grown to 26 conditions, so parity asserts the 20-overlap,
# not equality). Routing: litellm provider "openai" against the Together OpenAI-compatible
# gateway — the together_ai provider refuses `tools` for this model, and PB is all tool-calling.
# Cyber-security stays held-out TEST (the guard is active in the bp-sweep PB path).
# Endpoint torn down on EXIT.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/introspection_finetune_gemma30k
SHORTCUT=gemma-4-31b-intro30k
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
# The harness stamps OPENAI_API_KEY from API_KEYS (its .env holds only the OpenRouter key;
# load_dotenv() does not override an exported var, so this wins).
export API_KEYS="(\"$TOGETHER_API_KEY\")"

teardown() {
  echo "=== teardown $SHORTCUT endpoint (bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== deploy on 2xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$RUN" \
  --hardware 2x_nvidia_h100_80gb_sxm --inactive-timeout 60 || exit 1
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
            tools=tools, temperature=0.0, max_tokens=50,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        m = r.choices[0].message
        if m.tool_calls:
            print("serves OK, tool_calls:", [t.function.name for t in m.tool_calls]); sys.exit(0)
        print("  no tool_calls yet:", repr(m.content)[:60])
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("endpoint not serving tool calls after 5min")
PY

echo "=== PB sweep (guarded; 20 dev task-scenarios matching base gemma's original set) ==="
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

echo "=== verify condition parity with base gemma (20-overlap; base has 26 now) ==="
.venv/bin/python - <<'PY' || echo "WARN: condition mismatch vs base gemma — check before comparing"
import json
base = set(json.load(open('results/propensitybench/gemma-4-31b/targets.json'))['targets'])
res = set(json.load(open('results/propensitybench/gemma-4-31b-intro30k/targets.json'))['targets'])
print(f"tuned {len(res)} conditions; shared with base {len(base & res)}; tuned-only {sorted(res - base)[:2]}")
assert len(res) == 20 and res <= base
PY

echo "=== ask-method predictions (FULL grids, re-tune-proof) on PB ==="
.venv/bin/bp-benchmark --phase generate \
  --evals propensitybench --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_pb.log" 2>&1 || echo "WARN: PB ask preds failed"

echo "=== PB endpoint phase done ==="
