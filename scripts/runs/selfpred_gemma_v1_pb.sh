#!/usr/bin/env bash
# PropensityBench fill for gemma-4-31b-selfpred-v1: the last gap in the three main finetune
# pairs. Adapted from selfpred_gemma_resid_pb.sh; ask methods run FULL grids (not
# --tuned-only) so a later re-tune cannot orphan the files.
# data/full + --max-total-scenarios 20 reproduces base gemma's 20 dev task-scenarios exactly
# (verified offline against results/propensitybench/gemma-4-31b/targets.json). Routing: litellm
# provider "openai" against the Together OpenAI-compatible gateway — the together_ai provider
# refuses `tools` for this model, and PB is all tool-calling. Endpoint torn down on EXIT.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_gemma
SHORTCUT=gemma-4-31b-selfpred-v1
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

echo "=== PB sweep (guarded; 20 dev task-scenarios matching base gemma) ==="
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

echo "=== verify condition parity with base gemma ==="
.venv/bin/python - <<'PY' || echo "WARN: condition mismatch vs base gemma — check before comparing"
import json
base = set(json.load(open('results/propensitybench/gemma-4-31b/targets.json'))['targets'])
res = set(json.load(open('results/propensitybench/gemma-4-31b-selfpred-v1/targets.json'))['targets'])
print(f"shared {len(base & res)}/20; resid-only {sorted(res - base)[:2]}")
assert base == res
PY

echo "=== ask-method predictions (FULL grids, re-tune-proof) on PB ==="
.venv/bin/bp-benchmark --phase generate \
  --evals propensitybench --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_pb_selfreport.log" 2>&1 || echo "WARN: PB self_report preds failed"

echo "=== PB endpoint phase done ==="
# Offline afterwards: cross_model_mean for base+resid on PB, then
#   bp-evaluate --pool gemmaresidpair --evals propensitybench
