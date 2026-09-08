#!/usr/bin/env bash
# PB for the Qwen2.5-72B pair: base was never measured, tuned v1 has only a 5-condition fragment.
# Each model: deploy 4xH100 -> guarded sweep (base llama's exact 20 dev task-scenarios) -> targets
# (v1's old fragment records pool in, same model) -> tuned methods (self_report, informed_oracle,
# pairwise, value) -> teardown. litellm openai route (together_ai refuses `tools`).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
export API_KEYS="(\"$TOGETHER_API_KEY\")"
METHODS=self_report,informed_oracle,pairwise,value

teardown_all() {
  for OUT in logs/selfpred_corpus_qwen72 logs/selfpred_qwen72; do
    .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" >/dev/null 2>&1 || true
  done
  echo "=== teardown backstop ran ==="
}
trap teardown_all EXIT

phase() {  # phase <out-dir> <shortcut> <logname> [--deploy-model <id>]
  local OUT=$1 SC=$2 LN=$3; shift 3
  echo "=== deploy $SC on 4xH100 ==="
  .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
    --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 60 "$@" || { echo "DEPLOY FAILED: $SC"; return 1; }
  .venv/bin/python scripts/set_model_string.py "$SC" "$OUT" || return 1
  local EP
  EP=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/together_endpoint.json'))['endpoint_name'])")

  echo "=== serving check (tools) $SC ==="
  "$PB_PY" - "$EP" <<'PY' || echo "WARN: serving check failed for $SC"
import os, sys, time
import litellm
os.environ["OPENAI_API_KEY"] = os.environ["TOGETHER_API_KEY"]
tools = [{"type": "function", "function": {"name": "noop", "description": "no-op tool",
          "parameters": {"type": "object", "properties": {}}}}]
for _ in range(30):
    try:
        r = litellm.completion(model=sys.argv[1], custom_llm_provider="openai",
            api_base="https://api.together.xyz/v1",
            messages=[{"role": "user", "content": "Call the noop tool now."}],
            tools=tools, temperature=0.0, max_tokens=50)
        if r.choices[0].message.tool_calls:
            print("serves OK"); sys.exit(0)
    except Exception as e:
        print("  waiting:", str(e)[:80])
    time.sleep(10)
sys.exit("not serving tool calls after 5min")
PY

  echo "=== PB sweep: $SC ==="
  .venv/bin/bp-sweep --eval propensitybench --model "$SC" \
    --pb-repo "$PB_REPO" --pb-python "$PB_PY" --pb-input data/full \
    --grain task_scenario --max-total-scenarios 20 \
    --pb-provider openai --pb-model-name "$EP" --pb-api-base https://api.together.xyz/v1 \
    > "$LOGS/pb_sweep_$LN.log" 2>&1 || { echo "PB SWEEP FAILED: $SC"; }
  echo "=== extract targets: $SC ==="
  .venv/bin/bp-targets --eval propensitybench --model "$SC" \
    --pb-output "$PB_REPO/results/propensitybench/_harness/$SC" \
    --pb-desc-dir "$PB_REPO/data/full" --grain task_scenario \
    >> "$LOGS/pb_sweep_$LN.log" 2>&1 || { echo "PB TARGETS FAILED: $SC"; }

  echo "=== PB methods: $SC ==="
  .venv/bin/bp-benchmark --phase generate --tuned-only --evals propensitybench \
    --models "$SC" --methods "$METHODS" --concurrency 6 \
    > "$LOGS/pb_methods_$LN.log" 2>&1 || echo "WARN: methods failed for $SC"

  echo "=== teardown $SC ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || \
    echo "!!! TEARDOWN FAILED for $SC — delete by hand !!!"
}

phase logs/selfpred_corpus_qwen72 qwen2.5-72b qwen72_base --deploy-model Qwen/Qwen2.5-72B-Instruct
phase logs/selfpred_qwen72 qwen2.5-72b-selfpred-v1 qwen72_v1

echo "=== condition parity check ==="
.venv/bin/python - <<'PY' || echo "WARN: parity mismatch"
import json
ref = set(json.load(open('results/propensitybench/llama-3.3-70b-tg/targets.json'))['targets'])
for m in ('qwen2.5-72b', 'qwen2.5-72b-selfpred-v1'):
    got = set(json.load(open(f'results/propensitybench/{m}/targets.json'))['targets'])
    print(f"{m}: {len(got & ref)}/{len(ref)} shared, {len(got - ref)} extra")
PY

echo "=== offline xmm + report refresh ==="
.venv/bin/bp-benchmark --phase generate --evals propensitybench \
  --models qwen2.5-72b,qwen2.5-72b-selfpred-v1 --methods cross_model_mean --concurrency 4 \
  > "$LOGS/pb_xmm_qwen72.log" 2>&1 || echo "WARN: qwen xmm failed"
.venv/bin/bp-evaluate --pool finetunes > "$LOGS/refresh_finetunes2.log" 2>&1 || echo "WARN finetunes refresh"
.venv/bin/bp-evaluate --pool all > "$LOGS/refresh_all2.log" 2>&1 || echo "WARN all refresh"
echo "=== all done ==="
