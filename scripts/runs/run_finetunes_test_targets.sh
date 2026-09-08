#!/usr/bin/env bash
# Finetune TEST-target completion (2026-08-15: complete test targets everywhere).
# MEASUREMENT ONLY — no predictions; rates stay in logs (pre-registration gate).
# Gaps: no finetune has PB test cells (dev-20 vintage); llama-tg misses reward_hacking's
# 23 test conditions. Phase A (serverless, safe to run alongside the MASK endpoint driver):
# gemma base + llama-tg PB test top-ups via the openai route, llama-tg RH sandbox sweep +
# test-key merge. Phase B (endpoints — run ONLY after run_finetunes_mask.sh has finished,
# the state dirs collide): gemma-v1 / qwen72 base / qwen72-v1 / v2fix PB test top-ups.
# Usage: run_finetunes_test_targets.sh A   (or B)
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
PHASE="${1:?pass phase A or B}"
LOGS=logs/finetune_test_targets; mkdir -p "$LOGS"
B=results/_backup/targets_pre_test_$(date +%Y%m%d); mkdir -p "$B"
export API_KEYS="(\"$TOGETHER_API_KEY\")"
API=https://api.together.xyz/v1

pb_test() {  # pb_test <shortcut> <pb-model-name>
  cp "results/propensitybench/$1/targets.json" "$B/pb_targets_$1.json" 2>/dev/null || true
  .venv/bin/python scripts/pb_topup.py --model "$1" --split test \
    --pb-provider openai --pb-model-name "$2" --pb-api-base "$API" \
    > "$LOGS/pb_test_$1.log" 2>&1 && echo "[ok] pb-test $1" || echo "[FAIL] pb-test $1"
}

if [ "$PHASE" = A ]; then
  pb_test gemma-4-31b "google/gemma-4-31B-it" &
  pb_test llama-3.3-70b-tg "meta-llama/Llama-3.3-70B-Instruct-Turbo" &
  (
    M=llama-3.3-70b-tg; SB="/tmp/claude-1000/bp_sandbox_rh_$M"; rm -rf "$SB"; mkdir -p "$SB"
    BP_RESULTS_DIR="$SB" .venv/bin/bp-sweep --eval reward_hacking --model "$M" --concurrency 8 \
      > "$LOGS/rh_$M.log" 2>&1 \
      && BP_RESULTS_DIR="$SB" .venv/bin/bp-targets --eval reward_hacking --model "$M" \
           >> "$LOGS/rh_$M.log" 2>&1 \
      && cp "results/reward_hacking/$M/targets.json" "$B/rh_targets_$M.json" \
      && .venv/bin/python - "$M" "$SB" <<'PY' && echo "[ok] rh $M" || echo "[FAIL] rh $M"
import json, sys
m, sb = sys.argv[1], sys.argv[2]
canon = f"results/reward_hacking/{m}/targets.json"
doc = json.load(open(canon)); fresh = json.load(open(f"{sb}/reward_hacking/{m}/targets.json"))
added = 0
for k, cell in fresh["targets"].items():
    if k not in doc["targets"]:
        doc["targets"][k] = cell; added += 1
json.dump(doc, open(canon, "w"), indent=1)
print(f"[merge {m}] +{added} conditions -> {len(doc['targets'])}")
PY
    rm -rf "$SB"
  ) &
  wait
else
  source <(sed -n '/^teardown_all()/,/^}/p;/^serve_check()/,/^}/p' scripts/runs/run_finetunes_new_evals.sh)
  trap teardown_all EXIT
  ep_pb_test() {  # ep_pb_test <out-dir> <hardware> <shortcut> [thinking-off] [--deploy-model <id>]
    local OUT=$1 HW=$2 SC=$3 THINK=${4:-}; shift 4 || shift $#
    .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
      --hardware "$HW" --inactive-timeout 60 "$@" || { echo "[FAIL] deploy $SC"; return 1; }
    .venv/bin/python scripts/set_model_string.py "$SC" "$OUT" || return 1
    serve_check "$OUT" "$THINK" || echo "WARN: serve check $SC"
    local EP
    EP=$(.venv/bin/python -c "import json;print(json.load(open('$OUT/together_endpoint.json'))['endpoint_name'])")
    pb_test "$SC" "$EP"
    .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || echo "!!! TEARDOWN FAILED $SC !!!"
  }
  ep_pb_test logs/selfpred_gemma 2x_nvidia_h100_80gb_sxm gemma-4-31b-selfpred-v1 thinking-off
  ep_pb_test logs/selfpred_corpus_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b "" --deploy-model Qwen/Qwen2.5-72B-Instruct
  ep_pb_test logs/selfpred_qwen72 4x_nvidia_h100_80gb_sxm qwen2.5-72b-selfpred-v1 ""
  ep_pb_test logs/selfpred_corpus_r2_fixed 4x_nvidia_h100_80gb_sxm llama-3.3-70b-selfpred-v2fix-tg ""
fi

echo "############ FINETUNE TEST TARGETS PHASE $PHASE FINISHED ############"
.venv/bin/python - <<'PY'
import json
sp = json.load(open("results/propensitybench/splits.json"))["assignments"]
for m in ["gemma-4-31b","gemma-4-31b-selfpred-v1","qwen2.5-72b","qwen2.5-72b-selfpred-v1",
          "llama-3.3-70b-tg","llama-3.3-70b-selfpred-v2fix-tg"]:
    try:
        t = json.load(open(f"results/propensitybench/{m}/targets.json"))["targets"]
        n_test = sum(1 for c in t if sp.get("/".join(c.split("/")[:2])) == "test")
        print(f"  {m}: PB total {len(t)}, test cells {n_test}")
    except FileNotFoundError:
        print(f"  {m}: PB missing")
try:
    t = json.load(open("results/reward_hacking/llama-3.3-70b-tg/targets.json"))["targets"]
    print(f"  llama-3.3-70b-tg: reward_hacking {len(t)}/57")
except FileNotFoundError:
    pass
PY
