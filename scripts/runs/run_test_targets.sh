#!/usr/bin/env bash
# TEST-SPLIT TARGET MEASUREMENT ONLY (2026-08-15). Pre-registration gate: this run
# measures behavior and extracts targets for the frozen test split but computes NO predictions,
# and nothing here surfaces test rates into the conversation — summaries go to logs only.
#
# Gaps closed (from the coverage census, counts only):
#   tau2:        small six have 12/19 tasks (frontier already 19) -> run the 7 test tasks in a
#                SANDBOXED results root, merge simulations into the canonical single-file
#                results.json (dev sims untouched), re-extract targets for BOTH tau2 evals.
#   discrimeval: llama-4-maverick + qwen3.7-plus-low have 66/99 (all 33 missing are test) ->
#                sandbox sweep + extract, merge ONLY missing keys into canonical targets.
#   PB:          the small-four cohort misses the 10-condition measured test union (cyber +
#                non-cyber test workspaces) -> pb_topup --split test (include-test-domain).
# sycophancy/capability_mmlu/reward_hacking targets already include test (57 everywhere);
# MASK cells are shared-measured (24 incl. test).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/test_targets; mkdir -p "$LOGS"
B=results/_backup/targets_pre_test_$(date +%Y%m%d); mkdir -p "$B"
TEST_TASK_IDS=(1 48 38 10 31 34 36)
SMALL=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

tau2_one() {
  local M="$1"; local SB="/tmp/claude-1000/bp_sandbox_tau2_$M"
  rm -rf "$SB"; mkdir -p "$SB"
  cp "results/tau2_policy/_harness/$M/results.json" "$B/tau2_results_$M.json"
  BP_RESULTS_DIR="$SB" .venv/bin/bp-sweep --eval tau2_policy --model "$M" \
    --task-ids "${TEST_TASK_IDS[@]}" || { echo "[$M] FAIL tau2 test sweep"; return 1; }
  .venv/bin/python - "$M" "$SB" <<'PY' || return 1
import json, sys
m, sb = sys.argv[1], sys.argv[2]
canon = f"results/tau2_policy/_harness/{m}/results.json"
old = json.load(open(canon)); new = json.load(open(f"{sb}/tau2_policy/_harness/{m}/results.json"))
old_ids = {t.get("id") for t in old["tasks"]}
add_tasks = [t for t in new["tasks"] if t.get("id") not in old_ids]
add_sims = [s for s in new["simulations"] if s.get("task_id") not in old_ids]
old["tasks"] += add_tasks; old["simulations"] += add_sims
json.dump(old, open(canon, "w"))
print(f"[merge {m}] +{len(add_tasks)} tasks, +{len(add_sims)} sims "
      f"-> {len(old['tasks'])} tasks, {len(old['simulations'])} sims")
PY
  cp "results/tau2_policy/$M/targets.json" "$B/tau2_policy_targets_$M.json"
  cp "results/tau2_transfer/$M/targets.json" "$B/tau2_transfer_targets_$M.json"
  .venv/bin/bp-targets --eval tau2_policy --model "$M" >> "$LOGS/tau2_$M.log" 2>&1 \
    && .venv/bin/bp-targets --eval tau2_transfer --model "$M" >> "$LOGS/tau2_$M.log" 2>&1 \
    || { echo "[$M] FAIL tau2 re-extract"; return 1; }
  rm -rf "$SB"
}

disc_one() {
  local M="$1"; local SB="/tmp/claude-1000/bp_sandbox_disc_$M"
  rm -rf "$SB"; mkdir -p "$SB"
  BP_RESULTS_DIR="$SB" .venv/bin/bp-sweep --eval discrimeval --model "$M" \
    > "$LOGS/disc_sweep_$M.log" 2>&1 || { echo "[$M] FAIL disc sweep"; return 1; }
  BP_RESULTS_DIR="$SB" .venv/bin/bp-targets --eval discrimeval --model "$M" \
    >> "$LOGS/disc_sweep_$M.log" 2>&1 || { echo "[$M] FAIL disc extract"; return 1; }
  cp "results/discrimeval/$M/targets.json" "$B/discrimeval_targets_$M.json"
  .venv/bin/python - "$M" "$SB" <<'PY' || return 1
import json, sys
m, sb = sys.argv[1], sys.argv[2]
canon = f"results/discrimeval/{m}/targets.json"
doc = json.load(open(canon)); fresh = json.load(open(f"{sb}/discrimeval/{m}/targets.json"))
added = 0
for k, cell in fresh["targets"].items():
    if k not in doc["targets"]:
        doc["targets"][k] = cell; added += 1
json.dump(doc, open(canon, "w"), indent=1)
print(f"[merge {m}] +{added} conditions -> {len(doc['targets'])}")
PY
  rm -rf "$SB"
}

echo "=== tau2 test tasks (small six, parallel) ==="
for M in "${SMALL[@]}"; do ( tau2_one "$M" > "$LOGS/tau2_$M.log" 2>&1 \
  && echo "[ok] tau2 $M" || echo "[FAIL] tau2 $M" ) & done
wait

echo "=== discrimeval test cells (maverick + qwen3.7, parallel) ==="
for M in llama-4-maverick qwen3.7-plus-low; do ( disc_one "$M" \
  && echo "[ok] disc $M" || echo "[FAIL] disc $M" ) & done
wait

echo "=== PB test cells (small-four cohort, parallel) ==="
for M in deepseek-v4-flash-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low; do
  ( cp "results/propensitybench/$M/targets.json" "$B/pb_targets_$M.json"
    .venv/bin/python scripts/pb_topup.py --model "$M" --split test \
      > "$LOGS/pb_test_$M.log" 2>&1 && echo "[ok] pb $M" || echo "[FAIL] pb $M" ) &
done
wait

echo "############ TEST TARGETS FINISHED $(date +%H:%M:%S) ############"
echo "(verification is COUNT-ONLY; rates stay in logs)"
.venv/bin/python - <<'PY'
import json
sp = json.load(open("results/propensitybench/splits.json"))["assignments"]
for m in ["deepseek-v4-flash-low","gemini-3.1-flash-lite-low","gpt-5.4-nano-low",
          "llama-3.3-70b","llama-4-maverick","qwen3.7-plus-low"]:
    n_t2 = len(json.load(open(f"results/tau2_policy/{m}/targets.json"))["targets"])
    n_tt = len(json.load(open(f"results/tau2_transfer/{m}/targets.json"))["targets"])
    n_d  = len(json.load(open(f"results/discrimeval/{m}/targets.json"))["targets"])
    pb   = json.load(open(f"results/propensitybench/{m}/targets.json"))["targets"]
    n_pt = sum(1 for c in pb if sp.get("/".join(c.split("/")[:2])) == "test")
    print(f"  {m}: tau2 {n_t2}/19+{n_tt}/19, disc {n_d}/99, PB test cells {n_pt}/10")
PY
