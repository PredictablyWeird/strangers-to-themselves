#!/usr/bin/env bash
# Full results for claude-sonnet-5-low (first frontier model): measure behavior + extract targets
# on all 6 canonical evals, then generate predictions at the FROZEN tuned settings only (the
# argument sweep is retired — see methods.yaml; list_experiment is excluded via DEFAULT_METHODS).
# few_shot/few_shot_other are untuned single settings, generated explicitly on the 5 evals where
# the pool has them (not reward_hacking), mirroring pool coverage.
#
# Idempotent: bp-sweep/bp-benchmark skip existing work, so re-running resumes. Logs to
# logs/sonnet5/<stage>.log; touches logs/sonnet5/<stage>.done on success.
#
#   scripts/runs/run_sonnet5.sh            # everything
#   STAGES="mmlu syc" scripts/runs/run_sonnet5.sh   # selected stages only
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"

M=claude-sonnet-5-low
BP=.venv/bin
LOGS=logs/sonnet5; mkdir -p "$LOGS"
STAGES=${STAGES:-"mmlu syc discrim rh pb tau2 predict"}

stage() {  # stage <name> <cmd...>: run once, log, mark done
  local name="$1"; shift
  [[ " $STAGES " == *" $name "* ]] || { echo "--- skip $name (not in STAGES)"; return 0; }
  [[ -f "$LOGS/$name.done" ]] && { echo "--- skip $name (done)"; return 0; }
  echo "=== $name START $(date +%H:%M:%S) ==="
  if "$@" >> "$LOGS/$name.log" 2>&1; then
    touch "$LOGS/$name.done"; echo "=== $name DONE $(date +%H:%M:%S) ==="
  else
    echo "!!! $name FAILED (see $LOGS/$name.log)"; return 1
  fi
}

measure() {  # measure <stage> <eval> [extra sweep args...]
  local name="$1" ev="$2"; shift 2
  stage "$name" bash -c "$BP/bp-sweep --eval $ev --model $M $* && $BP/bp-targets --eval $ev --model $M"
}

# --- measurement (cheap -> expensive) ---
measure mmlu    capability_mmlu     --concurrency 8
measure syc     sycophancy_pushback --concurrency 8
measure discrim discrimeval         --concurrency 8
measure rh      reward_hacking      --concurrency 16

# PropensityBench: Scale harness subprocess. Cap 24 TOTAL scenarios (round-robin across domains)
# to match the pool models' ~20-24 conditions — commit b98ebf9 found cap 6 (the built-in cost
# guardrail default) too thin to score. Cyber-security is measured like any domain but forced
# into the frozen test split by the eval's split seam.
stage pb bash -c "$BP/bp-sweep --eval propensitybench --model $M \
    --pb-repo '$PB_REPO' --pb-python '$PB_PY' --pb-input data/full \
    --grain task_scenario --max-total-scenarios 24 && \
  $BP/bp-targets --eval propensitybench --model $M \
    --pb-output '$PB_REPO/results/propensitybench/_harness/$M' \
    --pb-desc-dir '$PB_REPO/data/full' --grain task_scenario"

# tau2: external harness (12 tasks x 8 trials, multi-turn tool-calling; the token-heavy eval).
measure tau2 tau2_policy --max-concurrency 8

# --- predictions: frozen tuned settings only, plus the untuned few_shot pair ---
stage predict bash -c "\
  $BP/bp-benchmark --phase generate --tuned-only \
    --evals sycophancy_pushback,discrimeval,capability_mmlu,reward_hacking,tau2_policy,propensitybench \
    --models $M --concurrency 6 && \
  $BP/bp-benchmark --phase generate \
    --evals sycophancy_pushback,discrimeval,capability_mmlu,tau2_policy,propensitybench \
    --models $M --methods few_shot,few_shot_other --concurrency 6"

echo "############ run_sonnet5 finished $(date +%H:%M:%S) ############"
for f in "$LOGS"/*.done; do [ -e "$f" ] && echo "  DONE $(basename "${f%.done}")"; done
