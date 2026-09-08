#!/usr/bin/env bash
# Deploy a Together dedicated endpoint, run every measurement + prediction for that model, tear the
# endpoint down. Dedicated endpoints bill per minute, so teardown is on an EXIT trap: it runs on
# success, on failure, and on Ctrl-C.
#
#   scripts/runs/run_endpoint_model.sh <models.yaml-shortcut> <log-dir> <hardware> [deploy-model]
#
# `deploy-model` serves a base model whose serverless twin does not exist (e.g. Qwen2.5-72B-Instruct)
# instead of the finetune output recorded in <log-dir>/together_job.json.
#
# SKIP_MEASURE=1 goes straight to the prediction stages — for topping up a model whose behavior is
# already measured, without paying to re-run the (much longer) measurement.
set -uo pipefail

SHORTCUT=${1:?shortcut}; OUT=${2:?log dir}; HW=${3:?hardware}; DEPLOY_MODEL=${4:-}
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown $SHORTCUT (endpoint bills per minute) ==="
  .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || \
    echo "!!! TEARDOWN FAILED for $OUT — delete the endpoint by hand !!!"
}
trap teardown EXIT

echo "=== deploy $SHORTCUT on $HW ==="
if [[ -n "$DEPLOY_MODEL" ]]; then
  .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" --hardware "$HW" \
    --inactive-timeout 30 --deploy-model "$DEPLOY_MODEL" || exit 1
else
  .venv/bin/python scripts/finetune_together.py deploy --out "$OUT" --hardware "$HW" \
    --inactive-timeout 30 || exit 1
fi

# The endpoint name carries a fresh suffix on every deploy, so models.yaml must be repointed.
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$OUT" || exit 1

# A dedicated endpoint serves only us, so push concurrency harder than on the shared serverless
# tiers: the endpoint bills per minute, which makes wall-clock the cost driver.
if [[ "${SKIP_MEASURE:-}" != "1" ]]; then
  echo "=== measure behavior: tau2_policy ==="
  .venv/bin/bp-sweep --eval tau2_policy --model "$SHORTCUT" --max-concurrency 8 \
    > "$LOGS/tau2_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval tau2_policy --model "$SHORTCUT" >> "$LOGS/tau2_$SHORTCUT.log" 2>&1 || \
    echo "WARN: tau2_policy failed for $SHORTCUT (see $LOGS/tau2_$SHORTCUT.log)"

  echo "=== measure behavior: reward_hacking ==="
  .venv/bin/bp-sweep --eval reward_hacking --model "$SHORTCUT" --concurrency 16 \
    > "$LOGS/rh_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval reward_hacking --model "$SHORTCUT" >> "$LOGS/rh_$SHORTCUT.log" 2>&1 || \
    echo "WARN: reward_hacking failed for $SHORTCUT (see $LOGS/rh_$SHORTCUT.log)"
else
  echo "=== SKIP_MEASURE=1: behavior already measured, predicting only ==="
fi

# Tuned settings for the evals that have a tuning.json; reward_hacking has none, so it takes the
# full (small) grid. oracle_pairwise is new and untuned everywhere, but its grid is a single
# setting, so it is generated directly rather than via --tuned-only.
echo "=== predictions: tuned settings ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals sycophancy_pushback,discrimeval,capability_mmlu,tau2_policy \
  --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_tuned.log" 2>&1 || echo "WARN: tuned predictions failed for $SHORTCUT"

echo "=== predictions: oracle_pairwise + reward_hacking ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods oracle_pairwise --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_op.log" 2>&1 || echo "WARN: oracle_pairwise failed for $SHORTCUT"

.venv/bin/bp-benchmark --phase generate --evals reward_hacking --models "$SHORTCUT" \
  --methods self_report,value,pairwise,informed_oracle,oracle_pairwise --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_rh.log" 2>&1 || echo "WARN: reward_hacking predictions failed for $SHORTCUT"

echo "=== done $SHORTCUT ==="
