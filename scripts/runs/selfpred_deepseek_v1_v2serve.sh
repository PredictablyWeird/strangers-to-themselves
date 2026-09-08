#!/usr/bin/env bash
# DeepSeek selfpred-v1 eval phase against an ALREADY-DEPLOYED v2/DMI endpoint (the classic v1
# endpoint path cannot start LoRAs on the V4-Flash base — logs/selfpred_deepseek/DMI_ATTEMPT.md).
# Serving = the registered "Final Merged" model on endpoint bp-ds-spv1; adapter identity was
# verified (deployment model_id + behavioral diff vs base) before this script runs. Assumes
# logs/selfpred_deepseek/together_endpoint.json holds the v2 ids and models.yaml is repointed.
# Teardown on EXIT via scripts/dmi_teardown.py (two-phase rm, retried until gone).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_deepseek
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=deepseek-v4-flash-selfpred-v1-tg
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown $SHORTCUT v2 endpoint (bills per minute) ==="
  .venv/bin/python scripts/dmi_teardown.py "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete deployment+endpoint by hand (tg beta endpoints rm) !!!"
}
trap teardown EXIT

echo "=== tuned ring predictions (rate objective) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model tuned --base-shortcut deepseek-v4-flash-off-tg --run "$RUN" --workers 16 || exit 1

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

echo "=== endpoint phase done — teardown before the offline pass ==="
teardown; trap - EXIT

echo "=== cross_model_mean (offline; no endpoint) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods cross_model_mean --concurrency 4 \
  > "$LOGS/pred_${SHORTCUT}_xmm.log" 2>&1 || echo "WARN: cross_model_mean failed"

echo "=== done ==="
