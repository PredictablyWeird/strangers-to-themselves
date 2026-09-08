#!/usr/bin/env bash
# v2fix is a BRAND-NEW model: run_endpoint_model.sh measured only tau2+reward_hacking (the other
# finetunes already had syco/discrim/mmlu from earlier sessions). This measures those three evals'
# behavior for v2fix and predicts all methods on them. Endpoint torn down on EXIT (bills per minute).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
SHORTCUT=llama-3.3-70b-selfpred-v2fix-tg
OUT=logs/selfpred_corpus_r2_fixed
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() { echo "=== teardown $SHORTCUT (bills per minute) ==="; .venv/bin/python scripts/finetune_together.py teardown --out "$OUT" || echo "!!! TEARDOWN FAILED !!!"; }
trap teardown EXIT

echo "=== deploy $SHORTCUT on 4xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$OUT" \
  --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 30 || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$OUT" || exit 1

for EV in sycophancy_pushback discrimeval capability_mmlu; do
  echo "=== measure behavior: $EV ==="
  .venv/bin/bp-sweep --eval "$EV" --model "$SHORTCUT" --concurrency 12 \
    > "$LOGS/${EV}_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval "$EV" --model "$SHORTCUT" >> "$LOGS/${EV}_$SHORTCUT.log" 2>&1 || \
    echo "WARN: $EV failed for $SHORTCUT (see $LOGS/${EV}_$SHORTCUT.log)"
done

echo "=== predictions: tuned settings (syco/discrim/mmlu) ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_rest_tuned.log" 2>&1 || echo "WARN: tuned preds failed"

echo "=== predictions: oracle_pairwise (syco/discrim/mmlu) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods oracle_pairwise --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_rest_op.log" 2>&1 || echo "WARN: oracle_pairwise failed"

echo "=== done $SHORTCUT (syco/discrim/mmlu) ==="
