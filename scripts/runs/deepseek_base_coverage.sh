#!/usr/bin/env bash
# DeepSeek finetune family, ITEM 1: base coverage for deepseek-v4-flash-off-tg. Everything here is
# SERVERLESS (no endpoint, no teardown): 3-eval behavior baseline + full-grid ask predictions,
# the base-side Binder held-out eval for the intro30k pair, and the base selfpred rings for both
# the v1 run (rate objective only — the gemma-v1 precedent) and the resid run (--deviation).
# All calls carry chat_template_kwargs.thinking=false via the shortcut / --base-shortcut /
# --extra-body (DeepSeek's native template flag; see the deepseek-v4-flash-off-tg entry).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=deepseek-v4-flash-off-tg
BASE_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731
EXTRA_BODY='{"chat_template_kwargs": {"thinking": false}}'
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

for EV in sycophancy_pushback discrimeval capability_mmlu; do
  echo "=== measure behavior: $EV ==="
  .venv/bin/bp-sweep --eval "$EV" --model "$SHORTCUT" --concurrency 12 \
    > "$LOGS/${EV}_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval "$EV" --model "$SHORTCUT" >> "$LOGS/${EV}_$SHORTCUT.log" 2>&1 || \
    echo "WARN: $EV failed for $SHORTCUT (see $LOGS/${EV}_$SHORTCUT.log)"
done

echo "=== ask predictions (FULL grids) on the 3 evals ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 12 \
  > "$LOGS/pred_${SHORTCUT}_full.log" 2>&1 || echo "WARN: ask predictions failed"

echo "=== base-side Binder held-out eval (serverless, thinking off) ==="
if [ -f logs/introspection_finetune_deepseek30k/eval_deepseek-ai-DeepSeek-V4-Flash-0731.json ]; then
  echo "already stored — skipping"
else
  .venv/bin/python scripts/finetune_together.py evaluate \
    --out logs/introspection_finetune_deepseek30k \
    --model "$BASE_MODEL" --extra-body "$EXTRA_BODY" --workers 16 || echo "WARN: base Binder eval failed"
fi

echo "=== base selfpred rings: v1 run (rate objective only) ==="
if [ -f logs/selfpred_deepseek/pred_base.jsonl ]; then
  echo "already stored — skipping"
else
  .venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
    --model base --base-shortcut "$SHORTCUT" --run logs/selfpred_deepseek --workers 16 || \
    echo "WARN: v1 base rings failed"
fi

echo "=== base selfpred rings: resid run (rate + deviation objectives) ==="
if [ -f logs/selfpred_deepseek_resid/pred_base.jsonl ]; then
  echo "already stored — skipping"
else
  .venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
    --model base --base-shortcut "$SHORTCUT" --deviation --run logs/selfpred_deepseek_resid \
    --workers 16 || echo "WARN: resid base rings failed"
fi

echo "=== base coverage done ==="
