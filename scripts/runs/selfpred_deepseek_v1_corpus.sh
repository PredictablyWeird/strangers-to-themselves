#!/usr/bin/env bash
# DeepSeek selfpred-v1 corpus fill: the 59,570-item tuned behavior re-elicitation the v1 session
# skipped, plus the offline v1 report — gives the fixed-point row (tuned-vs-own, drift) that
# tab:selfpred-fixedpoint needs. DMI v2 serving of the merged v1 checkpoint (dmi_deploy verifies
# model_id); teardown on EXIT via dmi_teardown (scale-to-zero then endpoint delete). Report
# naming matches the exporter's v1 convention: results/reports/selfpred_finetune_deepseek.*
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_deepseek
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
# Together model/revision ids for this finetune (account-scoped; override for your own job).
ML=${ML:?set ML to the Together model id (ml_...)}
RV=${RV:?set RV to the Together model revision id (rv_...)}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

teardown() {
  echo "=== teardown v2 endpoint (bills per minute) ==="
  .venv/bin/python scripts/dmi_teardown.py "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete deployment+endpoint by hand (tg beta endpoints rm) !!!"
}
trap teardown EXIT

echo "=== deploy merged v1 model (v2/DMI, model_id-verified) ==="
.venv/bin/python scripts/dmi_deploy.py "$RUN" bp-ds-spv1 "$ML" "$RV" || exit 1
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")

echo "=== tuned behavior re-elicitation on the corpus (59,570 items — the long pass) ==="
.venv/bin/python scripts/build_selfpred_corpus.py --evals-repo "$EVALS_REPO" --stage elicit \
  --out "$RUN" --behavior-file behavior_tuned.jsonl \
  --model-name "$EP" --extra-body-from deepseek-v4-flash-off-tg --workers 16 || exit 1

echo "=== endpoint phase done — teardown before the offline report ==="
teardown; trap - EXIT

echo "=== offline report (rate objective, v1 convention) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage report \
  --run "$RUN" --behavior logs/selfpred_corpus_deepseek/behavior_deepseek.jsonl \
  --out results/reports/selfpred_finetune_deepseek.md || exit 1

echo "=== done ==="
