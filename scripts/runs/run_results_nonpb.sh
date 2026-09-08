#!/usr/bin/env bash
# Produce complete, up-to-date NON-PropensityBench results for the three target models.
# PropensityBench is intentionally excluded (harness measurement pending user confirmation of the
# scenario cap). Resumable: bp-sweep --skip-existing and bp-benchmark skip already-generated files.
#
# Cheapest steps first (llm_prediction refresh) so a misconfig surfaces before costly measurement.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a          # export provider API keys for all subprocesses
BP=.venv/bin

NONPB_EVALS=discrimeval,capability_mmlu,sycophancy_pushback
ALL_MODELS=llama-3.3-70b,llama-4-maverick,deepseek-v4-flash-low

step() { echo; echo ">>> STEP: $*"; }
run()  { echo "+ $*"; "$@" || echo "ERROR ($?) in: $*"; }

# --- Phase 1: refresh llm_prediction (fixed parser) where predictions already exist (cheap) -------
step "llm_prediction refresh — existing non-PB cells"
run $BP/bp-benchmark --evals discrimeval,capability_mmlu,sycophancy_pushback --models llama-3.3-70b \
    --methods llm_prediction --phase generate --fresh
run $BP/bp-benchmark --evals discrimeval --models llama-4-maverick \
    --methods llm_prediction --phase generate --fresh

# --- Phase 2: measure behavior + generate ALL valid methods for the new cells ---------------------
# measure <eval> <model> : run the eval's behavior harness then aggregate to targets.json
measure() {
  local ev="$1" m="$2"
  step "measure behavior — $ev / $m"
  case "$ev" in
    discrimeval)                 run $BP/bp-sweep --eval "$ev" --model "$m" ;;
    capability_mmlu|sycophancy_pushback) run $BP/bp-sweep --eval "$ev" --model "$m" ;;
  esac
  run $BP/bp-targets --eval "$ev" --model "$m"
}
predict_all() {
  local ev="$1" m="$2"
  step "predict all valid methods — $ev / $m"
  run $BP/bp-benchmark --evals "$ev" --models "$m" --phase generate
}

for cell in \
  "capability_mmlu llama-4-maverick" \
  "sycophancy_pushback llama-4-maverick" \
  "discrimeval deepseek-v4-flash-low" \
  "capability_mmlu deepseek-v4-flash-low" \
  "sycophancy_pushback deepseek-v4-flash-low" ; do
  set -- $cell
  measure "$1" "$2"
  predict_all "$1" "$2"
done

# --- Phase 3: interim scoring over the non-PB evals (PB folded in after its measurement) ----------
step "tune + evaluate + plots (non-PB, dev)"
run $BP/bp-tune     --evals "$NONPB_EVALS" --models "$ALL_MODELS"
run $BP/bp-evaluate --evals "$NONPB_EVALS" --models "$ALL_MODELS" --split dev
run $BP/python -m behavior_prediction.plots

echo; echo ">>> ALL DONE (non-PB)"
