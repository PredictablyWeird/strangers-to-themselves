#!/usr/bin/env bash
# Resume the non-PB results after the third_person removal: finish the remaining deepseek cells.
# (llama-3.3-70b and llama-4-maverick non-PB cells are already complete; PB is a separate phase.)
# Idempotent: bp-benchmark skips already-generated prediction files; measurement is guarded on
# an existing targets.json so we never re-query behavior we already have.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
BP=.venv/bin

NONPB_EVALS=discrimeval,capability_mmlu,sycophancy_pushback
ALL_MODELS=llama-3.3-70b,llama-4-maverick,deepseek-v4-flash-low
M=deepseek-v4-flash-low

step() { echo; echo ">>> STEP: $*"; }
run()  { echo "+ $*"; "$@" || echo "ERROR ($?) in: $*"; }

# deepseek discrimeval: targets already exist -> just finish the remaining method predictions.
step "predict all valid methods — discrimeval / $M (resume)"
run $BP/bp-benchmark --evals discrimeval --models "$M" --phase generate

# deepseek capability_mmlu + sycophancy_pushback: measure (if needed) then predict.
for ev in capability_mmlu sycophancy_pushback; do
  if [ -f "results/$ev/$M/targets.json" ]; then
    step "measure behavior — $ev / $M (SKIP: targets.json exists)"
  else
    step "measure behavior — $ev / $M"
    run $BP/bp-sweep   --eval "$ev" --model "$M"
    run $BP/bp-targets --eval "$ev" --model "$M"
  fi
  step "predict all valid methods — $ev / $M"
  run $BP/bp-benchmark --evals "$ev" --models "$M" --phase generate
done

# Interim scoring over the non-PB evals (PB folded in later by run_results_pb.sh).
step "tune + evaluate + plots (non-PB, dev)"
run $BP/bp-tune     --evals "$NONPB_EVALS" --models "$ALL_MODELS"
run $BP/bp-evaluate --evals "$NONPB_EVALS" --models "$ALL_MODELS" --split dev
run $BP/python -m behavior_prediction.plots

echo; echo ">>> ALL DONE (non-PB resume)"
