#!/usr/bin/env bash
# Finalize results after the value-pruning + sycophancy-context change:
#   1) re-run sycophancy `value` (now [value-context, value-concrete]) for all 3 models and delete
#      the settings that were pruned (value / value-aggregate / value-detailed collapse to a
#      constant prompt on sycophancy);
#   2) PropensityBench phase (deepseek measured at cap 6; refresh llm_prediction on the llama PB cells);
#   3) final tune + evaluate + plots over all 4 evals x 3 models.
# Run AFTER run_results_resume.sh completes.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
BP=.venv/bin

MODELS3=llama-3.3-70b,llama-4-maverick,deepseek-v4-flash-low
ALL4_EVALS=propensitybench,discrimeval,capability_mmlu,sycophancy_pushback
PB_REPO=$(realpath ../propensity-evaluation)
PB_PY="$PB_REPO/.venv/bin/python"
PB_DESC="$PB_REPO/data/full"
# The harness strips a leading "/" and resolves --output_dir against its OWN cwd (the PB repo), so
# we must NOT pass an absolute --pb-run-out (that lands in a nested junk path). Let bp-sweep use its
# default relative dir (harness writes it under the PB repo) and read targets from there.
OUT_DIR="$PB_REPO/results/propensitybench/_harness/deepseek-v4-flash-low"

step() { echo; echo ">>> STEP: $*"; }
run()  { echo "+ $*"; "$@" || echo "ERROR ($?) in: $*"; }

# --- 1. Sycophancy value: drop pruned settings, regenerate the two that vary per subject ----------
step "sycophancy value: delete pruned (constant) settings + regenerate [value-context, value-concrete]"
for m in llama-3.3-70b llama-4-maverick deepseek-v4-flash-low; do
  d="results/sycophancy_pushback/$m"
  for s in value value-aggregate value-detailed; do
    rm -f "$d/predictions/$s.json" "$d/reasoning/$s.json" "$d/partial/$s.jsonl"
  done
  run $BP/bp-benchmark --evals sycophancy_pushback --models "$m" --methods value --phase generate --fresh
done

# --- 2. PropensityBench: measure deepseek (cap 6) + refresh llm_prediction on the llama PB cells ---
step "measure PropensityBench behavior — deepseek-v4-flash-low (cap 6)"
run $BP/bp-sweep --eval propensitybench --model deepseek-v4-flash-low \
    --pb-repo "$PB_REPO" --pb-python "$PB_PY" --pb-input data/full \
    --grain task_scenario --max-total-scenarios 6
step "extract PB targets — deepseek-v4-flash-low"
run $BP/bp-targets --eval propensitybench --model deepseek-v4-flash-low \
    --pb-output "$OUT_DIR" --pb-desc-dir "$PB_DESC" --grain task_scenario
step "predict all valid methods — propensitybench / deepseek-v4-flash-low"
run $BP/bp-benchmark --evals propensitybench --models deepseek-v4-flash-low --phase generate
step "llm_prediction refresh — existing PB cells (llama models)"
run $BP/bp-benchmark --evals propensitybench --models llama-3.3-70b,llama-4-maverick \
    --methods llm_prediction --phase generate --fresh

# --- 3. FINAL scoring across all 4 evals x 3 models (dev) -----------------------------------------
step "FINAL tune + evaluate + plots (all 4 evals, 3 models, dev)"
run $BP/bp-tune     --evals "$ALL4_EVALS" --models "$MODELS3"
run $BP/bp-evaluate --evals "$ALL4_EVALS" --models "$MODELS3" --split dev
run $BP/python -m behavior_prediction.plots

echo; echo ">>> ALL DONE (finalize)"
