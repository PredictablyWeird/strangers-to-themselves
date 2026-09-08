#!/usr/bin/env bash
# Add qwen3.7-plus-low as a full 4th model across all 4 active dev evals, then re-score everything.
# qwen already has discrimeval (66) + PB (6) targets; it's missing several method predictions there
# (llm_prediction, value settings, behavioral_sampling) and has no MMLU/sycophancy yet.
# Idempotent: bp-benchmark skips already-generated prediction files; measurement is guarded on targets.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; source .env; set +a
BP=.venv/bin

M=qwen3.7-plus-low
MODELS4=llama-3.3-70b,llama-4-maverick,deepseek-v4-flash-low,qwen3.7-plus-low
ALL4_EVALS=propensitybench,discrimeval,capability_mmlu,sycophancy_pushback

step() { echo; echo ">>> STEP: $*"; }
run()  { echo "+ $*"; "$@" || echo "ERROR ($?) in: $*"; }

# --- 1. Fill missing predictions on evals qwen already has targets for (cheaper; validates first) --
step "fill missing predictions — discrimeval + propensitybench / $M"
run $BP/bp-benchmark --evals discrimeval,propensitybench --models "$M" --phase generate

# --- 2. Measure + predict the two evals qwen doesn't have yet ------------------------------------
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

# --- 3. FINAL scoring across all 4 evals x 4 models (dev) ----------------------------------------
step "FINAL tune + evaluate + plots (all 4 evals, 4 models, dev)"
run $BP/bp-tune     --evals "$ALL4_EVALS" --models "$MODELS4"
run $BP/bp-evaluate --evals "$ALL4_EVALS" --models "$MODELS4" --split dev
run $BP/python -m behavior_prediction.plots

# --- 4. Rebuild the paper PDF so it reflects the refreshed tables/figures ------------------------
step "recompile paper PDF"
( cd paper && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/qwen_tex1.log 2>&1 \
            && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/tmp/qwen_tex2.log 2>&1 \
            && echo "paper compiled OK" || echo "ERROR: paper compile failed (see /tmp/qwen_tex2.log)" )

echo; echo ">>> ALL DONE (qwen add)"
