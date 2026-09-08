#!/usr/bin/env bash
# Minimal end-to-end results pipeline: tune -> evaluate -> export (json + LaTeX macros) -> plots.
#
# Scoped down on purpose so it runs instantly on data already on disk and is trivially extensible:
# to widen the scope later, just add models/methods/evals to the lists below (or drop the --models
# flag to fall back to methods.yaml's full selection_pool).
#
# DEV SPLIT ONLY. The test split is the frozen, held-out generalization set. Do NOT add
# `--split test` here without EXPLICIT user instruction — bp-evaluate guards it behind --allow-test.
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS=llama-3.3-70b,llama-4-maverick
METHODS=self_report,value
EVALS=propensitybench,discrimeval

echo ">> tune (dev; pick best setting per method, scoped to the minimal model/method set)"
bp-tune     --evals "$EVALS" --models "$MODELS" --methods "$METHODS"

echo ">> evaluate (dev; emits evaluation.md/.html + evaluation.json + paper/generated/results.tex)"
bp-evaluate --evals "$EVALS" --models "$MODELS" --split dev

echo ">> plots (reads evaluation.json; writes figures to paper/generated/)"
python -m behavior_prediction.plots

echo ">> done. Machine-readable: results/reports/evaluation.json ; paper inputs: paper/generated/"
