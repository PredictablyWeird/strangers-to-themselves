#!/usr/bin/env bash
# Canonical dev pass: re-tune all 8 active evals + dev evaluation + the full downstream
# report/figure/tex chain, producing the complete updated dev results in one run.
# ENTIRELY OFFLINE: no step calls a subject model (pool-derived predictor refresh, tuning,
# evaluation, exports, figures, LaTeX build all read cached files).
#
# Availability decision (CLAIMS_AUDIT §4, resolved 2026-08-14): list_experiment and
# oracle_xmm/oracle_xmm_learned STAY in the explicit tuning lists (existing files rescored,
# $0) so the paper's macros keep rendering; cutting them remains a review-time option.
# Method lists = each eval's prior tuning.json set ∪ {informed_sampling, protocol_report,
# protocol_sampling} (bp-tune skips non-applicable combos).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/stage7; mkdir -p "$LOGS"
EVALS=(sycophancy_pushback discrimeval capability_mmlu reward_hacking tau2_policy tau2_transfer propensitybench mask_subdomain_pressure)
ALL12=deepseek-v4-flash-low,gemini-3.1-flash-lite-low,gpt-5.4-nano-low,llama-3.3-70b,llama-4-maverick,qwen3.7-plus-low,claude-sonnet-5-off,claude-sonnet-5-low,gpt-5.5-off,gpt-5.5-low,deepseek-v4-pro-off,deepseek-v4-pro-low
SMALL6=deepseek-v4-flash-low,gemini-3.1-flash-lite-low,gpt-5.4-nano-low,llama-3.3-70b,llama-4-maverick,qwen3.7-plus-low
FAIL=0
step() { echo; echo "########## $1  ($(date +%H:%M:%S)) ##########"; }
run()  { "$@" || { echo "!!! STEP FAILED: $*"; FAIL=$((FAIL+1)); }; }

step "guard: frontier new-eval lanes all done"
for M in claude-sonnet-5-off claude-sonnet-5-low gpt-5.5-off gpt-5.5-low deepseek-v4-pro-off deepseek-v4-pro-low; do
  [ -f "logs/frontier_new_evals/$M.done" ] || { echo "ABORT: $M lane not done"; exit 1; }
done

step "backup tuning docs"
B=results/_backup/tuning_pre_stage7_$(date +%Y%m%d); mkdir -p "$B"
for e in "${EVALS[@]}"; do cp "results/$e/tuning.json" "$B/${e}_tuning.json"; done

step "offline refresh: pool-derived predictors on the 2 new evals (small pool ran with only 6 donors)"
run .venv/bin/bp-benchmark --phase generate --evals tau2_transfer,mask_subdomain_pressure \
  --models "$SMALL6" --methods cross_model_mean,report_mean,oracle_report_mean --fresh \
  > "$LOGS/pool_derived_refresh.log" 2>&1

step "offline refresh: oracle_xmm ensembles from current components (all 12 x 8 evals)"
run .venv/bin/bp-benchmark --phase generate --evals "$(IFS=,; echo "${EVALS[*]}")" \
  --models "$ALL12" --methods oracle_xmm,oracle_xmm_learned --fresh \
  > "$LOGS/ensemble_refresh.log" 2>&1

step "re-tune (explicit full method lists per eval)"
for e in "${EVALS[@]}"; do
  ML=$(.venv/bin/python - "$e" "$B" <<'PY'
import json, sys
prior = set(json.load(open(f"{sys.argv[2]}/{sys.argv[1]}_tuning.json"))["methods"])
prior |= {"informed_sampling", "protocol_report", "protocol_sampling"}
print(",".join(sorted(prior)))
PY
)
  echo "[tune $e] methods: $ML"
  run .venv/bin/bp-tune --evals "$e" --methods "$ML" >> "$LOGS/tune.log" 2>&1
done

step "canonical dev evaluation (writes evaluation.* + paper/generated)"
run .venv/bin/bp-evaluate --pool default > "$LOGS/evaluate_default.log" 2>&1

step "auxiliary pool reports (suffixed files only)"
run .venv/bin/bp-evaluate --pool finetunes > "$LOGS/evaluate_finetunes.log" 2>&1
run .venv/bin/bp-evaluate --pool gemmaresidpair --evals sycophancy_pushback,discrimeval,capability_mmlu,propensitybench > "$LOGS/evaluate_gemmaresidpair.log" 2>&1
run .venv/bin/bp-evaluate --pool all > "$LOGS/evaluate_all.log" 2>&1

step "downstream analyses, exports, figures"
for S in predictability_decomposition export_grain_macros frontier_selfspec learned_flip_sweep \
         tuning_sensitivity oracle_identity_transfer identity_transfer_figure hero_figure \
         spearman_robustness significance_tests generic_prior_analysis \
         pilot_protocol_report_score export_selfpred_paper; do
  run .venv/bin/python "scripts/$S.py" > "$LOGS/$S.log" 2>&1
done
run .venv/bin/python -m behavior_prediction.plots > "$LOGS/plots.log" 2>&1

step "LaTeX build check"
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex > ../"$LOGS"/latex.log 2>&1 ) \
  || { echo "!!! LaTeX build FAILED (see $LOGS/latex.log)"; FAIL=$((FAIL+1)); }
grep -i "undefined\|multiply" paper/main.log 2>/dev/null | head -10 || true

step "DONE — $FAIL step(s) failed"
exit 0
