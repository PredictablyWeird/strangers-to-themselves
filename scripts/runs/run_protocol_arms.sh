#!/usr/bin/env bash
# Protocol-informed appendix arms, small pool only (2026-08-12):
#  - protocol_report top-up on the two evals it was never run on (tau2_transfer,
#    mask_subdomain_pressure; the other in-scope evals already have files; discrimeval is out
#    of scope by design).
#  - protocol_sampling (NEW pilot: informed_sampling with a protocol-only generation prompt)
#    on all 7 in-scope evals.
# Canonical prediction paths (predictions/<method>.json) — these are pilot methods excluded
# from DEFAULT_METHODS, so no default run touches them; the appendix scores them by name.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
# MMLU (sycophancy/capability oracle items) is fully cached; offline mode stops load_dataset
# phoning home — 18 parallel lanes hit the anonymous HF API 429 limit (2026-08-15).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/protocol_arms; mkdir -p "$LOGS"

MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)
SAMPLING_EVALS=sycophancy_pushback,capability_mmlu,reward_hacking,tau2_policy,tau2_transfer,propensitybench,mask_subdomain_pressure
REPORT_EVALS=tau2_transfer,mask_subdomain_pressure

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  local EXTRA=()
  if [[ "$M" == gemini-* ]]; then
    EXTRA=(--method-config protocol_sampling:grader_model=gpt-5.4-nano-low)
  fi
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals "$REPORT_EVALS" --models "$M" \
      --methods protocol_report --concurrency 6 || echo "[$M] WARN protocol_report"
    .venv/bin/bp-benchmark --phase generate --evals "$SAMPLING_EVALS" --models "$M" \
      --methods protocol_sampling --concurrency 6 "${EXTRA[@]}" \
      || { echo "[$M] FAIL protocol_sampling"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one_model "$M" & done
wait

echo "############ PROTOCOL ARMS FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
