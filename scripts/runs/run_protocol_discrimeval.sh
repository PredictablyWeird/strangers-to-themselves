#!/usr/bin/env bash
# Protocol-tier arms on DiscrimEval (2026-08-16): protocol_report + protocol_sampling for
# the small pool, dev split. Mirrors run_protocol_arms.sh; the first lane runs alone so the
# comparative generation cache is warmed once and every model reuses the SAME neutral case sets.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# tau2-bench clone (default: sibling of this repo). The tau2 oracle methods silently
# produce nothing if this is wrong, so it warns rather than failing late.
export TAU2_REPO="${TAU2_REPO:-$PWD/../tau2-bench}"
[ -d "$TAU2_REPO" ] || echo "WARN: TAU2_REPO=$TAU2_REPO does not exist; tau2 items will be empty"
LOGS=logs/protocol_discrimeval; mkdir -p "$LOGS"

MODELS=(llama-3.3-70b deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-4-maverick qwen3.7-plus-low)

one_model() {
  local M="$1"; local L="$LOGS/$M.log"; rm -f "$LOGS/$M.done" "$LOGS/$M.failed"
  local EXTRA=()
  if [[ "$M" == gemini-* ]]; then
    EXTRA=(--method-config protocol_sampling:grader_model=gpt-5.4-nano-low)
  fi
  {
    echo "=== [$M] START $(date +%H:%M:%S) ==="
    .venv/bin/bp-benchmark --phase generate --evals discrimeval --models "$M" \
      --methods protocol_report,protocol_sampling --concurrency 6 "${EXTRA[@]}" \
      || { echo "[$M] FAIL"; touch "$LOGS/$M.failed"; exit 1; }
    echo "=== [$M] DONE $(date +%H:%M:%S) ==="
    touch "$LOGS/$M.done"
  } > "$L" 2>&1
}

# Cache-warm lane first (generates the shared neutral case scripts once), then the rest parallel.
one_model "${MODELS[0]}"
for M in "${MODELS[@]:1}"; do one_model "$M" & done
wait

echo "############ PROTOCOL DISCRIMEVAL FINISHED $(date +%H:%M:%S) ############"
for M in "${MODELS[@]}"; do
  if   [ -f "$LOGS/$M.done" ];   then echo "  DONE   $M"
  elif [ -f "$LOGS/$M.failed" ]; then echo "  FAILED $M (see $LOGS/$M.log)"
  else echo "  ????   $M"; fi
done
