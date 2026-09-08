#!/usr/bin/env bash
# Generate tau2_policy informed_oracle for the 6 default models. Idempotent (bp-benchmark skips
# existing prediction files), so re-running only fills the ones not yet done. Writes a per-model
# STATUS line and a run-level DONE marker into a fresh timestamped run dir (no rm of any path).
set -uo pipefail
cd "$(dirname "$0")/../.."
export TAU2_REPO=${TAU2_REPO:-$PWD/../tau2-bench}
set -a; . .env 2>/dev/null; set +a
RUN="logs/tau2_informed_oracle/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "logs -> $RUN"
MODELS=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

one() {
  local M="$1"; local L="$RUN/$M.log"
  { echo "=== [$M] start $(date +%H:%M:%S) ==="
    if .venv/bin/bp-benchmark --phase generate --evals tau2_policy --models "$M" \
         --methods informed_oracle --concurrency 6; then
      echo "STATUS=DONE"
    else
      echo "STATUS=FAILED"
    fi
    echo "=== [$M] end $(date +%H:%M:%S) ==="
  } > "$L" 2>&1
}

for M in "${MODELS[@]}"; do one "$M" & done
wait

{
  echo "############ tau2 informed_oracle gen finished $(date +%H:%M:%S) ############"
  for M in "${MODELS[@]}"; do
    st=$(grep -o 'STATUS=[A-Z]*' "$RUN/$M.log" 2>/dev/null | tail -1)
    n=$(ls "results/tau2_policy/$M/predictions/informed_oracle.json" 2>/dev/null | wc -l)
    printf "  %-28s %-14s informed_oracle.json:%s\n" "$M" "$st" "$n"
  done
} | tee "$RUN/summary.txt"
