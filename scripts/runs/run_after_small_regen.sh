#!/usr/bin/env bash
# Chain orchestrator (2026-08-12): once the small-pool sampling regeneration
# finishes, launch the three follow-on stages that were waiting on its rate budget and its
# fresh k=25 informed_sampling baselines:
#   1. frontier sampling regeneration        (scripts/runs/run_sampling_regen_frontier.sh)
#   2. protocol-informed appendix arms       (scripts/runs/run_protocol_arms.sh)
#   3. sampling budget ablation k=50 x r=2   (scripts/run_method_variant.py, side paths)
# The three run concurrently — they share the global elicitation rate ceiling either way.
# Waits on the lane markers in logs/sampling_regen_small/ (done OR failed per model, so a
# single failed lane cannot hang the chain; failures are reported, not silently absorbed).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
LOGS=logs/chain; mkdir -p "$LOGS"
SMALL=(deepseek-v4-flash-low gemini-3.1-flash-lite-low gpt-5.4-nano-low llama-3.3-70b llama-4-maverick qwen3.7-plus-low)

echo "[chain] waiting for small-pool sampling regen lanes..."
while :; do
  pending=0
  for M in "${SMALL[@]}"; do
    [ -f "logs/sampling_regen_small/$M.done" ] || [ -f "logs/sampling_regen_small/$M.failed" ] || pending=$((pending+1))
  done
  [ "$pending" -eq 0 ] && break
  sleep 300
done
for M in "${SMALL[@]}"; do
  [ -f "logs/sampling_regen_small/$M.failed" ] && echo "[chain] WARNING: small-regen lane FAILED for $M"
done
echo "[chain] small-pool regen finished at $(date +%H:%M:%S); launching follow-on stages"

bash scripts/runs/run_sampling_regen_frontier.sh > "$LOGS/frontier.log" 2>&1 &
FRONTIER=$!
bash scripts/runs/run_protocol_arms.sh > "$LOGS/protocol_arms.log" 2>&1 &
PROTOCOL=$!
{
  for M in llama-3.3-70b gpt-5.4-nano-low; do
    .venv/bin/python scripts/run_method_variant.py --method informed_sampling \
      --eval sycophancy_pushback --model "$M" --set k=50 \
      --out "results/_pilot/budget_k50_syc_$M.json" \
      || echo "ABLATION FAILED for $M"
  done
} > "$LOGS/ablation.log" 2>&1 &
ABLATION=$!

wait $FRONTIER; echo "[chain] frontier regen exited ($(date +%H:%M:%S))"
wait $PROTOCOL; echo "[chain] protocol arms exited ($(date +%H:%M:%S))"
wait $ABLATION; echo "[chain] ablation exited ($(date +%H:%M:%S))"
echo "[chain] ALL FOLLOW-ON STAGES FINISHED $(date +%H:%M:%S)"
tail -n 8 "$LOGS/frontier.log" "$LOGS/protocol_arms.log" "$LOGS/ablation.log"
