#!/usr/bin/env bash
# DeepSeek RESIDUAL-target selfpred eval phase on v2/DMI serving (merged checkpoint — see
# logs/selfpred_deepseek/DMI_ATTEMPT.md for why not v1/adapter). Mirrors
# selfpred_deepseek_resid_run.sh but deploys via scripts/dmi_deploy.py (raw SDK, model_id
# verified) and tears down via scripts/dmi_teardown.py (two-phase rm). Endpoint on EXIT trap.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_deepseek_resid
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
SHORTCUT=deepseek-v4-flash-selfpred-resid-tg
# Together model/revision ids for this finetune (account-scoped; override for your own job).
ML=${ML:?set ML to the Together model id (ml_...)}
RV=${RV:?set RV to the Together model revision id (rv_...)}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"

[ -f "$RUN/pred_base.jsonl" ] || { echo "run deepseek_base_coverage.sh first (no pred_base)"; exit 1; }

teardown() {
  echo "=== teardown $SHORTCUT v2 endpoint (bills per minute) ==="
  .venv/bin/python scripts/dmi_teardown.py "$RUN" || \
    echo "!!! TEARDOWN FAILED — delete deployment+endpoint by hand (tg beta endpoints rm) !!!"
}
trap teardown EXIT

echo "=== deploy merged resid model (v2/DMI, model_id-verified) ==="
.venv/bin/python scripts/dmi_deploy.py "$RUN" bp-ds-spresid "$ML" "$RV" || exit 1
.venv/bin/python scripts/set_model_string.py "$SHORTCUT" "$RUN" || exit 1
EP=$(.venv/bin/python -c "import json;print(json.load(open('$RUN/together_endpoint.json'))['endpoint_name'])")

echo "=== behavioral check: tuned vs base on training prompts (thinking off) ==="
.venv/bin/python - "$RUN" "$EP" <<'PY' || { echo "TUNED MODEL LOOKS DEGENERATE — stopping"; exit 1; }
import json, os, sys, itertools
from openai import OpenAI
c = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1")
eb = {"chat_template_kwargs": {"thinking": False}}
rows = [json.loads(l) for l in itertools.islice(open(os.path.join(sys.argv[1], "sft_train.jsonl")), 0, 2000)][::250][:8]
def ask(model, prompt):
    r = c.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}],
                                  temperature=0.0, max_tokens=8, extra_body=eb)
    return (r.choices[0].message.content or "").strip()
diff = empty = 0
for row in rows:
    t = ask(sys.argv[2], row["prompt"]); b = ask("deepseek-ai/DeepSeek-V4-Flash-0731", row["prompt"])
    diff += (t != b); empty += (not t)
    print(f"target={row['completion']!r:10s} tuned={t!r:12s} base={b!r:12s} {'DIFF' if t!=b else 'same'}")
print(f"{diff}/{len(rows)} differ; {empty} empty")
if empty >= 2: sys.exit("tuned outputs empty")
PY

echo "=== tuned ring predictions (both objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage predict \
  --model tuned --base-shortcut deepseek-v4-flash-off-tg --deviation --run "$RUN" --workers 16 || exit 1

for EV in sycophancy_pushback discrimeval capability_mmlu; do
  echo "=== measure behavior: $EV ==="
  .venv/bin/bp-sweep --eval "$EV" --model "$SHORTCUT" --concurrency 12 \
    > "$LOGS/${EV}_$SHORTCUT.log" 2>&1 && \
    .venv/bin/bp-targets --eval "$EV" --model "$SHORTCUT" >> "$LOGS/${EV}_$SHORTCUT.log" 2>&1 || \
    echo "WARN: $EV failed for $SHORTCUT (see $LOGS/${EV}_$SHORTCUT.log)"
done

echo "=== ask predictions (tuned settings) on the 3 evals ==="
.venv/bin/bp-benchmark --phase generate --tuned-only \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods self_report,value,pairwise,informed_oracle --concurrency 6 \
  > "$LOGS/pred_${SHORTCUT}_tuned.log" 2>&1 || echo "WARN: ask predictions failed"

echo "=== tuned behavior re-elicitation on the corpus (59,570 items — the long pass) ==="
.venv/bin/python scripts/build_selfpred_corpus.py --evals-repo "$EVALS_REPO" --stage elicit \
  --out "$RUN" --behavior-file behavior_tuned.jsonl \
  --model-name "$EP" --extra-body-from deepseek-v4-flash-off-tg --workers 16 || exit 1

echo "=== endpoint phase done — teardown before the offline passes ==="
teardown; trap - EXIT

echo "=== cross_model_mean (offline; no endpoint) ==="
.venv/bin/bp-benchmark --phase generate \
  --evals sycophancy_pushback,discrimeval,capability_mmlu \
  --models "$SHORTCUT" --methods cross_model_mean --concurrency 4 \
  > "$LOGS/pred_${SHORTCUT}_xmm.log" 2>&1 || echo "WARN: cross_model_mean failed"

echo "=== offline report (rate + deviation objectives) ==="
.venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --stage report \
  --run "$RUN" --behavior logs/selfpred_corpus_deepseek/behavior_deepseek.jsonl \
  --donor-behavior logs/selfpred_corpus/behavior_llama.jsonl \
                   logs/selfpred_corpus_qwen72/behavior_qwen72.jsonl \
  --out results/reports/selfpred_finetune_deepseek_resid.md || exit 1

echo "=== done ==="
