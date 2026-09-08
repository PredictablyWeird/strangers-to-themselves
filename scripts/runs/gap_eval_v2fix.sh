#!/usr/bin/env bash
# Gap-correlation self-knowledge eval for the corrected v2fix (selfpred.tex Table 1). Deploy v2fix,
# elicit its rate self-predictions + re-elicit its behavior on the held-out categories, teardown, then
# score -> results/reports/selfpred_finetune_llama_v2fix.{md,json}. Base predictions are re-run
# serverless (cheap) so nothing stale is reused. Endpoint torn down on EXIT (bills per minute).
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
RUN=logs/selfpred_corpus_r2_fixed          # has together_job.json (v2fix) + splits copied here
EVALS_REPO=${EVALS_REPO:-$PWD/../anthropic-evals}
LOGS=logs/finetune_runs; mkdir -p "$LOGS"
R() { .venv/bin/python scripts/selfpred_report.py --evals-repo "$EVALS_REPO" --run "$RUN" "$@"; }

teardown() { echo "=== teardown v2fix (bills per minute) ==="; .venv/bin/python scripts/finetune_together.py teardown --out "$RUN" || echo "!!! TEARDOWN FAILED !!!"; }
trap teardown EXIT

echo "=== deploy v2fix on 4xH100 ==="
.venv/bin/python scripts/finetune_together.py deploy --out "$RUN" \
  --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 30 || exit 1
# serving check (openai-compat .xyz)
.venv/bin/python - "$RUN" <<'PY' || { echo "SERVING CHECK FAILED"; exit 1; }
import json,os,sys,time
from openai import OpenAI
rec=json.loads(open(os.path.join(sys.argv[1],"together_endpoint.json")).read())
c=OpenAI(api_key=os.environ["TOGETHER_API_KEY"],base_url="https://api.together.xyz/v1")
for _ in range(30):
    try:
        r=c.chat.completions.create(model=rec["endpoint_name"],
            messages=[{"role":"user","content":"Reply with: ok"}],temperature=0.0,max_tokens=5)
        if (r.choices[0].message.content or "").strip(): print("serves OK"); sys.exit(0)
    except Exception as e: print("  waiting:",str(e)[:70])
    time.sleep(10)
sys.exit("not serving")
PY

echo "=== predict base (serverless llama-3.3-70b-tg) ==="
R --stage predict --model base --base-shortcut llama-3.3-70b-tg || echo "WARN base predict"
echo "=== predict tuned (v2fix endpoint) ==="
R --stage predict --model tuned || echo "WARN tuned predict"
echo "=== drift: re-elicit v2fix behavior on held-out categories ==="
R --stage drift || echo "WARN drift"

echo "=== teardown before scoring (report needs no endpoint) ==="
teardown; trap - EXIT

echo "=== report -> selfpred_finetune_llama_v2fix ==="
R --stage report --out results/reports/selfpred_finetune_llama_v2fix.md || echo "WARN report"
echo "=== DONE. self_knowledge: ==="
.venv/bin/python -c "import json;d=json.load(open('results/reports/selfpred_finetune_llama_v2fix.json'));print(json.dumps(d.get('self_knowledge'),indent=1))"
