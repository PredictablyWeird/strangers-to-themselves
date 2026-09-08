#!/usr/bin/env bash
# Deploy v1, print the RAW inference response (no swallowed exceptions) to find why it returns empty,
# then tear down. Diagnostic only — does not run the binder collection.
set -uo pipefail
cd "$(dirname "$0")/../.."
set -a; . .env 2>/dev/null; set +a
V1_OUT=logs/selfpred_llama

teardown() { echo "=== teardown ==="; .venv/bin/python scripts/finetune_together.py teardown --out "$V1_OUT" || echo "!!! TEARDOWN FAILED !!!"; }
trap teardown EXIT

.venv/bin/python scripts/finetune_together.py deploy --out "$V1_OUT" \
  --hardware 4x_nvidia_h100_80gb_sxm --inactive-timeout 20 || exit 1

.venv/bin/python - "$V1_OUT" <<'PY'
import json, os, sys, time
from together import Together
rec = json.loads(open(os.path.join(sys.argv[1], "together_endpoint.json")).read())
c = Together(api_key=os.environ["TOGETHER_API_KEY"])
print("endpoint_name:", rec["endpoint_name"])
print("model id     :", rec["model"])
prompts = [("simple", "Reply with the single word: ok"),
           ("binderish", "You are given a hypothetical. Answer concisely.\nQuestion: Is the sky blue on a clear day? Answer yes or no.")]
for cand_label, cand in (("model_id", rec["model"]), ("endpoint_name", rec["endpoint_name"])):
    print(f"\n===== candidate: {cand_label} = {cand} =====")
    for mt in (5, 50):
        for plabel, p in prompts:
            for attempt in range(3):
                try:
                    r = c.chat.completions.create(model=cand,
                        messages=[{"role":"user","content":p}], temperature=0.0, max_tokens=mt)
                    ch = r.choices[0]
                    content = ch.message.content
                    fr = getattr(ch, "finish_reason", None)
                    print(f"  [{plabel} mt={mt}] OK content={content!r} finish={fr}")
                    break
                except Exception as e:
                    print(f"  [{plabel} mt={mt}] ERROR {type(e).__name__}: {str(e)[:160]}")
                    time.sleep(8)
PY
