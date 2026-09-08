#!/usr/bin/env python3
"""Deploy a registered v2/DMI model (merged finetune checkpoint) on a fresh dedicated endpoint.

The tg CLI silently normalizes a private finetuned model to its BASE revision (verified
2026-08-17, logs/selfpred_deepseek/DMI_ATTEMPT.md), so this goes through the raw SDK with an
explicit model_id + model_revision_id and then VERIFIES the deployment's model_id matches
before declaring success. Writes together_endpoint.json into the state dir in the same shape
the v1 flow used (endpoint_name = the inference model string), plus the v2 ids that
scripts/dmi_teardown.py needs.

    python scripts/dmi_deploy.py <state-dir> <endpoint-name> <ml_id> <rv_id>

Endpoint names are immutable and single-use per live endpoint; teardown deletes the whole
endpoint, so re-deploys can reuse the same name.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

CONFIG = "cr_CdgwhfgugpYUNLFFE1TEA"  # sole certified V4-Flash-0731 config (2xB200 NVFP4)
PROJECT = "proj_CUK9bg9ceP2jGjDUnaxx5"


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    out, ep_name, ml, rv = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
    from together import Together
    c = Together(api_key=os.environ["TOGETHER_API_KEY"], project_id=PROJECT)

    ep = c.beta.endpoints.create(name=ep_name)
    print(f"endpoint {ep.id} ({ep.name})")
    dep = c.beta.endpoints.deployments.create(
        endpoint_id=ep.id, name=f"{ep_name.split('/')[-1]}-merged",
        model_id=ml, model_revision_id=rv, config_id=CONFIG,
        autoscaling={"min_replicas": 1, "max_replicas": 1, "scale_to_zero_window": "900s"})
    print(f"deployment {dep.id}")

    rec = {"endpoint_id": ep.id, "deployment_id": dep.id, "endpoint_name": ep.name,
           "model": f"{ml}/{rv} (merged, DMI v2)", "v2": True, "deleted": False}
    (out / "together_endpoint.json").write_text(json.dumps(rec, indent=2))

    last = ""
    for _ in range(60):  # ~20 min budget for weight loading
        d = c.beta.endpoints.deployments.retrieve(dep.id, endpoint_id=ep.id)
        st = d.status
        cur = f"{getattr(st, 'state', st)} | {getattr(st, 'message', '')}"
        if cur != last:
            print(f"  {time.strftime('%H:%M:%S')} {cur}", flush=True)
            last = cur
        s = str(getattr(st, "state", ""))
        if "READY" in s:
            got = getattr(d, "model_id", None) or getattr(d, "modelId", None)
            if got != ml:
                sys.exit(f"DEPLOYMENT SERVES WRONG MODEL: {got} != {ml} — tear down NOW")
            print(f"READY; verified model_id == {ml}. Inference model string: {ep.name}")
            return 0
        if "FAILED" in s or "ERROR" in s:
            sys.exit(f"deployment entered {s} — tear down and investigate")
        time.sleep(20)
    sys.exit("deployment not READY after 20min — tear down and investigate")


if __name__ == "__main__":
    raise SystemExit(main())
