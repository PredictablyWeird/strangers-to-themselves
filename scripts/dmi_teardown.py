#!/usr/bin/env python3
"""Tear down a Dedicated-Endpoints-v2 (DMI) deployment + endpoint. Two-phase: `rm` on a live
deployment only scales it to zero, so the delete must be RETRIED until the API says gone
(memory: together-v1-endpoints-disabled.md). Verifies afterwards that the endpoint no longer
lists — a deployment left half-deleted bills per minute.

    python scripts/dmi_teardown.py <state-dir-with-together_endpoint.json>
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


def main() -> int:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    out = Path(sys.argv[1])
    rec = json.loads((out / "together_endpoint.json").read_text())
    if rec.get("deleted"):
        print(f"endpoint {rec['endpoint_id']} already deleted")
        return 0
    from together import Together
    c = Together(api_key=os.environ["TOGETHER_API_KEY"],
                 project_id=os.environ.get("TOGETHER_PROJECT_ID", "proj_CUK9bg9ceP2jGjDUnaxx5"))
    ep, dep = rec["endpoint_id"], rec.get("deployment_id")

    # SDK deployments.delete does NOT initiate scale-down (unlike `tg beta endpoints rm`, which
    # does), and a stopped deployment still 409s "referenced in endpoint traffic split" — so the
    # working sequence (verified 2026-08-17) is: scale the deployment to 0 replicas, wait for
    # STOPPED (GPUs released -> billing dead), then delete the ENDPOINT, which takes its
    # deployments with it.
    if dep:
        try:
            d = c.beta.endpoints.deployments.retrieve(dep, endpoint_id=ep)
            c.beta.endpoints.deployments.update(
                dep, endpoint_id=ep, etag=d.etag,
                autoscaling={"min_replicas": 0, "max_replicas": 0})
            print(f"deployment {dep}: scale-to-zero requested")
            for _ in range(40):
                d = c.beta.endpoints.deployments.retrieve(dep, endpoint_id=ep)
                s = str(d.status.state)
                if "STOPPED" in s or "FAILED" in s:
                    print(f"deployment {dep}: {s} (GPUs released)")
                    break
                time.sleep(15)
            else:
                sys.exit(f"deployment {dep} did not stop — it bills per minute, intervene by hand")
        except Exception as e:  # noqa: BLE001 — already gone is fine
            if "not found" not in str(e).lower():
                print(f"deployment stop: {str(e)[:150]}")

    for i in range(10):
        try:
            c.beta.endpoints.delete(ep)
            print(f"endpoint {ep} deleted (deployments removed with it)")
            break
        except Exception as e:  # noqa: BLE001
            if "not found" in str(e).lower():
                print(f"endpoint {ep} already gone")
                break
            print(f"  endpoint delete retry {i}: {str(e)[:100]}")
            time.sleep(10)

    live = []
    try:
        for e in c.beta.endpoints.list():
            live.append(getattr(e, "id", None))
    except Exception as exc:  # noqa: BLE001
        print("list check failed:", str(exc)[:120])
    if ep in live:
        sys.exit(f"endpoint {ep} STILL LISTED — verify by hand, it may bill")
    print(f"remaining v2 endpoints: {live or 'none'}")
    rec["deleted"] = True
    (out / "together_endpoint.json").write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
