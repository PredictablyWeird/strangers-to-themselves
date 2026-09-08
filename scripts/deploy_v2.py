#!/usr/bin/env python3
"""Deploy / tear down a fine-tuned model on a Together **Dedicated Endpoints v2** endpoint.

Together disabled Dedicated Endpoints v1 create for our org (~2026-07-15); v2 is the only path.
v1's single ``endpoints.create(model, hardware, autoscaling)`` is replaced by a multi-step flow:

    endpoint = client.beta.endpoints.create(name=...)                 # just a name (free, no GPU)
    client.beta.endpoints.deployments.create(                         # this is what launches GPUs
        endpoint_id=endpoint.id, name=..., model=<v2 model resource>,
        config=<base model's certified config resource>, enable_lora=True,
        autoscaling={min_replicas, max_replicas, scale_to_zero_window})
    ...run inference...
    client.beta.endpoints.delete(endpoint.id)                         # teardown

Requires ``together>=2.24.0`` and ``TOGETHER_API_KEY``. The org's numeric project id is discovered
via ``client.whoami()``.

IDLE AUTO-SHUTDOWN (safety net): every deployment is created with ``min_replicas=0`` and
``scale_to_zero_window="15m"``. Together then **automatically stops the deployment and releases its
GPUs after 15 minutes with no traffic** — so a crashed run or a skipped teardown cannot bill
indefinitely. ``teardown`` still deletes the endpoint explicitly; the idle window is the backstop.

PREREQUISITE — the fine-tune must exist as a v2 *model resource*. v1 fine-tune outputs
(``phblandfort_ac3d/…``) are NOT v2 model resources and ``beta.models.list()`` starts empty; deploy
one via the v2 UI (https://api.together.ai/endpoints/configure) once, which registers it, or await
the migration step from Together support. This script looks the model up by name substring in
``beta.models.list()`` and fails clearly if it is not registered yet.

    python scripts/deploy_v2.py deploy   --match selfpred-gemma
    python scripts/deploy_v2.py teardown --match selfpred-gemma
    python scripts/deploy_v2.py list
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

IDLE_WINDOW = "15m"   # scale_to_zero_window: GPUs released after this much idle time
STATE_DIR = Path("logs/endpoints_v2")


def _client():
    from together import Together
    load_dotenv()
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        raise SystemExit("TOGETHER_API_KEY is unset")
    c = Together(api_key=key)
    proj = c.whoami().project_id           # numeric project id, e.g. proj_...
    return Together(api_key=key, project_id=proj), proj


def _find_model(c, match: str):
    """The one registered v2 model whose name/id contains ``match`` (case-insensitive)."""
    models = list(c.beta.models.list())
    hits = [m for m in models
            if match.lower() in (getattr(m, "name", "") + getattr(m, "id", "")).lower()]
    if not hits:
        names = [getattr(m, "name", getattr(m, "id", "?")) for m in models]
        raise SystemExit(
            f"no registered v2 model matches {match!r}. Registered v2 models: {names or '(none)'}.\n"
            f"Register it by deploying once via https://api.together.ai/endpoints/configure, "
            f"or ask Together support for the v1->v2 model migration step.")
    if len(hits) > 1:
        raise SystemExit(f"{match!r} matches multiple models: "
                         f"{[getattr(m,'name',m.id) for m in hits]}; be more specific.")
    return hits[0]


def _certified_config(c, model) -> str:
    """The base architecture's certified deployment config resource for ``model``.

    A LoRA deploys on its base model's certified serving config (hardware + parallelism +
    quantization). We read it from the base arch's ``deployment_profiles`` rather than hardcoding
    GPU counts, so each family (gemma / llama / qwen) gets the right config automatically.
    """
    base_id = getattr(model, "base_model_id", None) or getattr(model, "baseModelId", None)
    if not base_id:
        raise SystemExit(f"model {getattr(model,'name',model.id)} has no base_model_id; "
                         f"cannot resolve a serving config.")
    sup = c.beta.models.retrieve_supported(id=base_id)
    profiles = getattr(sup, "deployment_profiles", None) or getattr(sup, "deploymentProfiles", None)
    if not profiles:
        raise SystemExit(f"base {base_id} exposes no deployment_profiles (certified config).")
    prof = profiles[0]
    cfg = getattr(prof, "config", None)
    print(f"  base {base_id}: {getattr(prof,'gpu_count','?')}x {getattr(prof,'gpu_type','?')} "
          f"({getattr(prof,'quantization','?')}), config {cfg}")
    return cfg


def deploy(match: str) -> None:
    c, proj = _client()
    model = _find_model(c, match)
    model_res = f"projects/{proj}/models/{model.id}"
    config = _certified_config(c, model)

    ep = c.beta.endpoints.create(name=f"bp-{match}".replace("_", "-").lower()[:60])
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / f"{match}.json").write_text(json.dumps(
        {"endpoint_id": ep.id, "model_id": model.id, "model_name": getattr(model, "name", "")}, indent=2))
    print(f"endpoint {ep.id} created for {getattr(model,'name',model.id)}")

    c.beta.endpoints.deployments.create(
        endpoint_id=ep.id, name="bp-dep", model=model_res, config=config, enable_lora=True,
        # min_replicas=0 + scale_to_zero_window => auto-stop after IDLE_WINDOW of no traffic.
        autoscaling={"min_replicas": 0, "max_replicas": 1, "scale_to_zero_window": IDLE_WINDOW})
    print(f"deployment created (idle auto-shutdown after {IDLE_WINDOW}). Polling for readiness…")

    for _ in range(120):                       # up to ~40 min; v2 bring-up is usually a few min
        e = c.beta.endpoints.retrieve(ep.id)
        state = str(getattr(e, "state", "?"))
        print(f"  state: {state}")
        if state.upper().endswith(("READY", "STARTED", "RUNNING")):
            break
        if "FAIL" in state.upper() or "ERROR" in state.upper():
            raise SystemExit(f"deployment entered {state}; tearing down manually recommended")
        time.sleep(20)
    print(f"endpoint {ep.id} ready. Inference-addressable name: {getattr(e,'name',ep.id)}")
    print(f"REMEMBER to `teardown --match {match}` when done (idle backstop is {IDLE_WINDOW}).")


def teardown(match: str) -> None:
    c, _ = _client()
    rec_path = STATE_DIR / f"{match}.json"
    ep_id = None
    if rec_path.exists():
        ep_id = json.loads(rec_path.read_text()).get("endpoint_id")
    if not ep_id:                              # fall back to a live-endpoint name match
        for e in c.beta.endpoints.list().data:
            if match.replace("_", "-").lower() in str(getattr(e, "name", "")).lower():
                ep_id = e.id
                break
    if not ep_id:
        print(f"no endpoint found for {match!r} — nothing to tear down")
        return
    c.beta.endpoints.delete(ep_id)
    if rec_path.exists():
        rec_path.unlink()
    live = [e.id for e in c.beta.endpoints.list().data]
    if ep_id in live:
        raise SystemExit(f"endpoint {ep_id} STILL LISTED after delete — it may still bill")
    print(f"deleted endpoint {ep_id}. Remaining v2 endpoints: {live or 'none'}")


def show_list() -> None:
    c, _ = _client()
    print("registered v2 models:")
    for m in c.beta.models.list():
        print(f"  {getattr(m,'id','?')}  {getattr(m,'name','')}")
    print("live v2 endpoints:")
    for e in c.beta.endpoints.list().data:
        print(f"  {e.id}  {getattr(e,'name','')}  {getattr(e,'state','')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["deploy", "teardown", "list"])
    ap.add_argument("--match", help="substring of the v2 model name/id to deploy or tear down")
    a = ap.parse_args()
    if a.stage == "list":
        show_list()
    elif not a.match:
        raise SystemExit("--match is required for deploy/teardown")
    elif a.stage == "deploy":
        deploy(a.match)
    else:
        teardown(a.match)


if __name__ == "__main__":
    main()
