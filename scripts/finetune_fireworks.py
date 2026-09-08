#!/usr/bin/env python3
"""Introspection self-prediction LoRA on Fireworks, for tool-calling evals (PropensityBench).

Companion to ``scripts/finetune_introspection.py`` (Tinker). Tinker is sampling-only and
can't be driven by PropensityBench's harness (litellm + OpenAI-style tool-calling), so for
PB we finetune on Fireworks, which serves the result behind an OpenAI-compatible tool-calling
endpoint that both the PB harness (litellm ``fireworks_ai/``) and inspect_ai (``fireworks/``
provider) reach.

Default base is Llama-3.3-70B-Instruct: we already have base-model results for it across the
evals, so the finetuned model compares directly against them. Self-prediction requires
training on the model's OWN behavior, so this collects Llama's object-level responses fresh
(the Qwen SFT data can't be reused) — from the SAME Fireworks deployment that will serve the
finetuned model, keeping the "self" consistent.

Stages (state under --out, gitignored):
  prepare   sample k train + eval_n eval rows over the 10 behavioral properties (shared with
            the Tinker script via import)
  collect   Fireworks temp-0 object-level responses on the base model; extract property targets
  data      convert to Fireworks CHAT jsonl; create + upload a dataset
  finetune  launch a LoRA SFT job and poll to completion
  deploy    on-demand deployment for the finetuned model (a 70B LoRA isn't served serverless;
            the base model is)

Env (from .env): FIREWORKS_API_KEY — one key serves all three consumers (this script,
inspect_ai's `fireworks/` provider, the PB harness's litellm `fireworks_ai/`). The account is
auto-discovered from the key; override with FIREWORKS_ACCOUNT_ID if the key sees several.

Usage:
  .venv/bin/python scripts/finetune_fireworks.py prepare --k 30000 --eval-n 500
  .venv/bin/python scripts/finetune_fireworks.py collect
  .venv/bin/python scripts/finetune_fireworks.py data
  .venv/bin/python scripts/finetune_fireworks.py finetune
  .venv/bin/python scripts/finetune_fireworks.py deploy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_introspection as intro  # noqa: E402  (shared prepare + property extraction)

API_ROOT = "https://api.fireworks.ai/v1"
INFERENCE_BASE = "https://api.fireworks.ai/inference/v1"
BASE_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
DATASET_ID = "introspection-llama-k30k"
OUTPUT_MODEL = "introspection-llama-k30k"


# --- config / auth -----------------------------------------------------------------------

def _key() -> str:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        sys.exit("FIREWORKS_API_KEY is not set (add it to .env)")
    return key


def _account(key: str) -> str:
    account = os.environ.get("FIREWORKS_ACCOUNT_ID")
    if account:
        return account
    r = requests.get(f"{API_ROOT}/accounts", headers={"Authorization": f"Bearer {key}"},
                     timeout=30)
    r.raise_for_status()
    accounts = r.json().get("accounts", [])
    if len(accounts) != 1:
        sys.exit(f"expected 1 account, found {len(accounts)}; set FIREWORKS_ACCOUNT_ID")
    return accounts[0]["name"].split("/", 1)[1]  # "accounts/<slug>" -> "<slug>"


def _headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# --- stages ------------------------------------------------------------------------------

def stage_collect(out: Path, model: str, workers: int) -> None:
    """Temp-0 object-level responses via the Fireworks OpenAI-compatible endpoint; extract
    property targets; write SFT pairs. Resumable (caches per (property, row_idx))."""
    from openai import OpenAI

    rows = intro.load_jsonl(out / "train_rows.jsonl") + intro.load_jsonl(out / "eval_rows.jsonl")
    if not rows:
        sys.exit("run `prepare` first")
    done_path = out / "object_level.jsonl"
    done = {(d["property"], d["row_idx"]) for d in intro.load_jsonl(done_path)}
    todo = [r for r in rows if (r["property"], r["row_idx"]) not in done]
    print(f"{len(todo)} object-level prompts to collect ({len(done)} cached)")

    if todo:
        client = OpenAI(api_key=_key(), base_url=INFERENCE_BASE)

        def worker(row: dict) -> dict:
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": row["object_level_prompt"]}],
                        temperature=0.0, max_tokens=200)
                    text = resp.choices[0].message.content or ""
                    return {"property": row["property"], "row_idx": row["row_idx"],
                            "response": text,
                            "target": intro.extract_property(row["property"], text, row)}
                except Exception:  # noqa: BLE001 — retry transient API errors
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))

        with done_path.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
            for i, rec in enumerate(pool.map(worker, todo), 1):
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}")

    records = intro.load_jsonl(done_path)
    n_bad = sum(1 for r in records if r["target"] is None)
    print(f"collected {len(records)} responses ({n_bad} without extractable target, dropped)")

    targets = {(r["property"], r["row_idx"]): r["target"] for r in records}
    sft = []
    for row in intro.load_jsonl(out / "train_rows.jsonl"):
        target = targets.get((row["property"], row["row_idx"]))
        if target is not None:
            sft.append({"prompt": row["hypothetical_prompt"], "completion": target})
    intro.save_jsonl(sft, out / "sft_train.jsonl")
    print(f"wrote {len(sft)} SFT pairs to {out / 'sft_train.jsonl'}")


def stage_data(out: Path) -> None:
    key = _key()
    account = _account(key)
    pairs = intro.load_jsonl(out / "sft_train.jsonl")
    if not pairs:
        sys.exit("run `collect` first")
    chat_path = out / "sft_train_fireworks.jsonl"
    intro.save_jsonl([{"messages": [
        {"role": "user", "content": p["prompt"]},
        {"role": "assistant", "content": p["completion"]},
    ]} for p in pairs], chat_path)
    print(f"wrote {len(pairs)} CHAT examples to {chat_path}")

    base = f"{API_ROOT}/accounts/{account}"
    r = requests.post(f"{base}/datasets", headers=_headers(key), json={
        "datasetId": DATASET_ID,
        "dataset": {"displayName": DATASET_ID, "exampleCount": str(len(pairs)),
                    "format": "CHAT"},
    }, timeout=60)
    if r.status_code not in (200, 409):  # 409 = already exists, fine to re-upload
        r.raise_for_status()
    print(f"dataset {DATASET_ID}: {'exists' if r.status_code == 409 else 'created'}")

    with chat_path.open("rb") as fh:
        r = requests.post(f"{base}/datasets/{DATASET_ID}:upload",
                          headers={"Authorization": f"Bearer {key}"},
                          files={"file": (chat_path.name, fh)}, timeout=600)
    r.raise_for_status()
    print(f"uploaded {chat_path.name} to dataset {DATASET_ID}")


def stage_finetune(epochs: int, lora_rank: int, lr: float) -> None:
    key = _key()
    account = _account(key)
    r = requests.post(f"{API_ROOT}/accounts/{account}/supervisedFineTuningJobs",
                      headers=_headers(key), json={
                          "dataset": DATASET_ID, "baseModel": BASE_MODEL,
                          "outputModel": OUTPUT_MODEL, "epochs": epochs,
                          "learningRate": lr, "loraRank": lora_rank,
                      }, timeout=60)
    r.raise_for_status()
    job_name = r.json()["name"]  # accounts/<acct>/supervisedFineTuningJobs/<id>
    print(f"job {job_name} created; polling...")

    while True:
        r = requests.get(f"{API_ROOT}/{job_name}", headers=_headers(key), timeout=60)
        r.raise_for_status()
        state = r.json().get("state", "UNKNOWN")
        print(f"  state: {state}")
        if any(t in state for t in ("COMPLET", "FAIL", "CANCEL")):
            break
        time.sleep(60)
    model = f"accounts/{account}/models/{OUTPUT_MODEL}"
    print(f"final state: {state}; finetuned model: {model}")
    if "COMPLET" in state:
        print("next: `finetune_fireworks.py deploy` (a 70B LoRA needs an on-demand deployment)")


def stage_deploy() -> None:
    key = _key()
    account = _account(key)
    model = f"accounts/{account}/models/{OUTPUT_MODEL}"
    r = requests.post(f"{API_ROOT}/accounts/{account}/deployments", headers=_headers(key),
                      json={"deployment": {"displayName": OUTPUT_MODEL, "baseModel": model}},
                      timeout=60)
    r.raise_for_status()
    dep = r.json()
    print(f"deployment {dep.get('name')} created; state {dep.get('state')}")
    print("\nOnce READY, add to behavior_prediction/models.yaml:")
    print(f"  llama-3.3-70b-intro30k-fw:\n    model: fireworks/{model}")
    print("\nThen run PropensityBench on the finetuned model:")
    print("  bp-sweep --eval propensitybench --model llama-3.3-70b-intro30k-fw \\")
    print("    --pb-repo <clone> --pb-python <harness-venv> --pb-input data/full \\")
    print(f"    --pb-provider fireworks_ai --pb-model-name {model} \\")
    print("    --pb-api-base https://api.fireworks.ai/inference/v1 \\")
    print("    --grain task_scenario --max-total-scenarios 20")
    print("Compare against the existing base results/propensitybench/llama-3.3-70b/.")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "collect", "data", "finetune", "deploy"])
    ap.add_argument("--out", type=Path, default=Path("logs/introspection_finetune_llama30k"))
    ap.add_argument("--k", type=int, default=30000, help="train set size (prepare)")
    ap.add_argument("--eval-n", type=int, default=500, help="held-out eval size (prepare)")
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help="Fireworks base model queried in collect / tuned in finetune")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.stage == "prepare":
        intro.stage_prepare(args.out, args.k, args.eval_n)
    elif args.stage == "collect":
        stage_collect(args.out, args.base_model, args.workers)
    elif args.stage == "data":
        stage_data(args.out)
    elif args.stage == "finetune":
        stage_finetune(args.epochs, args.lora_rank, args.lr)
    elif args.stage == "deploy":
        stage_deploy()


if __name__ == "__main__":
    main()
