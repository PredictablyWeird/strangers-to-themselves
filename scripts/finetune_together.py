#!/usr/bin/env python3
"""Introspection self-prediction LoRA on Together AI, for tool-calling evals (PropensityBench).

Third backend after Tinker (sampling-only, no tool-calling) and Fireworks (Llama/Qwen both
deprecated there). Together serves Llama-3.3-70B with tool-calling AND serverless LoRA, and
we already have base-model results for llama-3.3-70b across the evals, so the finetuned model
compares directly against them.

Self-prediction requires training on the model's OWN behavior, so this collects Llama's
object-level responses fresh from Together's serverless Llama-3.3-70B (the Qwen SFT data can't
be reused). Reuses the shared prepare/extraction logic from finetune_introspection.py.

Stages (state under --out, gitignored):
  prepare   sample k train + eval_n eval rows over the 10 behavioral properties
  collect   Together temp-0 object-level responses on the base model; extract property targets
  finetune  upload CHAT jsonl + launch a LoRA SFT job on the Reference base; poll to completion

Env (from .env): TOGETHER_API_KEY (inspect_ai's `together/` provider and this SDK) — aliased
from TOGETHER_AI_API_KEY, which the PB harness's litellm `together_ai/` also reads.

Usage:
  .venv/bin/python scripts/finetune_together.py prepare --k 30000 --eval-n 500
  .venv/bin/python scripts/finetune_together.py collect
  .venv/bin/python scripts/finetune_together.py finetune
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_introspection as intro  # noqa: E402  (shared prepare + property extraction)

# Serverless FP8 deployment used to collect object-level behavior and to evaluate the base
# model (the finetuned LoRA is served serverless on the same FP8 Llama base).
INFER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# Finetuning base (Together's LoRA-tunable Reference weights).
TRAIN_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Reference"
SUFFIX = "introspection-llama-k30k"


# Chat inference must go through the OpenAI-compatible gateway at api.together.XYZ, not the SDK's
# default api.together.AI — the .ai route 404s "Model not found" for some serverless models
# (google/gemma-4-31B-it, verified 2026-08-15) and for dedicated-endpoint LoRAs (see
# build_selfpred_corpus.py). Non-chat stages (files/fine_tuning/endpoints) work on the default.
CHAT_GATEWAY = "https://api.together.xyz/v1"


def _client(base_url: str | None = None):
    from together import Together

    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        sys.exit("TOGETHER_API_KEY is not set (aliased from TOGETHER_AI_API_KEY in .env)")
    return Together(api_key=key, **({"base_url": base_url} if base_url else {}))


def stage_collect(out: Path, model: str, workers: int,
                  extra_body: dict | None = None) -> None:
    """Temp-0 object-level responses via Together; extract property targets; write SFT pairs.
    Resumable (caches per (property, row_idx)).

    ``extra_body`` carries provider passthrough — for Together's reasoning models the only way to
    get a direct answer is ``{"chat_template_kwargs": {"enable_thinking": false}}``; without it they
    spend the whole token budget thinking and return an empty string (e.g. gemma-4-31B)."""
    client = _client(CHAT_GATEWAY)
    extra_kw = {"extra_body": extra_body} if extra_body else {}
    rows = intro.load_jsonl(out / "train_rows.jsonl") + intro.load_jsonl(out / "eval_rows.jsonl")
    if not rows:
        sys.exit("run `prepare` first")
    done_path = out / "object_level.jsonl"
    done = {(d["property"], d["row_idx"]) for d in intro.load_jsonl(done_path)}
    todo = [r for r in rows if (r["property"], r["row_idx"]) not in done]
    print(f"{len(todo)} object-level prompts to collect ({len(done)} cached)")

    def worker(row: dict) -> dict:
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": row["object_level_prompt"]}],
                    temperature=0.0, max_tokens=200, **extra_kw)
                text = resp.choices[0].message.content or ""
                return {"property": row["property"], "row_idx": row["row_idx"],
                        "response": text,
                        "target": intro.extract_property(row["property"], text, row)}
            except Exception:  # noqa: BLE001 — retry transient API errors
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))

    if todo:
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


def stage_deploy(out: Path, hardware: str = "4x_nvidia_h100_80gb_sxm",
                 deploy_model: str | None = None, inactive_timeout: int = 120) -> None:
    """Serve a model on a dedicated endpoint.

    Together has no serverless LoRA, so a finetuned model needs a dedicated endpoint (~$0.18–0.72/min
    by hardware — run `teardown` as soon as you are done). The model string callers must use is the
    endpoint's NAME (which carries a fresh suffix each time it is created); passing the model id
    instead 400s with "non-serverless model".

    By default deploys the finetune output in ``together_job.json``. ``deploy_model`` overrides it —
    used to serve a *base* model whose serverless (Turbo) twin does not exist, so its own behavior can
    be collected for the self-prediction corpus (e.g. Qwen2.5-72B-Instruct)."""
    client = _client()
    model = deploy_model or json.loads((out / "together_job.json").read_text())["output_name"]
    # A short idle window silently STOPS the endpoint during any pause mid-run (e.g. between two
    # passes, or while a failed chunk is retried), after which every request hangs against a dead
    # endpoint — so ``inactive_timeout`` must stay comfortably longer than the longest expected gap
    # between calls. It is only a crash backstop anyway: the run driver's EXIT-trap ``teardown``
    # deletes the endpoint the instant a model finishes, so idle billing after normal completion is
    # zero regardless of this value. (Default 120; the driver passes 30 — long enough that a
    # continuous-load run never trips it, tight enough to cap damage from an unclean kill.)
    ep = client.endpoints.create(
        model=model, hardware=hardware,
        autoscaling={"min_replicas": 1, "max_replicas": 1},
        inactive_timeout=inactive_timeout, state="STARTED")
    rec = {"endpoint_id": ep.id, "endpoint_name": ep.name, "model": model, "deleted": False}
    (out / "together_endpoint.json").write_text(json.dumps(rec, indent=2))
    print(f"endpoint {ep.id} on {hardware}; call it as:\n  {ep.name}")
    while True:
        e = client.endpoints.retrieve(ep.id)
        print(f"  state: {e.state}")
        if str(e.state).upper() in ("STARTED", "ERROR", "FAILED"):
            break
        time.sleep(20)
    print(f"final state: {e.state}. REMEMBER: `teardown` when finished (billed per minute).")


def stage_teardown(out: Path) -> None:
    """Delete the dedicated endpoint. Idempotent."""
    client = _client()
    path = out / "together_endpoint.json"
    if not path.exists():
        print("no endpoint record — nothing to tear down")
        return
    rec = json.loads(path.read_text())
    if rec.get("deleted"):
        print(f"endpoint {rec['endpoint_id']} already deleted")
        return
    client.endpoints.delete(rec["endpoint_id"])
    rec["deleted"] = True
    path.write_text(json.dumps(rec, indent=2))
    print(f"deleted endpoint {rec['endpoint_id']}")

    # `mine=True`: an unfiltered list also returns Together's public dedicated endpoints, which we
    # neither own nor pay for — they would show up as spurious "remaining" entries.
    resp = client.endpoints.list(type="dedicated", mine=True)
    live = [e.id for e in (getattr(resp, "data", None) or [])]
    if rec["endpoint_id"] in live:
        sys.exit(f"endpoint {rec['endpoint_id']} STILL LIVE after delete — it bills per minute")
    print(f"our remaining dedicated endpoints: {live or 'none'}")


def stage_finetune(out: Path, base_model: str, epochs: int, lora_rank: int, lr: float,
                   suffix: str, from_checkpoint: str | None = None) -> None:
    client = _client()
    pairs = intro.load_jsonl(out / "sft_train.jsonl")
    if not pairs:
        sys.exit("run `collect` first")
    chat_path = out / "sft_train_together.jsonl"
    intro.save_jsonl([{"messages": [
        {"role": "user", "content": p["prompt"]},
        {"role": "assistant", "content": p["completion"]},
    ]} for p in pairs], chat_path)
    print(f"wrote {len(pairs)} CHAT examples to {chat_path}")

    up = client.files.upload(file=str(chat_path), purpose="fine-tune")
    file_id = up.id
    print(f"uploaded as {file_id}")

    # `from_checkpoint` continues a previous LoRA. Leave it unset for a Picard iteration of
    # G(M) = finetune(base, behavior(M)) — every round then shares one recipe and is comparable.
    kw = {"from_checkpoint": from_checkpoint} if from_checkpoint else {"model": base_model}
    job = client.fine_tuning.create(
        training_file=file_id, lora=True,
        n_epochs=epochs, lora_r=lora_rank, learning_rate=lr, suffix=suffix,
        train_on_inputs=False, **kw)  # loss on the assistant target only, not the prompt
    print(f"job {job.id} created; polling...")

    def record(status: str) -> None:
        name = getattr(job, "output_name", None) or getattr(job, "model_output_name", None)
        (out / "together_job.json").write_text(json.dumps(
            {"job_id": job.id, "status": status, "output_name": name,
             "base_model": base_model, "n_train": len(pairs)}, indent=2))

    record(str(getattr(job, "status", "submitted")))  # submission survives even if polling dies

    while True:
        job = client.fine_tuning.retrieve(job.id)
        status = str(getattr(job, "status", "unknown"))
        print(f"  status: {status}")
        if status.lower().split(".")[-1] in ("completed", "error", "failed", "cancelled"):
            break
        time.sleep(60)
    record(status)
    name = getattr(job, "output_name", None) or getattr(job, "model_output_name", None)
    print(f"final status: {status}; finetuned model: {name}")
    if name:
        print("\nAdd to behavior_prediction/models.yaml:")
        print(f"  llama-3.3-70b-intro30k-tg:\n    model: together/{name}")
        print("  llama-3.3-70b-tg:\n    model: together/meta-llama/Llama-3.3-70B-Instruct-Turbo")


def stage_evaluate(out: Path, model: str, workers: int,
                   extra_body: dict | None = None) -> None:
    """Held-out self-prediction accuracy: ask `model` the eval-split hypotheticals, score vs
    the collected targets + a per-property mode baseline. `model` is a Together model id
    (base serverless, or the finetuned LoRA via its dedicated endpoint). ``extra_body`` as in
    `collect` — must match the collection-time template flags (e.g. gemma's enable_thinking)."""
    import re
    from collections import Counter

    client = _client(CHAT_GATEWAY)
    eval_rows = intro.load_jsonl(out / "eval_rows.jsonl")
    targets = {(r["property"], r["row_idx"]): r["target"]
               for r in intro.load_jsonl(out / "object_level.jsonl")}
    rows = [r for r in eval_rows if targets.get((r["property"], r["row_idx"])) is not None]
    if not rows:
        sys.exit("run `prepare` and `collect` first")

    def ask(row: dict) -> str:
        for attempt in range(4):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": row["hypothetical_prompt"]}],
                    temperature=0.0, max_tokens=20,
                    **({"extra_body": extra_body} if extra_body else {}))
                return r.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        preds = list(pool.map(ask, rows))

    per_prop: dict[str, list[bool]] = {}
    mode_hits: dict[str, list[bool]] = {}
    mode = {p: Counter(intro.norm_answer(targets[(r["property"], r["row_idx"])])
                       for r in rows if r["property"] == p).most_common(1)[0][0]
            for p in {r["property"] for r in rows}}
    for row, pred in zip(rows, preds):
        truth = intro.norm_answer(targets[(row["property"], row["row_idx"])])
        per_prop.setdefault(row["property"], []).append(intro.norm_answer(pred) == truth)
        mode_hits.setdefault(row["property"], []).append(mode[row["property"]] == truth)

    result = {"model": model, "n": len(rows),
              "accuracy": sum(sum(v) for v in per_prop.values()) / len(rows),
              "mode_baseline": sum(sum(v) for v in mode_hits.values()) / len(rows),
              "per_property": {p: {"n": len(v), "accuracy": sum(v) / len(v),
                                   "mode_baseline": sum(mode_hits[p]) / len(v)}
                               for p, v in sorted(per_prop.items())}}
    slug = re.sub(r"[^a-zA-Z0-9.-]+", "-", model)[-80:]
    (out / f"eval_{slug}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("model", "n", "accuracy", "mode_baseline")}, indent=2))


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "collect", "finetune", "evaluate",
                                      "deploy", "teardown"])
    ap.add_argument("--out", type=Path, default=Path("logs/introspection_finetune_llama30k"))
    ap.add_argument("--suffix", default=SUFFIX, help="finetuned-model name suffix")
    ap.add_argument("--hardware", default="4x_nvidia_h100_80gb_sxm",
                    help="endpoint hardware; gemma-4-31B fits 2x_nvidia_h100_80gb_sxm ($0.18/min)")
    ap.add_argument("--deploy-model", default=None,
                    help="deploy this model id instead of the finetune output (serve a base model)")
    ap.add_argument("--inactive-timeout", type=int, default=120,
                    help="minutes of idle before the endpoint auto-stops (crash backstop; the run "
                         "driver passes 30). Must exceed the longest mid-run gap or the run breaks.")
    ap.add_argument("--from-checkpoint", default=None,
                    help="continue a previous LoRA instead of starting from --train-model")
    ap.add_argument("--k", type=int, default=30000)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--infer-model", default=INFER_MODEL, help="base model queried in collect")
    ap.add_argument("--extra-body", default=None,
                    help="JSON provider passthrough for collect, e.g. "
                         '\'{"chat_template_kwargs": {"enable_thinking": false}}\' '
                         "(required for gemma-4-31B — see models.yaml)")
    ap.add_argument("--model", default=INFER_MODEL, help="model scored in evaluate")
    ap.add_argument("--train-model", default=TRAIN_MODEL, help="LoRA finetune base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.stage == "prepare":
        intro.stage_prepare(args.out, args.k, args.eval_n)
    elif args.stage == "collect":
        stage_collect(args.out, args.infer_model, args.workers,
                      json.loads(args.extra_body) if args.extra_body else None)
    elif args.stage == "finetune":
        stage_finetune(args.out, args.train_model, args.epochs, args.lora_rank, args.lr,
                       args.suffix, args.from_checkpoint)
    elif args.stage == "evaluate":
        stage_evaluate(args.out, args.model, args.workers,
                       json.loads(args.extra_body) if args.extra_body else None)
    elif args.stage == "deploy":
        stage_deploy(args.out, args.hardware, args.deploy_model, args.inactive_timeout)
    elif args.stage == "teardown":
        stage_teardown(args.out)


if __name__ == "__main__":
    main()
