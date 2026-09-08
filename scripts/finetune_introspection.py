#!/usr/bin/env python3
"""Finetune a model for introspection (self-prediction) on Tinker, a la Binder et al. 2024.

Replicates the paper's self-prediction training (arxiv 2410.13787) at small scale:
reuse the hypothetical questions from their released dataset
(HF: thejaminator/introspection_self_predict), but regenerate the targets from the
target model's OWN temperature-0 behavior, then LoRA-finetune via the Tinker API.

Standalone experiment runner — does NOT touch the package's method registry or results/.

Note on the model: Tinker retired all Llama models (incl. Llama 3.3 70B) on 2026-06-12,
so the default base is Qwen/Qwen3-30B-A3B-Instruct-2507 — similar size and, unlike the
Qwen3.5/3.6 lines, a non-thinking model (no <think> blocks). Swap via --base-model; the
enable_thinking=False template kwarg is passed everywhere so thinking-family Qwen models
also work (they get an empty <think> block and answer directly).

Stages (each resumable; state lives in --out, gitignored under logs/):
  prepare   sample k train + n eval rows, stratified over the 10 behavioral properties
  collect   get the model's temp-0 answers to the object-level prompts, extract properties
  finetune  build masked-prompt SFT datums (hypothetical -> extracted property), train LoRA
  evaluate  score hypothetical accuracy on the eval split (base model, 'tuned', or a
            tinker:// weights path) against the collected ground truth + mode baseline

Usage (TINKER_API_KEY is read from .env):
  .venv/bin/python scripts/finetune_introspection.py prepare --k 1000 --eval-n 200
  .venv/bin/python scripts/finetune_introspection.py collect
  .venv/bin/python scripts/finetune_introspection.py finetune
  .venv/bin/python scripts/finetune_introspection.py evaluate            # base model
  .venv/bin/python scripts/finetune_introspection.py evaluate --model tuned
"""
from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

DATASET = "thejaminator/introspection_self_predict"
PROPERTIES = [
    "among_a_or_c", "among_b_or_d", "ethical_stance",
    "first_character", "second_character", "third_character",
    "first_word", "second_word", "third_word", "starts_with_vowel",
]

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
VOWELS = set("aeiou")


# --- property extraction: model's object-level response -> training target --------------

def _option_letter(text: str) -> str | None:
    m = re.search(r"[A-D]", text.strip().upper())
    return m.group(0) if m else None


def extract_property(prop: str, response: str, row: dict) -> str | None:
    """Return the target answer string, or None if the response doesn't support it."""
    text = response.strip()
    if not text:
        return None
    words = text.split()
    try:
        if prop == "first_character":
            return text[0]
        if prop == "second_character":
            return text[1]
        if prop == "third_character":
            return text[2]
        if prop == "first_word":
            return words[0]
        if prop == "second_word":
            return words[1]
        if prop == "third_word":
            return words[2]
    except IndexError:
        return None
    if prop == "starts_with_vowel":
        return "true" if text[0].lower() in VOWELS else "false"
    letter = _option_letter(text)
    if letter is None:
        return None
    if prop == "among_a_or_c":
        return "true" if letter in ("A", "C") else "false"
    if prop == "among_b_or_d":
        return "true" if letter in ("B", "D") else "false"
    if prop == "ethical_stance":
        match = row.get("option_matching_ethical_stance")
        return None if match is None else ("true" if letter == match.strip().upper() else "false")
    raise ValueError(f"unknown property: {prop}")


def norm_answer(text: str) -> str:
    return text.strip().strip(string.punctuation + "'\"").lower()


# --- tinker helpers ----------------------------------------------------------------------

def make_sampling_client(model: str, state_path: Path):
    """`model` is a base-model name, a tinker:// weights path, or the literal 'tuned'."""
    import tinker

    if model == "tuned":
        if not state_path.exists():
            sys.exit("no finetune state found — run `finetune` first")
        model = json.loads(state_path.read_text())["model_path"]
    sc = tinker.ServiceClient()
    if model.startswith("tinker://"):
        return sc.create_sampling_client(model_path=model), model
    return sc.create_sampling_client(base_model=model), model


def render_user_prompt(tok, prompt: str) -> list[int]:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=True,
                                   return_dict=False, enable_thinking=False)


def sample_batch(client, tok, prompts: list[str], max_tokens: int, window: int,
                 on_result=None) -> list[str]:
    """Temp-0 samples for all prompts, `window` concurrent requests, 2 retries each."""
    from tinker import types

    out: list[str | None] = [None] * len(prompts)
    for start in range(0, len(prompts), window):
        chunk = list(enumerate(prompts))[start:start + window]
        futures = {i: client.sample(
            prompt=types.ModelInput.from_ints(render_user_prompt(tok, p)),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=0.0),
        ) for i, p in chunk}
        for i, fut in futures.items():
            for attempt in range(3):
                try:
                    seq = fut.result().sequences[0]
                    out[i] = tok.decode(seq.tokens, skip_special_tokens=True)
                    break
                except Exception as e:  # noqa: BLE001 — retry transient API errors
                    if attempt == 2:
                        raise
                    time.sleep(2 ** (attempt + 1))
                    fut = client.sample(
                        prompt=types.ModelInput.from_ints(render_user_prompt(tok, prompts[i])),
                        num_samples=1,
                        sampling_params=types.SamplingParams(max_tokens=max_tokens,
                                                             temperature=0.0))
            if on_result is not None:
                on_result(i, out[i])
        done = min(start + window, len(prompts))
        if done < len(prompts):
            print(f"  {done}/{len(prompts)}")
    return out  # type: ignore[return-value]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- stages ------------------------------------------------------------------------------

def _waterfill(capacity: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` as evenly as capacities allow (small configs capped, rest topped up)."""
    alloc: dict[str, int] = {}
    remaining = total
    items = sorted(capacity.items(), key=lambda kv: kv[1])
    for i, (cfg, cap) in enumerate(items):
        take = min(cap, remaining // (len(items) - i))
        alloc[cfg] = take
        remaining -= take
    return alloc


def stage_prepare(out: Path, k: int, eval_n: int) -> None:
    """Load k train + eval_n rows from the HF dataset, spread over the property configs
    as evenly as their sizes allow (ethical_stance has only 841 rows)."""
    from datasets import load_dataset  # bulk parquet download; the rows API rate-limits

    dsets = {cfg: load_dataset(DATASET, cfg, split="test") for cfg in PROPERTIES}
    sizes = {cfg: len(d) for cfg, d in dsets.items()}
    if k + eval_n > sum(sizes.values()):
        sys.exit(f"asked for {k + eval_n} rows but only {sum(sizes.values())} exist")
    eval_alloc = _waterfill(sizes, eval_n)
    train_alloc = _waterfill({c: sizes[c] - eval_alloc[c] for c in sizes}, k)

    train, eval_ = [], []
    for cfg in PROPERTIES:
        need = train_alloc[cfg] + eval_alloc[cfg]
        rows = [{"property": cfg, "row_idx": i, **dsets[cfg][i]} for i in range(need)]
        train.extend(rows[:train_alloc[cfg]])
        eval_.extend(rows[train_alloc[cfg]:need])
        print(f"{cfg}: {train_alloc[cfg]} train, {eval_alloc[cfg]} eval (of {sizes[cfg]})")
    save_jsonl(train, out / "train_rows.jsonl")
    save_jsonl(eval_, out / "eval_rows.jsonl")
    print(f"wrote {len(train)} train + {len(eval_)} eval rows to {out}")


def stage_collect(out: Path, model: str, window: int) -> None:
    """Temp-0 object-level responses for all rows; extract property targets."""
    rows = load_jsonl(out / "train_rows.jsonl") + load_jsonl(out / "eval_rows.jsonl")
    if not rows:
        sys.exit("run `prepare` first")
    done_path = out / "object_level.jsonl"
    done = {(d["property"], d["row_idx"]) for d in load_jsonl(done_path)}
    todo = [r for r in rows if (r["property"], r["row_idx"]) not in done]
    print(f"{len(todo)} object-level prompts to collect ({len(done)} cached)")

    if todo:
        client, _ = make_sampling_client(model, out / "tinker_state.json")
        tok = client.get_tokenizer()
        with done_path.open("a") as fh:
            def write(i: int, resp: str) -> None:
                row = todo[i]
                fh.write(json.dumps({
                    "property": row["property"], "row_idx": row["row_idx"],
                    "response": resp,
                    "target": extract_property(row["property"], resp, row),
                }, ensure_ascii=False) + "\n")
                fh.flush()

            sample_batch(client, tok, [r["object_level_prompt"] for r in todo],
                         max_tokens=200, window=window, on_result=write)

    records = load_jsonl(done_path)
    n_bad = sum(1 for r in records if r["target"] is None)
    print(f"collected {len(records)} responses ({n_bad} without extractable target, dropped)")

    targets = {(r["property"], r["row_idx"]): r["target"] for r in records}
    sft = []
    for row in load_jsonl(out / "train_rows.jsonl"):
        target = targets.get((row["property"], row["row_idx"]))
        if target is not None:
            sft.append({"prompt": row["hypothetical_prompt"], "completion": target})
    save_jsonl(sft, out / "sft_train.jsonl")
    print(f"wrote {len(sft)} SFT pairs to {out / 'sft_train.jsonl'}")


def stage_finetune(out: Path, base_model: str, rank: int, epochs: int, lr: float,
                   batch_size: int, suffix: str) -> None:
    import tinker
    from tinker import types

    pairs = load_jsonl(out / "sft_train.jsonl")
    if not pairs:
        sys.exit("run `collect` first")
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=base_model, rank=rank)
    tok = tc.get_tokenizer()

    datums, n_skipped = [], 0
    for p in pairs:
        gen = render_user_prompt(tok, p["prompt"])
        full = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]},
             {"role": "assistant", "content": p["completion"]}],
            tokenize=True, return_dict=False, enable_thinking=False)
        if full[:len(gen)] != gen:
            n_skipped += 1
            continue
        datums.append(types.Datum(
            model_input=types.ModelInput.from_ints(full[:-1]),
            loss_fn_inputs={
                "weights": [0.0] * (len(gen) - 1) + [1.0] * (len(full) - len(gen)),
                "target_tokens": full[1:],
            }))
    print(f"built {len(datums)} datums ({n_skipped} skipped: template prefix mismatch)")

    rng = random.Random(0)
    step = 0
    for epoch in range(epochs):
        rng.shuffle(datums)
        for i in range(0, len(datums), batch_size):
            batch = datums[i:i + batch_size]
            fb = tc.forward_backward(batch, "cross_entropy")
            opt = tc.optim_step(types.AdamParams(learning_rate=lr))
            metrics = fb.result().metrics
            opt.result()
            step += 1
            loss = {k: round(float(v), 4) for k, v in metrics.items() if "loss" in k}
            print(f"epoch {epoch + 1}/{epochs} step {step}: {loss or metrics}")

    path = tc.save_weights_for_sampler(name=suffix).result().path
    state = {"base_model": base_model, "rank": rank, "epochs": epochs, "lr": lr,
             "n_train": len(datums), "model_path": path}
    (out / "tinker_state.json").write_text(json.dumps(state, indent=2))
    print(f"saved weights: {path}")
    print("evaluate with: evaluate --model tuned")


def stage_evaluate(out: Path, model: str, window: int) -> None:
    """Ask `model` the eval-split hypotheticals; score against extracted ground truth."""
    eval_rows = load_jsonl(out / "eval_rows.jsonl")
    targets = {(r["property"], r["row_idx"]): r["target"]
               for r in load_jsonl(out / "object_level.jsonl")}
    rows = [r for r in eval_rows if targets.get((r["property"], r["row_idx"])) is not None]
    if not rows:
        sys.exit("run `prepare` and `collect` first")

    client, resolved = make_sampling_client(model, out / "tinker_state.json")
    tok = client.get_tokenizer()
    preds = sample_batch(client, tok, [r["hypothetical_prompt"] for r in rows],
                         max_tokens=20, window=window)

    per_prop: dict[str, list[bool]] = {}
    mode_hits: dict[str, list[bool]] = {}
    mode = {p: Counter(norm_answer(targets[(r["property"], r["row_idx"])])
                       for r in rows if r["property"] == p).most_common(1)[0][0]
            for p in {r["property"] for r in rows}}
    for row, pred in zip(rows, preds):
        truth = norm_answer(targets[(row["property"], row["row_idx"])])
        per_prop.setdefault(row["property"], []).append(norm_answer(pred) == truth)
        mode_hits.setdefault(row["property"], []).append(mode[row["property"]] == truth)

    result = {"model": resolved, "n": len(rows),
              "accuracy": sum(sum(v) for v in per_prop.values()) / len(rows),
              "mode_baseline": sum(sum(v) for v in mode_hits.values()) / len(rows),
              "per_property": {p: {"n": len(v), "accuracy": sum(v) / len(v),
                                   "mode_baseline": sum(mode_hits[p]) / len(v)}
                               for p, v in sorted(per_prop.items())}}
    slug = re.sub(r"[^a-zA-Z0-9.-]+", "-", resolved)[-80:]
    path = out / f"eval_{slug}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved to {path}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["prepare", "collect", "finetune", "evaluate"])
    ap.add_argument("--out", default="logs/introspection_finetune", type=Path)
    ap.add_argument("--k", type=int, default=1000, help="train set size (prepare)")
    ap.add_argument("--eval-n", type=int, default=200, help="held-out eval size (prepare)")
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help="Tinker base model (collect object-level behavior + LoRA training)")
    ap.add_argument("--model", default=None,
                    help="model scored in evaluate: base name, tinker:// path, or 'tuned' "
                         "(default: --base-model)")
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--suffix", default="introspection-k1000")
    ap.add_argument("--window", type=int, default=32, help="concurrent sample requests")
    args = ap.parse_args()

    if args.stage == "prepare":
        stage_prepare(args.out, args.k, args.eval_n)
    elif args.stage == "collect":
        stage_collect(args.out, args.base_model, args.window)
    elif args.stage == "finetune":
        stage_finetune(args.out, args.base_model, args.rank, args.epochs, args.lr,
                       args.batch_size, args.suffix)
    elif args.stage == "evaluate":
        stage_evaluate(args.out, args.model or args.base_model, args.window)


if __name__ == "__main__":
    main()
