#!/usr/bin/env python3
"""PILOT (2026-08-07): "informed watching" on sycophancy_pushback — behavioral sampling
given the informed_oracle's information tier, i.e. the actual measurement items instead of
Frame-generated proxy scenarios.

Two variants, both running the eval's own two-turn protocol and scorer on k questions per
dev subject (model under test answers; flip rate = prediction):

* ``replay`` — the FIRST k questions, a subset of the measured 20: watching with full item
  information collapses into re-running the benchmark, so this is the sanity check that the
  watching pipeline recovers the target ranking (bounded only by sampling noise).
* ``fresh``  — questions 20..20+k, never touched by the measurement: the honest deployable
  version (same item distribution, disjoint items). Its gap to ``replay`` prices what
  item-level overlap contributes beyond distribution-level information.

Writes raw results to results/_pilot/ and prints dev Pearson/Spearman against the targets,
next to the same correlations for the existing on-disk methods (behavioral_sampling,
few_shot, informed_oracle, self_report) computed the same way on the same dev keys.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.evals import mmlu_data
from behavior_prediction.evals.sycophancy import pushback

MEASURED_N = 20  # questions_per_subject used by the real measurement


def flip_rates(results: dict[str, dict]) -> dict[str, float | None]:
    agg: dict[str, list[int]] = {}
    for entry in results.values():
        subj = entry["subject"]
        agg.setdefault(subj, [0, 0])
        if entry.get("first_idx") == entry.get("answer_idx"):
            agg[subj][1] += 1
            if entry.get("flip"):
                agg[subj][0] += 1
    return {s: (f / c if c else None) for s, (f, c) in agg.items()}


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y)) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def corr(pred: dict[str, float | None], target: dict[str, float], label: str) -> None:
    keys = sorted(k for k in target if pred.get(k) is not None)
    x = [pred[k] for k in keys]
    y = [target[k] for k in keys]
    if len(keys) < 3:
        print(f"{label:28s} n={len(keys)} (too few)")
        return
    pr, sr = _pearson(x, y), _pearson(_ranks(x), _ranks(y))
    print(f"{label:28s} n={len(keys):3d}  pearson={pr:+.3f}  spearman={sr:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--k", type=int, default=5, help="questions per subject per variant")
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--score-only", action="store_true",
                    help="skip measurement; score the saved results file")
    args = ap.parse_args()

    spec = get_spec("sycophancy_pushback")
    model_full, reasoning = common.resolve_model(args.model)
    slug = common.model_slug(args.model)
    manifest = splits.load_manifest(spec.name)
    targets_doc = common.load_json(f"results/{spec.name}/{slug}/targets.json")
    targets = targets_doc["targets"]
    dev_keys = splits.split_keys(spec, targets, manifest, "dev")
    target_rates = {k: targets[k]["rate"] for k in dev_keys
                    if targets[k].get("rate") is not None}
    print(f"dev subjects with targets: {len(target_rates)}")

    out_path = Path(f"results/_pilot/informed_watching_{spec.name}_{slug}_k{args.k}.json")
    if args.score_only:
        doc = json.loads(out_path.read_text())
    else:
        items: dict[str, tuple[mmlu_data.Question, int]] = {}
        variant_of: dict[str, str] = {}
        n_missing_fresh = 0
        for subj in sorted(target_rates):
            qs = mmlu_data.load_questions(subj, MEASURED_N + args.k)
            for i, q in enumerate(qs[: args.k]):
                key = f"replay//{subj}//{i}"
                items[key] = (q, pushback.wrong_letter_idx(q.answer_idx))
                variant_of[key] = "replay"
            fresh = qs[MEASURED_N: MEASURED_N + args.k]
            if len(fresh) < args.k:
                n_missing_fresh += 1
            for i, q in enumerate(fresh):
                key = f"fresh//{subj}//{i}"
                items[key] = (q, pushback.wrong_letter_idx(q.answer_idx))
                variant_of[key] = "fresh"
        if n_missing_fresh:
            print(f"note: {n_missing_fresh} subjects have <{args.k} fresh questions "
                  f"beyond the measured {MEASURED_N} (kept whatever exists)")
        print(f"measuring {len(items)} two-turn items on {model_full} ...")
        results = pushback.measure_pushback(model_full, items,
                                            reasoning_config=reasoning,
                                            concurrency=args.concurrency)
        doc = {"model": model_full, "k": args.k,
               "results": results, "variant_of": variant_of}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=1))
        print(f"raw results -> {out_path}")

    results, variant_of = doc["results"], doc["variant_of"]
    print()
    for variant in ("replay", "fresh"):
        sub = {k: v for k, v in results.items() if variant_of[k] == variant}
        corr(flip_rates(sub), target_rates, f"informed_watching[{variant}]")

    # Context: the existing on-disk methods, scored identically on the same dev keys.
    pred_dir = Path(f"results/{spec.name}/{slug}/predictions")
    for name in ("behavioral_sampling", "few_shot", "informed_oracle", "self_report",
                 "few_shot_oracle"):
        f = pred_dir / f"{name}.json"
        if not f.exists():
            continue
        preds = json.loads(f.read_text())["predictions"]
        corr({k: v.get("predicted_rate") for k, v in preds.items() if k in target_rates},
             target_rates, name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
