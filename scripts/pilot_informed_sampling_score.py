#!/usr/bin/env python3
"""Score the informed_sampling pool runs against behavioral_sampling (and the asking anchors)
on dev, per eval and pooled. Zero API calls — reads prediction files and targets."""
from __future__ import annotations

import json
from pathlib import Path

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec

POOL = ["llama-3.3-70b", "llama-4-maverick", "deepseek-v4-flash-low",
        "qwen3.7-plus-low", "gemini-3.1-flash-lite-low", "gpt-5.4-nano-low"]
EVALS = ["sycophancy_pushback", "reward_hacking", "capability_mmlu", "tau2_policy",
         "propensitybench"]
METHODS = ["behavioral_sampling", "informed_sampling", "self_report", "informed_oracle",
           "cross_model_mean"]


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = (sum((v - mx) ** 2 for v in x)) ** 0.5
    sy = (sum((v - my) ** 2 for v in y)) ** 0.5
    return float("nan") if sx == 0 or sy == 0 else \
        sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def corr(pred, target):
    keys = [k for k in target if pred.get(k) is not None]
    if len(keys) < 3:
        return None
    x, y = [pred[k] for k in keys], [target[k] for k in keys]
    r, rho = _pearson(x, y), _pearson(_ranks(x), _ranks(y))
    return (r, rho) if r == r and rho == rho else None


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return f"{'— (0m)':>26s}"
    r = sum(v[0] for v in vals) / len(vals)
    rho = sum(v[1] for v in vals) / len(vals)
    return f"r={r:+.3f} rho={rho:+.3f} ({len(vals)}m)"


def main() -> int:
    grand = {m: [] for m in METHODS}
    for ev in EVALS:
        spec = get_spec(ev)
        manifest = splits.load_manifest(ev)
        per = {m: [] for m in METHODS}
        for slug in POOL:
            tpath = Path(f"results/{ev}/{slug}/targets.json")
            if not tpath.exists():
                continue
            t = common.load_json(tpath)["targets"]
            dev = splits.split_keys(spec, t, manifest, "dev")
            tr = {k: t[k]["rate"] for k in dev if t[k].get("rate") is not None}
            for m in METHODS:
                f = Path(f"results/{ev}/{slug}/predictions/{m}.json")
                if not f.exists():
                    continue
                p = json.loads(f.read_text())["predictions"]
                c = corr({k: v.get("predicted_rate") for k, v in p.items()}, tr)
                per[m].append(c)
                grand[m].append(c)
        print(f"\n=== {ev} (dev, pool mean) ===")
        for m in METHODS:
            print(f"  {m:24s} {fmt(per[m])}")
    print(f"\n=== ALL FOUR EVALS pooled (macro over model×eval cells) ===")
    for m in METHODS:
        print(f"  {m:24s} {fmt(grand[m])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
