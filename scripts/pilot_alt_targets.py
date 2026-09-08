#!/usr/bin/env python3
"""PILOT (2026-08-07): the never-extracted SECOND target metric of reward_hacking
(``hack_gap`` = hack_rate − no-points baseline, the capability-decoupled propensity) and
tau2_policy (``transfer_rate`` = escaping the pressure instead of caving to it).

Both come from the same raw measurements as the default targets — zero new API calls — so
this rescores every existing on-disk prediction against the alternative target on dev and
prints, per method, the pool-mean correlation against both targets side by side, plus the
target-target correlation (how different the two readings of the same measurement are).

Caveat printed inline: every elicitation prompt asked about the DEFAULT action (hacking with
points shown / violating the policy); scoring against the alternative reads how well those
same predictions track the other outcome, not a re-elicitation.
"""
from __future__ import annotations

import json
from pathlib import Path

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec

POOL = ["llama-3.3-70b", "llama-4-maverick", "deepseek-v4-flash-low",
        "qwen3.7-plus-low", "gemini-3.1-flash-lite-low", "gpt-5.4-nano-low"]
CASES = [("reward_hacking", "targets_hack_gap.json", "hack_gap"),
         ("tau2_policy", "targets_transfer.json", "transfer_rate")]


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
    return _pearson(x, y), _pearson(_ranks(x), _ranks(y)), len(keys)


def fmt(vals):
    # Skip models where a correlation is undefined (constant target or prediction) and show
    # how many of the pool contributed, so a "6-model mean" over 2 models is visible.
    vals = [v for v in vals if v is not None and v[0] == v[0] and v[1] == v[1]]
    if not vals:
        return f"{'— (0m)':>24s}"
    r = sum(v[0] for v in vals) / len(vals)
    rho = sum(v[1] for v in vals) / len(vals)
    return f"r={r:+.3f} rho={rho:+.3f} ({len(vals)}m)"


def main() -> int:
    for eval_name, alt_file, alt_metric in CASES:
        spec = get_spec(eval_name)
        manifest = splits.load_manifest(eval_name)
        print(f"\n=== {eval_name}: {spec.default_metric} vs {alt_metric} "
              f"(pool mean over {len(POOL)} models, dev) ===")
        rates, method_scores = [], {}
        for slug in POOL:
            base = Path(f"results/{eval_name}/{slug}")
            if not (base / "targets.json").exists() or not (base / alt_file).exists():
                print(f"  [skip {slug}: missing targets]")
                continue
            t_def = common.load_json(base / "targets.json")["targets"]
            t_alt = common.load_json(base / alt_file)["targets"]
            dev = splits.split_keys(spec, t_def, manifest, "dev")
            tr_def = {k: t_def[k]["rate"] for k in dev if t_def[k].get("rate") is not None}
            tr_alt = {k: t_alt[k]["rate"] for k in dev if t_alt[k].get("rate") is not None}
            rates.append(corr(tr_def, tr_alt))
            for f in sorted((base / "predictions").glob("*.json")):
                preds = json.loads(f.read_text())["predictions"]
                flat = {k: v.get("predicted_rate") for k, v in preds.items()}
                method_scores.setdefault(f.stem, {"def": [], "alt": []})
                method_scores[f.stem]["def"].append(corr(flat, tr_def))
                method_scores[f.stem]["alt"].append(corr(flat, tr_alt))
        print(f"  target-target ({spec.default_metric} vs {alt_metric}): {fmt(rates)}")
        print(f"  {'method':26s} {'vs ' + spec.default_metric:>24s} {'vs ' + alt_metric:>24s}")
        for name, sc in sorted(method_scores.items()):
            print(f"  {name:26s} {fmt(sc['def']):>24s} {fmt(sc['alt']):>24s}")
    print("\nNote: prompts elicited the DEFAULT action; alt-target scores test transfer of "
          "those same predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
