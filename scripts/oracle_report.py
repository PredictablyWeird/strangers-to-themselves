"""Intermediate leaderboard for informed_oracle vs the cross-model prior and the noise ceiling.

Scores whatever informed_oracle prediction files exist on disk (dev split), so it can be run
while the reasoning-model runs are still generating. For each (eval, model) with an oracle file:
oracle dev r and MAE (the oracle outputs calibrated rates, so MAE is meaningful — unlike
pairwise/cot_flip), the cross_model_mean r (the baseline to beat), and the split-half noise
ceiling (reused from scripts/noise_ceiling.py's estimator). PB excluded.
"""
import json, glob, random
from statistics import mean
from collections import defaultdict
from pathlib import Path
from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["sycophancy_pushback", "capability_mmlu", "discrimeval"]
REPS = 200
rng = random.Random(1234)


def sb(r):
    return None if r is None else (2 * r) / (1 + r) if r > -1 else None


def fmt(x, w=7):
    return (f"{x:+.2f}".rjust(w) if x is not None else "--".rjust(w))


def split_half_rel(items_by_cond):
    rs = []
    for _ in range(REPS):
        pairs = []
        for o in items_by_cond.values():
            if len(o) < 4:
                continue
            oo = o[:]; rng.shuffle(oo); h = len(oo) // 2
            pairs.append((sum(oo[:h]) / h, sum(oo[h:2 * h]) / h))
        r = metrics.pearson(pairs)
        if r is not None:
            rs.append(r)
    return sb(mean(rs)) if rs else None


def ceiling(ev, spec, slug, targets, dev):
    if ev in ("sycophancy_pushback", "capability_mmlu"):
        raw = json.load(open(f"results/{ev}/{slug}/behavior_raw.json"))
        by = defaultdict(list)
        for e in raw["results"].values():
            if ev == "sycophancy_pushback" and e.get("first_idx") == e.get("answer_idx"):
                by[e["subject"]].append(1 if e.get("flip") else 0)
            elif ev == "capability_mmlu" and e.get("n_answered"):
                by[e["subject"]].append(1 if e.get("n_correct") else 0)
        rel = split_half_rel({s: v for s, v in by.items() if s in dev})
    elif ev == "discrimeval":
        rs = []
        for _ in range(REPS):
            reps = []
            for _2 in range(2):
                reps.append({k: sum(1 for _3 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                             for k, t in targets.items()
                             if k in dev and t.get("rate") is not None and t.get("n")})
            ta = {k: {**targets[k], "rate": reps[0].get(k)} for k in targets if k in reps[0]}
            r = metrics.pearson(list(spec.score_contrasts(ta, reps[1], keys=dev).values()))
            if r is not None:
                rs.append(r)
        rel = mean(rs) if rs else None
    else:
        rel = None
    return rel ** 0.5 if rel and rel > 0 else None


print(f"{'eval':<20} {'model':<26} {'oracle_r':>9} {'oracle_MAE':>11} "
      f"{'xmm_r':>7} {'ceiling':>8} {'n':>4}")
macro = defaultdict(list)
for ev in EVALS:
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    for tp in sorted(glob.glob(f"results/{ev}/*/targets.json")):
        slug = Path(tp).parent.name
        opath = Path(f"results/{ev}/{slug}/predictions/informed_oracle.json")
        if not opath.exists():
            continue
        targets = common.load_json(tp)["targets"]
        dev = splits.split_keys(spec, targets, man, "dev")
        opred = {k: v.get("predicted_rate") for k, v in common.load_json(opath)["predictions"].items()}
        o_r = metrics.corr_spec(spec, targets, opred, keys=dev)
        o_mae = metrics.mae_spec(spec, targets, opred, keys=dev)
        xpath = Path(f"results/{ev}/{slug}/predictions/cross_model_mean.json")
        xpred = ({k: v.get("predicted_rate")
                  for k, v in common.load_json(xpath)["predictions"].items()}
                 if xpath.exists() else {})
        x_r = metrics.corr_spec(spec, targets, xpred, keys=dev) if xpred else None
        ceil = ceiling(ev, spec, slug, targets, dev)
        n = len(metrics.scored_pairs(spec, targets, opred, dev))
        if o_r is not None:
            macro[("oracle", ev)].append(o_r)
        if x_r is not None:
            macro[("xmm", ev)].append(x_r)
        print(f"{ev:<20} {slug:<26} {fmt(o_r, 9)} {fmt(o_mae, 11)} "
              f"{fmt(x_r)} {fmt(ceil, 8)} {n:>4}")

print("\nmacro over models present:")
for ev in EVALS:
    om, xm = macro.get(("oracle", ev)), macro.get(("xmm", ev))
    print(f"  {ev:<20} oracle {fmt(mean(om)) if om else '  --':>7} ({len(om or [])})   "
          f"xmm {fmt(mean(xm)) if xm else '  --':>7} ({len(xm or [])})")
