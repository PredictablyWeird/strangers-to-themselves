"""Noise ceiling per (eval, model): the max dev correlation any predictor could reach against
the MEASURED targets, given the targets' own sampling noise.

Reliability rel = corr between two independent measurements of the same conditions; a perfect
predictor of the TRUE rates correlates sqrt(rel) with one measured replicate. Estimated by:
  - sycophancy / capability: real split-half over the per-question outcomes in behavior_raw
    (200 random splits), Spearman-Brown corrected to full length.  [assumption-free]
  - propensitybench: real split-half over the 6 per-tactic Bernoulli outcomes (3/3 splits, SB).
    Conservative: tactic heterogeneity counts as noise.
  - discrimeval: parametric bootstrap at the contrast grain (counts ~ Binom(n, p_hat) per cell,
    shared baseline resampled once per replicate; rel = mean corr between replicate gap vectors).
All restricted to the DEV split (where methods are scored).
"""
import json, glob, random
from statistics import mean, pstdev
from collections import defaultdict
from pathlib import Path
from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

POOL = ["llama-3.3-70b", "llama-4-maverick", "deepseek-v4-flash-low", "qwen3.7-plus-low"]
REPS = 200
rng = random.Random(1234)

def sb(r):  # Spearman-Brown: half-length reliability -> full-length
    return None if r is None else (2 * r) / (1 + r) if r > -1 else None

def split_half_rel(items_by_cond, reps=REPS):
    """items_by_cond: cond -> list of 0/1 outcomes. Mean split-half r across reps, SB-corrected."""
    rs = []
    for _ in range(reps):
        pairs = []
        for outcomes in items_by_cond.values():
            if len(outcomes) < 4:
                continue
            o = outcomes[:]; rng.shuffle(o)
            h = len(o) // 2
            a, b = o[:h], o[h:2*h]   # equal halves
            pairs.append((sum(a)/len(a), sum(b)/len(b)))
        r = metrics.pearson(pairs)
        if r is not None:
            rs.append(r)
    return sb(mean(rs)) if rs else None

def best_method_r(ev, spec, slug, targets, dev):
    best = (None, "")
    for f in glob.glob(f'results/{ev}/{slug}/predictions/*.json'):
        preds = {k: v.get("predicted_rate") for k, v in json.load(open(f))["predictions"].items()}
        r = metrics.corr_spec(spec, targets, preds, keys=dev)
        if r is not None and (best[0] is None or r > best[0]):
            best = (r, Path(f).stem)
    return best

print(f"{'eval':<20} {'model':<24} {'rel':>6} {'CEILING':>8} {'best method (dev r)':>32}")
for ev in ["sycophancy_pushback", "capability_mmlu", "propensitybench", "discrimeval"]:
    spec = get_spec(ev)
    manifest = splits.load_manifest(ev)
    for slug in POOL:
        tpath = Path(f'results/{ev}/{slug}/targets.json')
        if not tpath.exists():
            continue
        targets = common.load_json(tpath)["targets"]
        dev = splits.split_keys(spec, targets, manifest, "dev")
        rel = None
        if ev == "sycophancy_pushback":
            raw = json.load(open(f'results/{ev}/{slug}/behavior_raw.json'))
            by = defaultdict(list)
            for e in raw["results"].values():
                if e.get("first_idx") == e.get("answer_idx"):
                    by[e["subject"]].append(1 if e.get("flip") else 0)
            rel = split_half_rel({s: v for s, v in by.items() if s in dev})
        elif ev == "capability_mmlu":
            raw = json.load(open(f'results/{ev}/{slug}/behavior_raw.json'))
            by = defaultdict(list)
            for e in raw["results"].values():
                if e.get("n_answered"):
                    by[e["subject"]].append(1 if e.get("n_correct") else 0)
            rel = split_half_rel({s: v for s, v in by.items() if s in dev})
        elif ev == "propensitybench":
            by = {}
            for k, t in targets.items():
                if k not in dev or not t.get("by_tactic"):
                    continue
                by[k] = [1 if v["count"] else 0 for v in t["by_tactic"].values() if v.get("n")]
            rel = split_half_rel(by)
        elif ev == "discrimeval":
            rs = []
            for _ in range(REPS):
                reps_rates = []
                for _2 in range(2):
                    rr = {}
                    for k, t in targets.items():
                        if k in dev and t.get("rate") is not None and t.get("n"):
                            rr[k] = sum(1 for _3 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                    reps_rates.append(rr)
                ta = {k: {**targets[k], "rate": reps_rates[0].get(k)} for k in targets
                      if k in reps_rates[0]}
                pairs = list(spec.score_contrasts(ta, reps_rates[1], keys=dev).values())
                r = metrics.pearson(pairs)
                if r is not None:
                    rs.append(r)
            rel = mean(rs) if rs else None
        ceiling = None if rel is None or rel <= 0 else rel ** 0.5
        br, bm = best_method_r(ev, spec, slug, targets, dev)
        c = "  None" if ceiling is None else f"{ceiling:+.2f}"
        rl = "  None" if rel is None else f"{rel:+.2f}"
        b = "--" if br is None else f"{br:+.2f} ({bm})"
        print(f"{ev:<20} {slug:<24} {rl:>6} {c:>8} {b:>36}")
