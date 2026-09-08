"""Is the self-image anything more than 'the generic assistant'? Free analyses (no API calls).

Two artifact-only analyses backing the generic-self-image pillar (dev split, default pool):

1. REPORT-REPORT AGREEMENT, for self_report AND informed_oracle: per eval, the mean pairwise
   Pearson between the six models' tuned prediction vectors, next to the mean
   prediction -> own-behavior r. Reading: rr >> own means the channel carries one shared
   picture rather than six model-specific ones; own > rr means the elicitation genuinely
   tracks the reporter. Constant vectors (e.g. PropensityBench's uniform self_report denial)
   leave pairs undefined — itself the datum — counted and reported separately.

2. DONOR-COUNT CURVE for cross_model_mean: the bar recomputed with k = 1..5 measured donors
   (all C(5,k) subsets averaged, per target model). Quantifies how much measured
   infrastructure the outside-view bar actually needs — the fairness critique, in numbers.

Run:  .venv/bin/python scripts/generic_prior_analysis.py [> results/reports/generic_prior_analysis.txt]
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["discrimeval", "propensitybench", "capability_mmlu", "sycophancy_pushback",
         "reward_hacking", "tau2_policy"]
ROOT = Path(__file__).resolve().parent.parent
POOL = list(yaml.safe_load(open(ROOT / "behavior_prediction/models.yaml"))["pools"]["default"])


def tuned_self_report(ev: str, method: str = "self_report") -> str:
    try:
        doc = common.load_json(Path(f"results/{ev}/tuning.json"))
        return doc["methods"][method]["best_setting"]
    except (FileNotFoundError, KeyError, TypeError):
        return method


def load_preds(ev: str, slug: str, fname: str) -> dict[str, float] | None:
    p = Path(f"results/{ev}/{slug}/predictions/{fname}.json")
    if not p.exists():
        return None
    return {k: v.get("predicted_rate") for k, v in common.load_json(p)["predictions"].items()
            if isinstance(v, dict) and v.get("predicted_rate") is not None}


def is_bias(spec) -> bool:
    return getattr(spec, "scoring_semantics", "") == "bias_contrast"


def scored_own(spec, targets, preds, dev) -> float | None:
    """self_report -> own behavior, in the eval's scoring space (contrast gaps on bias evals)."""
    if is_bias(spec):
        sc = spec.score_contrasts(targets, preds, keys=dev)
        pairs = [(a, p) for a, p in sc.values() if a is not None and p is not None]
    else:
        pairs = [(t["rate"], preds[k]) for k, t in targets.items()
                 if k in dev and t.get("rate") is not None and preds.get(k) is not None]
    return metrics.pearson(pairs) if len(pairs) >= 3 else None


def report_report(ev: str, method: str = "self_report") -> None:
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    fname = tuned_self_report(ev, method)
    vecs: dict[str, dict[str, float]] = {}
    own: dict[str, float | None] = {}
    for slug in POOL:
        tpath = Path(f"results/{ev}/{slug}/targets.json")
        preds = load_preds(ev, slug, fname)
        if preds is None or not tpath.exists():
            continue
        targets = common.load_json(tpath)["targets"]
        dev = set(splits.split_keys(spec, targets, man, "dev"))
        if is_bias(spec):
            # prediction files are contrast-keyed on bias evals; restrict to dev contrasts
            devc = {c["key"] for t in targets.values()
                    if (c := spec.condition_contrast(t["condition"])) is not None
                    and spec.condition_key(t["condition"]) in dev}
            vecs[slug] = {k: v for k, v in preds.items() if k in devc}
        else:
            vecs[slug] = {k: v for k, v in preds.items() if k in dev}
        own[slug] = scored_own(spec, targets, preds, dev)
    pair_rs, const_pairs = [], 0
    for a, b in itertools.combinations(sorted(vecs), 2):
        shared = sorted(set(vecs[a]) & set(vecs[b]))
        r = metrics.pearson([(vecs[a][k], vecs[b][k]) for k in shared]) if len(shared) >= 3 else None
        if r is None:
            const_pairs += 1
        else:
            pair_rs.append(r)
    own_rs = [v for v in own.values() if v is not None]
    n_const_models = sum(1 for s in vecs
                         if len(set(round(v, 6) for v in vecs[s].values())) <= 1)
    print(f"{ev:<20} {method:<16} rr_mean {fmt(mean(pair_rs) if pair_rs else None)}"
          f"  (pairs {len(pair_rs)}, undef {const_pairs}, constant-models {n_const_models})"
          f"   self->own mean {fmt(mean(own_rs) if own_rs else None)} (n={len(own_rs)})")


def fmt(x) -> str:
    return f"{x:+.2f}" if x is not None else "  --"


def behavior_agreement(ev: str) -> None:
    """How much of this eval's structure is SHARED across models? Mean pairwise Pearson between
    the models' measured rate vectors, against the mean noise ceiling. High agreement means the
    eval mostly characterizes its conditions; the residual is the model-specific signal any
    introspective method could in principle report."""
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    vecs: dict[str, dict[str, float]] = {}
    for slug in POOL:
        tpath = Path(f"results/{ev}/{slug}/targets.json")
        if not tpath.exists():
            continue
        t = common.load_json(tpath)["targets"]
        dev = set(splits.split_keys(spec, t, man, "dev"))
        if is_bias(spec):
            sc = spec.score_contrasts(t, {k: v.get("rate") for k, v in t.items()}, keys=dev)
            vecs[slug] = {k: a for k, (a, _) in sc.items() if a is not None}
        else:
            vecs[slug] = {k: v["rate"] for k, v in t.items()
                          if k in dev and v.get("rate") is not None}
    rs = []
    for a, b in itertools.combinations(sorted(vecs), 2):
        shared = sorted(set(vecs[a]) & set(vecs[b]))
        r = metrics.pearson([(vecs[a][k], vecs[b][k]) for k in shared]) if len(shared) >= 3 else None
        if r is not None:
            rs.append(r)
    m = mean(rs) if rs else None
    print(f"{ev:<22} behavior-behavior r {fmt(m)} (n={len(rs)} model pairs)"
          f"   shared variance {m**2:.2f}" if m is not None else
          f"{ev:<22} behavior-behavior r   -- ")


def donor_curve(ev: str) -> None:
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    rates: dict[str, dict[str, float]] = {}
    targets_by: dict[str, dict] = {}
    for slug in POOL:
        tpath = Path(f"results/{ev}/{slug}/targets.json")
        if not tpath.exists():
            continue
        t = common.load_json(tpath)["targets"]
        targets_by[slug] = t
        rates[slug] = {k: v["rate"] for k, v in t.items() if v.get("rate") is not None}
    row = [f"{ev:<20}"]
    for k in range(1, len(POOL)):
        rs = []
        for m in targets_by:
            donors = [d for d in rates if d != m]
            dev = set(splits.split_keys(spec, targets_by[m], man, "dev"))
            for sub in itertools.combinations(donors, k):
                pred = {}
                for key in targets_by[m]:
                    vals = [rates[d][key] for d in sub if key in rates[d]]
                    if vals:
                        pred[key] = mean(vals)
                r = scored_own(spec, targets_by[m], pred, dev)
                if r is not None:
                    rs.append(r)
        row.append(f"k={k} {fmt(mean(rs) if rs else None)}")
    print("  ".join(row))


def abstract_correspondence(ev: str) -> None:
    """Do abstract questions describe the model's behavior in *abstract* scenarios?

    ``behavioral_sampling`` measures the model on GENERATED proxy scenarios written from the
    same abstract condition description the ask-channel methods are given. So it estimates what
    the model actually does in the scenarios an abstract question plausibly evokes, as opposed
    to the benchmark's real items. Three correlations separate a knowledge failure from a
    specification failure:

      ask -> benchmark   the paper's headline (near null)
      ask -> proxy       does the answer match behavior in the evoked scenarios?
      proxy -> benchmark do the evoked scenarios behave like the benchmark's items?

    If ask->proxy is much larger than ask->benchmark, the abstract question is being resolved
    to a different situation distribution than the one measured — the model would be answering
    a different question correctly, not answering this one wrongly.
    """
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    rows = []
    for slug in POOL:
        tpath = Path(f"results/{ev}/{slug}/targets.json")
        if not tpath.exists():
            continue
        targets = common.load_json(tpath)["targets"]
        dev = set(splits.split_keys(spec, targets, man, "dev"))
        ask = load_preds(ev, slug, tuned_self_report(ev, "self_report"))
        proxy = load_preds(ev, slug, tuned_self_report(ev, "behavioral_sampling"))
        if not ask or not proxy:
            continue
        r_ask_bench = scored_own(spec, targets, ask, dev)
        r_proxy_bench = scored_own(spec, targets, proxy, dev)
        keys = sorted(set(ask) & set(proxy))
        r_ask_proxy = (metrics.pearson([(proxy[k], ask[k]) for k in keys])
                       if len(keys) >= 3 else None)
        rows.append((r_ask_bench, r_ask_proxy, r_proxy_bench))
    if not rows:
        return
    def avg(i):
        vs = [r[i] for r in rows if r[i] is not None]
        return mean(vs) if vs else None
    print(f"{ev:<22} ask->benchmark {fmt(avg(0))}   ask->proxy {fmt(avg(1))}"
          f"   proxy->benchmark {fmt(avg(2))}   (n={len(rows)} models)")


def main() -> None:
    print("== Report-report agreement (dev; default pool): how much do models' elicited")
    print("   predictions agree with EACH OTHER vs with the reporter's own behavior? ==")
    for method in ("self_report", "informed_oracle"):
        for ev in EVALS:
            report_report(ev, method)
        print()
    print("\n== Cross-model BEHAVIOR agreement: how much of each eval is shared structure? ==")
    for ev in EVALS:
        behavior_agreement(ev)
    print("\n== Abstract-question correspondence: does the answer describe behavior in the")
    print("   scenarios the abstract description evokes, rather than the benchmark's items? ==")
    for ev in EVALS:
        abstract_correspondence(ev)
    print("\n== Donor-count curve for cross_model_mean (mean r over all C(5,k) donor subsets"
          " and target models; dev) ==")
    for ev in EVALS:
        donor_curve(ev)


if __name__ == "__main__":
    main()
