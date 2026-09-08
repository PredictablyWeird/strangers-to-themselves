"""Self-vs-generic phrasing ablation — all pre-specified analyses.

Dev split, six-model selection pool, no API calls. Two channels:

  report  A self_report (nudge off)  B self_report_genmin  C self_report_3p  D self_report_3p_gen
          E generic_report           [A* = tuned self_report, where the tuned setting has the nudge]
  oracle  A' informed_oracle  B' informed_oracle_genmin  C' informed_oracle_3p
          D' informed_oracle_3p_gen  E' generic_oracle

Per channel:
  1. Aggregate scoreboard: r vs own behavior per (model, eval); per-eval mean; macro-mean over
     evals with bootstrap 95% CI over (model, eval) cells; paired contrasts (A−B, C−D, A−C, B−D,
     B−E, D−E) with bootstrap 95% CI + exact sign-flip p; constancy diagnostics.
  2. Answer level: 5x5 arm-pair matrices (Pearson r, mean |diff|) averaged over models; within-arm
     cross-model agreement; the DIRECT TEST — corr(self − generic answer, own − cross-model
     behavior) for A−B and C−D.
  3. Self-specificity instruments on every arm: identity transfer (diag − off), partial r given
     cross_model_mean, own arm vs the same-arm committee of the other pool models.

Run:  PYTHONPATH=. .venv/bin/python scripts/selfgeneric_ablation.py [> results/reports/selfgeneric_ablation.txt]
"""
from __future__ import annotations

import argparse
import itertools
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import yaml

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

ROOT = Path(__file__).resolve().parent.parent
POOL = list(yaml.safe_load(open(ROOT / "behavior_prediction/methods.yaml"))["selection_pool"])
REPORT_EVALS = ["sycophancy_pushback", "discrimeval", "capability_mmlu", "reward_hacking",
                "tau2_policy", "tau2_transfer", "propensitybench", "mask_subdomain_pressure"]
ORACLE_EVALS = ["discrimeval", "capability_mmlu", "sycophancy_pushback", "reward_hacking",
                "propensitybench"]
CHANNELS = {
    "report": {"A": "self_report", "B": "self_report_genmin", "C": "self_report_3p",
               "D": "self_report_3p_gen", "E": "generic_report"},
    "oracle": {"A": "informed_oracle", "B": "informed_oracle_genmin", "C": "informed_oracle_3p",
               "D": "informed_oracle_3p_gen", "E": "generic_oracle"},
}
CONTRASTS = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("B", "E"), ("D", "E")]
BOOT, SEED = 2000, 0


# --- loading -------------------------------------------------------------------------------

def is_bias(spec) -> bool:
    return getattr(spec, "scoring_semantics", "") == "bias_contrast"


def tuned(ev: str, m: str) -> str:
    try:
        return common.load_json(Path(f"results/{ev}/tuning.json"))["methods"][m]["best_setting"]
    except Exception:
        return m


def load_doc(ev: str, slug: str, fname: str):
    p = Path(f"results/{ev}/{slug}/predictions/{fname}.json")
    return common.load_json(p)["predictions"] if p.exists() else None


def dev_actuals(spec, ev: str, slug: str) -> dict[str, float]:
    """Own behavior in the eval's scored space (contrast gaps on bias evals), dev keys only."""
    t = common.load_json(Path(f"results/{ev}/{slug}/targets.json"))["targets"]
    dev = set(splits.split_keys(spec, t, splits.load_manifest(ev), "dev"))
    if is_bias(spec):
        sc = spec.score_contrasts(t, {k: v.get("rate") for k, v in t.items()}, keys=dev)
        return {k: a for k, (a, _) in sc.items() if a is not None}
    return {k: v["rate"] for k, v in t.items() if k in dev and v.get("rate") is not None}


def scored_preds(spec, ev: str, slug: str, fname: str, keys: set[str]) -> dict[str, float] | None:
    """Predictions keyed in the scored space, restricted to ``keys``. Prediction files are already
    contrast-keyed on bias evals for the introspective/oracle methods; cell-keyed files (e.g.
    cross_model_mean) are mapped through ``score_contrasts``."""
    doc = load_doc(ev, slug, fname)
    if doc is None:
        return None
    preds = {k: v.get("predicted_rate") for k, v in doc.items()
             if isinstance(v, dict) and v.get("predicted_rate") is not None}
    if is_bias(spec) and not any(k in keys for k in preds):
        t = common.load_json(Path(f"results/{ev}/{slug}/targets.json"))["targets"]
        sc = spec.score_contrasts(t, preds)
        preds = {k: p for k, (_, p) in sc.items() if p is not None}
    return {k: v for k, v in preds.items() if k in keys}


def answers_01(ev: str, slug: str, fname: str, keys: set[str]) -> list[float]:
    """Every individual answer (samples / item predictions) for the constancy diagnostics."""
    doc = load_doc(ev, slug, fname) or {}
    out: list[float] = []
    for k, v in doc.items():
        if k not in keys or not isinstance(v, dict):
            continue
        if v.get("samples"):
            out += [abs(s) for s in v["samples"]]
        elif v.get("item_rates"):
            out += [abs(x) for x in v["item_rates"].values() if x is not None]
        elif v.get("predicted_rate") is not None:
            out.append(abs(v["predicted_rate"]))
    return out


# --- stats helpers ---------------------------------------------------------------------------

def r_of(x: dict[str, float], y: dict[str, float]) -> float | None:
    ks = sorted(set(x) & set(y))
    return metrics.pearson([(x[k], y[k]) for k in ks]) if len(ks) >= 3 else None


def ranks(vals: list[float]) -> list[float]:
    """Average ranks with ties (as scripts/spearman_robustness.py)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def rho_of(x: dict[str, float], y: dict[str, float]) -> float | None:
    """Spearman between two condition-keyed vectors, over their shared conditions."""
    ks = sorted(set(x) & set(y))
    if len(ks) < 3:
        return None
    return metrics.pearson(list(zip(ranks([x[k] for k in ks]), ranks([y[k] for k in ks]))))


def r0(x, y) -> float | None:
    """r with constant predictors scored 0 (the leaderboard convention); None if no overlap."""
    ks = set(x) & set(y)
    if len(ks) < 3:
        return None
    r = r_of(x, y)
    return 0.0 if r is None else r


def partial_r(x: dict, y: dict, z: dict) -> float | None:
    ks = sorted(set(x) & set(y) & set(z))
    if len(ks) < 4:
        return None
    rxy, rxz, ryz = (metrics.pearson([(x[k], y[k]) for k in ks]),
                     metrics.pearson([(x[k], z[k]) for k in ks]),
                     metrics.pearson([(y[k], z[k]) for k in ks]))
    if None in (rxy, rxz, ryz):
        return None
    den = math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return (rxy - rxz * ryz) / den if den > 1e-9 else None


def boot_mean(vals: list[float]) -> tuple[float, float, float]:
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(SEED)
    b = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(BOOT)])
    return a.mean(), *np.percentile(b, [2.5, 97.5])


def signflip_p(d: np.ndarray) -> float:
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if n <= 16:
        hits = sum(1 for s in itertools.product((1.0, -1.0), repeat=n)
                   if abs((d * s).mean()) >= obs - 1e-12)
        return hits / 2 ** n
    rng = np.random.default_rng(SEED)
    signs = rng.choice([1.0, -1.0], size=(20000, n))
    return float(((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum() + 1) / 20001)


def f(x, w=6, d=2) -> str:
    return f"{x:+{w}.{d}f}" if x is not None and not (isinstance(x, float) and math.isnan(x)) else " " * (w - 2) + "--"


# --- the analyses ----------------------------------------------------------------------------

class Channel:
    def __init__(self, name: str, evals: list[str]):
        self.name, self.evals, self.arms = name, evals, CHANNELS[name]
        self.act: dict[tuple[str, str], dict] = {}        # (ev, slug) -> actual vector
        self.pred: dict[tuple[str, str, str], dict] = {}  # (ev, slug, arm) -> pred vector
        self.cmm: dict[tuple[str, str], dict] = {}        # (ev, slug) -> cross_model_mean vector
        self.missing: list[str] = []
        self.agree: dict[tuple[str, str, str], tuple[float, float, int]] = {}  # (ev,a,b)->(r,rho,n)
        for ev in evals:
            spec = get_spec(ev)
            for slug in POOL:
                if not Path(f"results/{ev}/{slug}/targets.json").exists():
                    continue
                act = dev_actuals(spec, ev, slug)
                self.act[(ev, slug)] = act
                keys = set(act)
                for arm, base in self.arms.items():
                    p = scored_preds(spec, ev, slug, base, keys)
                    if p is None or not p:
                        self.missing.append(f"{ev}/{slug}/{base}")
                        continue
                    self.pred[(ev, slug, arm)] = p
                if name == "report":      # tuned self_report may differ (honesty nudge)
                    tb = tuned(ev, "self_report")
                    if tb != "self_report":
                        p = scored_preds(spec, ev, slug, tb, keys)
                        if p:
                            self.pred[(ev, slug, "A*")] = p
                c = scored_preds(spec, ev, slug, "cross_model_mean", keys)
                if c:
                    self.cmm[(ev, slug)] = c

    def arm_list(self) -> list[str]:
        arms = list(self.arms)
        if any(a == "A*" for (_, _, a) in self.pred):
            arms.append("A*")
        return arms

    def cell_r(self, ev, slug, arm) -> float | None:
        p = self.pred.get((ev, slug, arm))
        return r0(self.act[(ev, slug)], p) if p else None

    # 1 ------------------------------------------------------------------------------------
    def scoreboard(self, out: list[str]) -> None:
        arms = self.arm_list()
        out.append(f"\n=== [{self.name}] 1. AGGREGATE: Pearson r vs own behavior (dev, constant -> 0)")
        out.append("Arms: " + ", ".join(f"{a}={b}" for a, b in self.arms.items())
                   + (", A*=tuned self_report (nudge on)" if "A*" in arms else ""))
        hdr = f"{'eval':<24}" + "".join(f"{a:>8}" for a in arms) + "   n_models"
        out.append(hdr)
        cells: dict[str, list[float]] = defaultdict(list)
        for ev in self.evals:
            row, ns = [], set()
            for a in arms:
                rs = [self.cell_r(ev, s, a) for s in POOL if (ev, s) in self.act]
                rs = [r for r in rs if r is not None]
                row.append(mean(rs) if rs else None)
                cells[a] += rs
                ns.add(len(rs))
            out.append(f"{ev:<24}" + "".join(f"{f(x, 8)}" for x in row) + f"   {sorted(ns)}")
        out.append(f"{'MACRO (mean of cells)':<24}" + "".join(
            f"{f(boot_mean(cells[a])[0], 8)}" for a in arms))
        out.append(f"{'  95% CI lo':<24}" + "".join(f"{f(boot_mean(cells[a])[1], 8)}" for a in arms))
        out.append(f"{'  95% CI hi':<24}" + "".join(f"{f(boot_mean(cells[a])[2], 8)}" for a in arms))

        out.append(f"\n--- [{self.name}] paired contrasts (mean r difference over shared (model, eval) "
                   "cells; bootstrap 95% CI; exact sign-flip p)")
        out.append(f"{'contrast':<10}{'eval':<24}{'n':>4}{'diff':>8}{'lo':>8}{'hi':>8}{'p':>8}"
                   f"{'pos/neg':>10}")
        for a, b in CONTRASTS:
            for ev in self.evals + ["POOLED"]:
                evs = self.evals if ev == "POOLED" else [ev]
                d = [self.cell_r(e, s, a) - self.cell_r(e, s, b) for e in evs for s in POOL
                     if (e, s) in self.act and self.cell_r(e, s, a) is not None
                     and self.cell_r(e, s, b) is not None]
                if not d:
                    continue
                m, lo, hi = boot_mean(d)
                p = signflip_p(np.array(d))
                out.append(f"{a + '-' + b:<10}{ev:<24}{len(d):>4}{f(m, 8)}{f(lo, 8)}{f(hi, 8)}"
                           f"{p:>8.3f}{sum(x > 0 for x in d):>5}/{sum(x < 0 for x in d):<4}")

        out.append(f"\n--- [{self.name}] constancy diagnostics per arm (over (model, eval) cells / "
                   "all individual answers)")
        out.append(f"{'arm':<6}{'cells':>6}{'const':>7}{'level':>8}{'at0':>7}{'at100':>7}"
                   f"{'  (const = fraction of cells with a constant prediction vector)'}")
        for a in arms:
            n = const = 0
            answers: list[float] = []
            for ev in self.evals:
                base = self.arms.get(a) or tuned(ev, "self_report")
                for s in POOL:
                    p = self.pred.get((ev, s, a))
                    if not p:
                        continue
                    n += 1
                    const += len(set(round(v, 6) for v in p.values())) == 1
                    answers += answers_01(ev, s, base, set(self.act[(ev, s)]))
            if n:
                at0 = mean(x == 0 for x in answers) if answers else float("nan")
                at1 = mean(x == 1 for x in answers) if answers else float("nan")
                out.append(f"{a:<6}{n:>6}{const / n:>7.2f}{mean(answers) if answers else float('nan'):>8.2f}"
                           f"{at0:>7.2f}{at1:>7.2f}")

    # 2 ------------------------------------------------------------------------------------
    def answer_level(self, out: list[str]) -> None:
        arms = [a for a in self.arms]
        out.append(f"\n=== [{self.name}] 2. ANSWER LEVEL (per (model, eval) over shared conditions, "
                   "averaged over models)")
        for ev in self.evals:
            R = {(a, b): [] for a in arms for b in arms}
            RHO = {(a, b): [] for a in arms for b in arms}
            D = {(a, b): [] for a in arms for b in arms}
            for s in POOL:
                for a, b in itertools.product(arms, arms):
                    pa, pb = self.pred.get((ev, s, a)), self.pred.get((ev, s, b))
                    if not pa or not pb:
                        continue
                    ks = sorted(set(pa) & set(pb))
                    if len(ks) < 3:
                        continue
                    r = metrics.pearson([(pa[k], pb[k]) for k in ks])
                    if r is not None:
                        R[(a, b)].append(r)
                    rho = rho_of(pa, pb)
                    if rho is not None:
                        RHO[(a, b)].append(rho)
                    D[(a, b)].append(mean(abs(pa[k] - pb[k]) for k in ks))
            out.append(f"\n{ev}:  Pearson r between arms (upper) / mean |diff| (lower), n models in ()")
            out.append(f"{'':<6}" + "".join(f"{a:>12}" for a in arms))
            for a in arms:
                cells = []
                for b in arms:
                    if a == b:
                        cells.append(f"{'—':>12}")
                    elif arms.index(b) > arms.index(a):
                        v = R[(a, b)]
                        cells.append(f"{f(mean(v), 7) if v else '     --':>7}({len(v)})".rjust(12))
                    else:
                        v = D[(a, b)]
                        cells.append(f"{(f'{mean(v):.2f}' if v else '--'):>8}({len(v)})".rjust(12))
                out.append(f"{a:<6}" + "".join(cells))
            # within-arm cross-model agreement
            agr = []
            for a in arms:
                rs = []
                for s1, s2 in itertools.combinations(POOL, 2):
                    p1, p2 = self.pred.get((ev, s1, a)), self.pred.get((ev, s2, a))
                    if p1 and p2:
                        r = r_of(p1, p2)
                        if r is not None:
                            rs.append(r)
                agr.append(f"{a}={f(mean(rs), 5) if rs else '  --'}({len(rs)})")
            out.append("  cross-model agreement within arm (mean pairwise r over model pairs): "
                       + "  ".join(agr))
            # The self/generic contrast at THIS grain is what the paper's ordering claim rests
            # on: conditions are what every method is scored over, so Pearson and Spearman here
            # answer "does the swap reorder the conditions?" (the per-item view lives in
            # scripts/selfgeneric_samples.py and is a different, finer question).
            rankrow = []
            for a, b in (("A", "B"), ("C", "D")):
                r, rho = R.get((a, b), []), RHO.get((a, b), [])
                if r and rho:
                    self.agree[(ev, a, b)] = (mean(r), mean(rho), len(r))
                    rankrow.append(f"{a}-{b}: r={f(mean(r), 5)} rho={f(mean(rho), 5)}({len(r)})")
            if rankrow:
                out.append("  self/generic agreement at the CONDITION grain: " + "  ".join(rankrow))

        out.append(f"\n--- [{self.name}] DIRECT TEST: corr(self − generic answer, own − cross-model "
                   "behavior) per (model, eval); positive = the self question adds self-knowledge")
        out.append(f"{'pair':<8}{'eval':<24}{'n':>4}{'mean r':>8}{'lo':>8}{'hi':>8}{'p':>8}"
                   f"{'pos/neg':>10}")
        for a, b in (("A", "B"), ("C", "D")):
            pooled = []
            for ev in self.evals:
                rs = []
                for s in POOL:
                    pa, pb, cm = (self.pred.get((ev, s, a)), self.pred.get((ev, s, b)),
                                  self.cmm.get((ev, s)))
                    if not pa or not pb or not cm:
                        continue
                    act = self.act[(ev, s)]
                    ks = sorted(set(pa) & set(pb) & set(cm) & set(act))
                    if len(ks) < 4:
                        continue
                    diff = {k: pa[k] - pb[k] for k in ks}
                    resid = {k: act[k] - cm[k] for k in ks}
                    r = r_of(diff, resid)
                    rs.append(0.0 if r is None else r)   # constant diff -> no self content
                if rs:
                    m, lo, hi = boot_mean(rs)
                    out.append(f"{a + '-' + b:<8}{ev:<24}{len(rs):>4}{f(m, 8)}{f(lo, 8)}{f(hi, 8)}"
                               f"{signflip_p(np.array(rs)):>8.3f}"
                               f"{sum(x > 0 for x in rs):>5}/{sum(x < 0 for x in rs):<4}")
                    pooled += rs
            if pooled:
                m, lo, hi = boot_mean(pooled)
                out.append(f"{a + '-' + b:<8}{'POOLED':<24}{len(pooled):>4}{f(m, 8)}{f(lo, 8)}"
                           f"{f(hi, 8)}{signflip_p(np.array(pooled)):>8.3f}"
                           f"{sum(x > 0 for x in pooled):>5}/{sum(x < 0 for x in pooled):<4}")

    # 3 ------------------------------------------------------------------------------------
    def instruments(self, out: list[str]) -> None:
        arms = self.arm_list()
        out.append(f"\n=== [{self.name}] 3. SELF-SPECIFICITY INSTRUMENTS on every arm (pooled over evals; "
                   "per-eval rows below)")
        out.append(f"{'arm':<5}{'eval':<24}{'diag':>7}{'off':>7}{'transfer':>9}{'partial':>9}"
                   f"{'own':>7}{'committee':>10}{'own-comm':>9}")
        for a in arms:
            tot = defaultdict(list)
            for ev in self.evals:
                d, o, pr, own, comm = [], [], [], [], []
                slugs = [s for s in POOL if (ev, s) in self.act]
                for s1, s2 in itertools.product(slugs, slugs):
                    p = self.pred.get((ev, s1, a))
                    if not p:
                        continue
                    r = r_of(p, self.act[(ev, s2)])
                    if r is None:
                        r = 0.0 if s1 == s2 else None
                    if r is None:
                        continue
                    (d if s1 == s2 else o).append(r)
                for s in slugs:
                    p = self.pred.get((ev, s, a))
                    if not p:
                        continue
                    cm = self.cmm.get((ev, s))
                    if cm:
                        x = partial_r(p, self.act[(ev, s)], cm)
                        if x is not None:
                            pr.append(x)
                    others = [self.pred[(ev, t, a)] for t in slugs if t != s and (ev, t, a) in self.pred]
                    if len(others) >= 2:
                        ks = set(p)
                        for q in others:
                            ks &= set(q)
                        if len(ks) >= 3:
                            cvec = {k: mean(q[k] for q in others) for k in ks}
                            own.append(r0(p, self.act[(ev, s)]))
                            comm.append(r0(cvec, self.act[(ev, s)]))
                if d:
                    row = (mean(d), mean(o) if o else None,
                           (mean(d) - mean(o)) if o else None,
                           mean(pr) if pr else None, mean(own) if own else None,
                           mean(comm) if comm else None,
                           (mean(own) - mean(comm)) if own else None)
                    out.append(f"{a:<5}{ev:<24}" + "".join(f(x, w) for x, w in
                                                            zip(row, (7, 7, 9, 9, 7, 10, 9))))
                    tot["d"] += d; tot["o"] += o; tot["pr"] += pr; tot["own"] += own; tot["comm"] += comm
            if tot["d"]:
                row = (mean(tot["d"]), mean(tot["o"]) if tot["o"] else None,
                       (mean(tot["d"]) - mean(tot["o"])) if tot["o"] else None,
                       mean(tot["pr"]) if tot["pr"] else None,
                       mean(tot["own"]) if tot["own"] else None,
                       mean(tot["comm"]) if tot["comm"] else None,
                       (mean(tot["own"]) - mean(tot["comm"])) if tot["own"] else None)
                out.append(f"{a:<5}{'POOLED':<24}" + "".join(f(x, w) for x, w in
                                                              zip(row, (7, 7, 9, 9, 7, 10, 9))))
            out.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", default="report,oracle")
    args = ap.parse_args()
    out = ["Self-vs-generic phrasing ablation — dev split, "
           f"selection pool = {', '.join(POOL)}",
           "Arm codes: A self/2p (anchor)  B generic/2p (subject swap only)  C self/3p  "
           "D generic/3p (clean minimal pair)  E current generic control (preamble)"]
    channels: dict[str, Channel] = {}
    for name in args.channels.split(","):
        evals = REPORT_EVALS if name == "report" else ORACLE_EVALS
        ch = Channel(name, evals)
        if ch.missing:
            out.append(f"\n[{name}] missing prediction files ({len(ch.missing)}): "
                       + ", ".join(ch.missing[:12]) + (" ..." if len(ch.missing) > 12 else ""))
        ch.scoreboard(out)
        ch.answer_level(out)
        ch.instruments(out)
        channels[name] = ch
    print("\n".join(out))

    # Condition-grain self/generic agreement, exported for the paper: this is the grain every
    # method is scored at, so it is the number that answers "does removing the self reorder the
    # conditions?". Per-item agreement (scripts/selfgeneric_samples.py) is a separate, finer
    # claim and must not be quoted in its place.
    tex = ["% Auto-generated by scripts/selfgeneric_ablation.py -- DO NOT EDIT BY HAND.",
           "% Agreement between the self-framed and generic-subject arms' predictions over the",
           "% CONDITIONS they are scored on (dev split, selection pool): Pearson and Spearman,",
           "% averaged over models, per evaluation and pooled. Pair AB = published 2p pair,",
           "% CD = the clean 3p minimal pair."]
    for name, ch in channels.items():
        ctag = name.capitalize()
        for a, b in (("A", "B"), ("C", "D")):
            cells = [v for (ev, x, y), v in ch.agree.items() if (x, y) == (a, b)]
            for ev in ch.evals:
                v = ch.agree.get((ev, a, b))
                if not v:
                    continue
                # digits are illegal in LaTeX macro names, so tau2_policy -> TauPolicy
                etag = "".join(w.capitalize() for w in ev.replace("tau2", "tau").split("_"))
                tex += [f"\\newcommand{{\\sgAgree{ctag}{a}{b}{etag}R}}{{{v[0]:+.2f}}}",
                        f"\\newcommand{{\\sgAgree{ctag}{a}{b}{etag}Rho}}{{{v[1]:+.2f}}}"]
            if cells:
                tex += [f"\\newcommand{{\\sgAgree{ctag}{a}{b}R}}{{{mean(v[0] for v in cells):+.2f}}}",
                        f"\\newcommand{{\\sgAgree{ctag}{a}{b}Rho}}{{{mean(v[1] for v in cells):+.2f}}}",
                        f"\\newcommand{{\\sgAgree{ctag}{a}{b}NEvals}}{{{len(cells)}}}",
                        f"\\newcommand{{\\sgAgree{ctag}{a}{b}Min}}{{{min(v[1] for v in cells):+.2f}}}",
                        f"\\newcommand{{\\sgAgree{ctag}{a}{b}Max}}{{{max(v[1] for v in cells):+.2f}}}"]
    Path("paper/generated/selfgeneric_agreement.tex").write_text("\n".join(tex) + "\n")
    print("\nWrote paper/generated/selfgeneric_agreement.tex")


if __name__ == "__main__":
    main()
