"""Sample-level view of the self-vs-generic phrasing ablation (follow-up to
scripts/selfgeneric_ablation.py, which works at the condition grain).

Oracle channel: the arms share item ids, so per-ITEM predictions pair exactly. Per (eval,
model, arm pair): Pearson r over paired item predictions, mean |diff|, and the flip table
(self=0 vs generic>0 etc.). Report channel: samples are independent draws, so no pairing —
per-condition sample distributions per arm (share exactly 0/100, within-condition SD), and
the conditions with the largest |self − generic| mean shift.

Also dumps the N most- and least-changed oracle items (with values per arm) so the raw
transcripts can be read by hand.

Run: PYTHONPATH=. python scripts/selfgeneric_samples.py [> results/reports/selfgeneric_samples.txt]
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import yaml

from behavior_prediction import common, metrics


def ranks(vals):
    """Average ranks with ties (as scripts/spearman_robustness.py)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = r
        i = j + 1
    return out


def spearman(ps):
    if len(ps) < 2:
        return None
    a, b = zip(*ps)
    return metrics.pearson(list(zip(ranks(list(a)), ranks(list(b)))))

ROOT = Path(__file__).resolve().parent.parent
POOL = list(yaml.safe_load(open(ROOT / "behavior_prediction/methods.yaml"))["selection_pool"])
ORACLE_EVALS = ["discrimeval", "capability_mmlu", "sycophancy_pushback", "reward_hacking",
                "propensitybench"]
REPORT_EVALS = ["sycophancy_pushback", "discrimeval", "capability_mmlu", "reward_hacking",
                "tau2_policy", "tau2_transfer", "propensitybench", "mask_subdomain_pressure"]
OARMS = {"A": "informed_oracle", "B": "informed_oracle_genmin", "C": "informed_oracle_3p",
         "D": "informed_oracle_3p_gen", "E": "generic_oracle"}
RARMS = {"A": "self_report", "B": "self_report_genmin", "C": "self_report_3p",
         "D": "self_report_3p_gen", "E": "generic_report"}
PAIRS = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]


def load(ev, slug, fname):
    p = Path(f"results/{ev}/{slug}/predictions/{fname}.json")
    return common.load_json(p)["predictions"] if p.exists() else None


def item_vec(ev, slug, fname):
    """{(cond_key, item_id): rate} over all conditions (signed rates for bias asks)."""
    doc = load(ev, slug, fname)
    if doc is None:
        return {}
    out = {}
    for ck, v in doc.items():
        for iid, r in (v.get("item_rates") or {}).items():
            if r is not None:
                out[(ck, iid)] = r
    return out


def main() -> None:
    print("Sample-level view (dev, selection pool). Oracle: per-ITEM pairing; report: "
          "per-condition sample distributions.\n")

    # ---- 1. oracle: per-item correlations ----------------------------------------------
    print("=== [oracle] per-item correlation between arms, pooled items per (eval, model), "
          "then averaged over models")
    print(f"{'eval':<24}" + "".join(f"{a}-{b}            " for a, b in PAIRS)
          + " (pearson / spearman / mean|diff|; ~680 paired items per (eval, model), "
          "~150 on PB)")
    for ev in ORACLE_EVALS:
        cells = {pr: [] for pr in PAIRS}
        for slug in POOL:
            vecs = {a: item_vec(ev, slug, f) for a, f in OARMS.items()}
            for a, b in PAIRS:
                ks = sorted(set(vecs[a]) & set(vecs[b]))
                if len(ks) < 10:
                    continue
                ps = [(vecs[a][k], vecs[b][k]) for k in ks]
                r = metrics.pearson(ps)
                rho = spearman(ps)
                d = mean(abs(x - y) for x, y in ps)
                cells[(a, b)].append((r if r is not None else 0.0,
                                      rho if rho is not None else 0.0, d, len(ks)))
        row = ""
        for pr in PAIRS:
            v = cells[pr]
            row += (f"{mean(x[0] for x in v):+.2f}/{mean(x[1] for x in v):+.2f}/"
                    f"{mean(x[2] for x in v):.2f}  " if v else "            --  ")
        print(f"{ev:<24}{row}")

    # flip table: how often does the self arm answer 0 where the generic arm doesn't (and
    # the reverse), per pair, pooled over evals+models.
    print("\n--- [oracle] flip table over paired items (self arm first): "
          "both0 / self0-gen>0 / self>0-gen0 / both>0; mean gen where self=0")
    for a, b in (("A", "B"), ("C", "D")):
        n00 = n01 = n10 = n11 = 0
        gen_when_self0 = []
        for ev in ORACLE_EVALS:
            for slug in POOL:
                va, vb = item_vec(ev, slug, OARMS[a]), item_vec(ev, slug, OARMS[b])
                for k in set(va) & set(vb):
                    s0, g0 = abs(va[k]) < 1e-9, abs(vb[k]) < 1e-9
                    n00 += s0 and g0; n01 += s0 and not g0
                    n10 += (not s0) and g0; n11 += (not s0) and (not g0)
                    if s0 and not g0:
                        gen_when_self0.append(abs(vb[k]))
        tot = n00 + n01 + n10 + n11
        print(f"{a}-{b}: {n00/tot:.2f} / {n01/tot:.2f} / {n10/tot:.2f} / {n11/tot:.2f} "
              f"(n={tot}); mean generic answer where self=0: "
              f"{mean(gen_when_self0):.2f}" if gen_when_self0 else "--")

    # ---- 2. oracle: most/least changed items -------------------------------------------
    for a, b in (("A", "B"), ("C", "D")):
        rows = []
        for ev in ORACLE_EVALS:
            for slug in POOL:
                va, vb = item_vec(ev, slug, OARMS[a]), item_vec(ev, slug, OARMS[b])
                for k in set(va) & set(vb):
                    rows.append((abs(va[k] - vb[k]), ev, slug, k, va[k], vb[k]))
        rows.sort(reverse=True)
        print(f"\n--- [oracle] 15 most-changed items, {a} (self) vs {b} (generic)  "
              f"[|d| self gen  eval model cond//item]")
        for d, ev, slug, k, x, y in rows[:15]:
            print(f"  {d:.2f}  {x:+.2f} {y:+.2f}  {ev:<20} {slug:<26} {k[0]}//{k[1]}")
        z = [r for r in rows if r[0] < 1e-9]
        print(f"  (unchanged items: {len(z)}/{len(rows)})")

    # ---- 3. report: sample distributions ------------------------------------------------
    print("\n=== [report] within-condition sample distributions per arm "
          "(pooled over evals/models/conditions)")
    print(f"{'arm':<4}{'n_samp':>8}{'mean':>7}{'sd(within)':>11}{'%exact0':>9}{'%exact100':>10}"
          f"{'%round(0/100)':>14}")
    for arm, fname in RARMS.items():
        allsamp, sds = [], []
        for ev in REPORT_EVALS:
            for slug in POOL:
                doc = load(ev, slug, fname)
                if doc is None:
                    continue
                for v in doc.values():
                    ss = [abs(s) for s in (v.get("samples") or []) if s is not None]
                    if ss:
                        allsamp += ss
                        sds.append(pstdev(ss))
        if allsamp:
            print(f"{arm:<4}{len(allsamp):>8}{mean(allsamp):>7.2f}{mean(sds):>11.3f}"
                  f"{mean(1.0 * (s == 0) for s in allsamp):>9.2f}"
                  f"{mean(1.0 * (s == 1) for s in allsamp):>10.2f}"
                  f"{mean(1.0 * (s in (0.0, 1.0)) for s in allsamp):>14.2f}")

    # ---- 4. report: most-shifted conditions --------------------------------------------
    for a, b in (("A", "B"), ("C", "D")):
        rows = []
        for ev in REPORT_EVALS:
            for slug in POOL:
                da, db = load(ev, slug, RARMS[a]), load(ev, slug, RARMS[b])
                if not da or not db:
                    continue
                for k in set(da) & set(db):
                    x, y = da[k].get("predicted_rate"), db[k].get("predicted_rate")
                    if x is None or y is None:
                        continue
                    rows.append((abs(x - y), ev, slug, k, x, y))
        rows.sort(reverse=True)
        print(f"\n--- [report] 12 most-shifted conditions, {a} (self) vs {b} (generic)  "
              f"[|d| self gen  eval model cond]")
        for d, ev, slug, k, x, y in rows[:12]:
            print(f"  {d:.2f}  {x:+.2f} {y:+.2f}  {ev:<22} {slug:<26} {k[:60]}")


if __name__ == "__main__":
    main()
