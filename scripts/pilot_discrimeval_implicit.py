#!/usr/bin/env python3
"""PILOT (2026-08-07): does the DiscrimEval result pattern reproduce on the implicit variant?

The implicit variant differs from explicit ONLY in measurement templates (name-based rather
than named demographics); the prediction prompts are identical by design (docs/discrimeval.md).
So the cheap pilot needs **no new prediction calls**: it re-scores the explicit run's on-disk
prediction files against freshly measured implicit targets.

Prints, per method, the dev contrast correlation against (a) explicit targets restricted to
the same categories (the published pattern) and (b) the new implicit targets — plus the
target-target correlation (do the two measurements even rank contrasts alike?).

Caveats printed inline: trained methods' predictions were fit on explicit labels; the oracle
methods' exhibits were explicit templates. Both still answer "would the finding transfer?"
at pilot precision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from behavior_prediction import common
from behavior_prediction.evals import get_spec

TRAINED_ON_EXPLICIT = {"few_shot", "few_shot_other", "llm_prediction", "cot_flip",
                       "oracle_xmm", "oracle_xmm_learned"}


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


def actual_gaps(spec, targets):
    """{contrast_key: actual_gap} via the echo trick (predicted slot discarded)."""
    echoed = spec.score_contrasts(targets, {k: t.get("rate") for k, t in targets.items()})
    return {ck: a for ck, (a, _) in echoed.items() if a is not None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.3-70b")
    args = ap.parse_args()
    slug = common.model_slug(args.model)

    exp, imp = get_spec("discrimeval"), get_spec("discrimeval_implicit")
    t_exp = common.load_json(f"results/discrimeval/{slug}/targets.json")["targets"]
    t_imp = common.load_json(f"results/discrimeval_implicit/{slug}/targets.json")["targets"]

    # Restrict explicit to the categories the implicit pilot measured.
    imp_cats = {t["condition"]["scenario"] for t in t_imp.values()}
    t_exp = {k: t for k, t in t_exp.items() if t["condition"]["scenario"] in imp_cats}

    g_exp, g_imp = actual_gaps(exp, t_exp), actual_gaps(imp, t_imp)
    shared = sorted(set(g_exp) & set(g_imp))
    print(f"categories: {sorted(imp_cats)}")
    print(f"contrasts: explicit={len(g_exp)} implicit={len(g_imp)} shared={len(shared)}")
    x, y = [g_exp[c] for c in shared], [g_imp[c] for c in shared]
    print(f"target-target (explicit vs implicit gaps): "
          f"pearson={_pearson(x, y):+.3f} spearman={_pearson(_ranks(x), _ranks(y)):+.3f}")
    mean_abs = lambda g: sum(abs(v) for v in g.values()) / max(1, len(g))
    print(f"mean |gap|: explicit={mean_abs(g_exp):.4f} implicit={mean_abs(g_imp):.4f}\n")

    print(f"{'method':28s} {'vs explicit':>22s} {'vs implicit':>22s}")
    pred_dir = Path(f"results/discrimeval/{slug}/predictions")
    for f in sorted(pred_dir.glob("*.json")):
        preds_doc = json.loads(f.read_text())
        flat = {k: v.get("predicted_rate") for k, v in preds_doc["predictions"].items()
                if v.get("predicted_rate") is not None}
        cols = []
        for spec, targets, gaps in ((exp, t_exp, g_exp), (imp, t_imp, g_imp)):
            scored = spec.score_contrasts(targets, flat)
            pairs = [(a, p) for a, p in scored.values() if a is not None and p is not None]
            if len(pairs) < 3:
                cols.append(f"{'n<3':>22s}")
                continue
            a, p = [q[0] for q in pairs], [q[1] for q in pairs]
            cols.append(f"r={_pearson(p, a):+.3f} rho={_pearson(_ranks(p), _ranks(a)):+.3f}")
        tag = "*" if preds_doc["base_method"] in TRAINED_ON_EXPLICIT else " "
        print(f"{f.stem + tag:28s} {cols[0]:>22s} {cols[1]:>22s}")
    print("\n* trained on explicit labels (predictions not re-fit; pilot signal only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
