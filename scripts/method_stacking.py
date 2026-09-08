"""Out-of-fold stacked baselines on sycophancy_pushback (dev): does combining the cross-model
prior (xmm) with single-model signals beat xmm alone?

Follow-up to scripts/method_diagnostics.py, which showed several methods have strong PARTIAL r
given xmm (unique signal) despite lower raw r. Stacking is the constructive test: per CV fold
(splits.cv_folds, same folds the trained methods use), fit OLS of the target on the feature set
over the fit folds, predict the held-out fold, pool out-of-fold predictions, report Pearson r.
Features are all computable without the target model's own labels at predict time except the
learned stacking weights themselves (which are fit out-of-fold, like cot_flip's sign).

own_err = the model's OWN measured capability_mmlu error rate per subject — a single-model,
non-introspective behavioral feature (no other models involved).
"""
import glob
from statistics import mean
from pathlib import Path
from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EV = "sycophancy_pushback"
spec = get_spec(EV)
manifest = splits.load_manifest(EV)
slugs = sorted(Path(p).parent.name for p in glob.glob(f"results/{EV}/*/targets.json"))


def dev_rates(ev, slug):
    sp = get_spec(ev)
    man = splits.load_manifest(ev)
    t = common.load_json(f"results/{ev}/{slug}/targets.json")["targets"]
    dev = splits.split_keys(sp, t, man, "dev")
    return {k: v["rate"] for k, v in t.items() if k in dev and v.get("rate") is not None}


def pred_rates(slug, fname, raw=False):
    p = Path(f"results/{EV}/{slug}/predictions/{fname}")
    if not p.exists():
        return {}
    doc = common.load_json(p)
    key = "raw_predicted_rate" if raw else "predicted_rate"
    return {k: v.get(key, v.get("predicted_rate")) for k, v in doc["predictions"].items()}


def ols_fit(rows, y):
    """Least squares with intercept via normal equations (tiny feature counts)."""
    import itertools
    n, d = len(rows), len(rows[0]) + 1
    X = [[1.0, *r] for r in rows]
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(d)] for a in range(d)]
    b = [sum(X[i][a] * y[i] for i in range(n)) for a in range(d)]
    # Gaussian elimination with partial pivoting
    for c in range(d):
        piv = max(range(c, d), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        b[c], b[piv] = b[piv], b[c]
        if abs(A[c][c]) < 1e-12:
            continue
        for r in range(d):
            if r != c and A[r][c]:
                f = A[r][c] / A[c][c]
                A[r] = [x - f * y2 for x, y2 in zip(A[r], A[c])]
                b[r] -= f * b[c]
    return [b[c] / A[c][c] if abs(A[c][c]) > 1e-12 else 0.0 for c in range(d)]


def oof_r(target, feats, folds):
    """Pooled out-of-fold r of an OLS stack of `feats` (list of keyed dicts) against `target`."""
    keys = set(target)
    for f in feats:
        keys &= {k for k, v in f.items() if v is not None}
    pairs = []
    for held in folds:
        fit = [k for k in keys if k not in held]
        te = [k for k in keys if k in held]
        if len(fit) < len(feats) + 2 or not te:
            continue
        w = ols_fit([[f[k] for f in feats] for k in fit], [target[k] for k in fit])
        for k in te:
            pairs.append((target[k], w[0] + sum(wi * f[k] for wi, f in zip(w[1:], feats))))
    return metrics.pearson(pairs), len(pairs)


print(f"{'model':<28} " + "".join(f"{c:>16}" for c in
      ["xmm", "own_err", "xmm+own", "xmm+llmp", "xmm+cotf", "xmm+own+llmp"]))
macro = {c: [] for c in range(6)}
for slug in slugs:
    y = dev_rates(EV, slug)
    own_err = {k: 1 - v for k, v in dev_rates("capability_mmlu", slug).items()}
    others = [dev_rates(EV, o) for o in slugs if o != slug]
    keys = set(y)
    for o in others:
        keys &= set(o)
    xmm = {k: mean(o[k] for o in others) for k in keys}
    llmp = pred_rates(slug, "llm_prediction.json")
    cotf = pred_rates(slug, "cot_flip.json", raw=True)
    units = sorted(set(y))
    folds = [set(f) for f in splits.cv_folds(spec, units, seed=manifest.get("seed", 1234))]
    combos = [[xmm], [own_err], [xmm, own_err], [xmm, llmp], [xmm, cotf], [xmm, own_err, llmp]]
    cells = []
    for i, feats in enumerate(combos):
        r, n = oof_r(y, feats, folds)
        macro[i].append(r if r is not None else 0.0)
        cells.append(f"{r:+.2f} (n={n})" if r is not None else "--")
    print(f"{slug:<28} " + "".join(f"{c:>16}" for c in cells))
print(f"{'MACRO':<28} " + "".join(f"{mean(v):+.2f}".rjust(16) for v in macro.values()))
