"""Why does cross_model_mean win? Offline diagnostics (dev split, no new API calls, no PB).

Decomposes the question "are the models too incoherent for self-prediction to work?" into
separately measurable pieces, all computed from files already in results/:

  1. SHARED vs MODEL-SPECIFIC target variance. cross_model_mean can only ever predict the
     variance a model shares with the pool. Per (eval, model): target reliability rel (as in
     scripts/noise_ceiling.py), the r of the leave-one-out cross-model mean (xmm), the fraction
     of the *reliable* signal xmm already explains, and the reliability of the RESIDUAL
     (target minus its xmm regression) — i.e. whether there is any measurable model-specific
     signal left for a method to predict at all.
  2. UNIQUE method signal: for every stored prediction file, dev r as on the leaderboard plus
     the partial r controlling for xmm. A method matters only if the partial r is nonzero.
  3. PREDICTION COHERENCE: split-half reliability of each sampled method's own predictions
     (SB-corrected across the per-condition `samples`). Low = the self-reports are noise
     (incoherence); high reliability with low validity = coherent but WRONG self-model.
     Disattenuated r = r / sqrt(rel_pred * rel_target) estimates the method's asymptote.
  4. IDENTITY TRANSFER: score model A's elicited predictions against model B's targets.
     If off-diagonal ~= diagonal, the elicited signal is a generic difficulty prior, not
     self-knowledge. (Uses raw_predicted_rate where present so trained sign-flips don't leak;
     trained-on-own-labels methods are skipped.)
  5. OWN-BEHAVIOR CROSS-EVAL BASELINE: predict sycophancy flip rate from the model's OWN
     measured capability_mmlu error rate (and the reverse) — a single-model, non-introspective
     baseline: behavioral self-measurement vs verbal introspection.
"""
import json, glob, random
from statistics import mean
from collections import defaultdict
from pathlib import Path
from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["sycophancy_pushback", "capability_mmlu", "discrimeval"]  # PB deliberately excluded
REPS = 200
rng = random.Random(1234)

# Trained on the target model's own labels -> their predictions already encode the targets;
# transfer/partial analyses of them would be confounded.
LABEL_TRAINED = {"llm_prediction", "llm_prediction-full", "cot_flip", "cross_model_mean"}


def sb(r):
    return None if r is None else (2 * r) / (1 + r) if r > -1 else None


def fmt(x, w=6):
    return f"{x:+.2f}".rjust(w) if x is not None else "--".rjust(w)


def vec_pearson(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    return metrics.pearson([(a[k], b[k]) for k in ks]), len(ks)


def residualize(y: dict, x: dict) -> dict:
    """y minus its OLS fit on x, over shared keys."""
    ks = sorted(set(y) & set(x))
    if len(ks) < 3:
        return {}
    mx, my = mean(x[k] for k in ks), mean(y[k] for k in ks)
    sxx = sum((x[k] - mx) ** 2 for k in ks)
    if sxx == 0:
        return {k: y[k] - my for k in ks}
    b = sum((x[k] - mx) * (y[k] - my) for k in ks) / sxx
    return {k: y[k] - (my + b * (x[k] - mx)) for k in ks}


def partial_r(y: dict, p: dict, x: dict):
    """r(y, p | x): correlation of the two residuals."""
    ks = set(y) & set(p) & set(x)
    r, _ = vec_pearson(residualize({k: y[k] for k in ks}, x),
                       residualize({k: p[k] for k in ks}, x))
    return r


# --- scored vectors (route through score_contrasts for bias evals, like the leaderboard) -------

def is_bias(spec):
    return getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast"


def scored_vecs(spec, targets, preds, dev):
    """Aligned keyed (actual, predicted) vectors, in the eval's scoring space."""
    if is_bias(spec):
        sc = spec.score_contrasts(targets, preds, keys=dev)
        ok = {k: v for k, v in sc.items() if v[0] is not None and v[1] is not None}
        return {k: v[0] for k, v in ok.items()}, {k: v[1] for k, v in ok.items()}
    a = {k: t["rate"] for k, t in targets.items() if k in dev and t.get("rate") is not None}
    p = {k: v for k, v in preds.items() if k in a and v is not None}
    return {k: a[k] for k in p}, p


def actual_vec(spec, targets, dev):
    rates = {k: t.get("rate") for k, t in targets.items()}
    a, _ = scored_vecs(spec, targets, rates, dev)
    return a


# --- measurement halves (for target + residual reliability) ------------------------------------

def rep_halves(ev, spec, slug, targets, dev, reps=REPS):
    """Per rep, two independent keyed measurement vectors of the dev conditions in scoring
    space. Real split-half (half-length; SB applies) for syco/capability; parametric bootstrap
    replicates (full-length; no SB) for discrimeval."""
    halves, full_length = [], False
    if ev == "discrimeval":
        full_length = True
        base = {k: t for k, t in targets.items() if t.get("rate") is not None and t.get("n")}
        for _ in range(reps):
            reps_rates = [{k: sum(1 for _2 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                           for k, t in base.items()} for _3 in range(2)]
            ta = {k: {**targets[k], "rate": reps_rates[0][k]} for k in reps_rates[0]}
            sc = spec.score_contrasts(ta, reps_rates[1], keys=dev)
            ok = {k: v for k, v in sc.items() if v[0] is not None and v[1] is not None}
            halves.append(({k: v[0] for k, v in ok.items()}, {k: v[1] for k, v in ok.items()}))
        return halves, full_length
    raw = json.load(open(f"results/{ev}/{slug}/behavior_raw.json"))
    by = defaultdict(list)
    for e in raw["results"].values():
        if ev == "sycophancy_pushback":
            if e.get("first_idx") == e.get("answer_idx"):
                by[e["subject"]].append(1 if e.get("flip") else 0)
        elif e.get("n_answered"):
            by[e["subject"]].append(1 if e.get("n_correct") else 0)
    by = {s: v for s, v in by.items() if s in dev and len(v) >= 4}
    for _ in range(reps):
        h1, h2 = {}, {}
        for s, outcomes in by.items():
            o = outcomes[:]
            rng.shuffle(o)
            h = len(o) // 2
            h1[s], h2[s] = sum(o[:h]) / h, sum(o[h:2 * h]) / h
        halves.append((h1, h2))
    return halves, full_length


def halves_reliability(halves, full_length, xmm=None):
    """Mean corr between the halves (residualized on xmm first when given), SB-corrected for
    half-length halves."""
    rs = []
    for h1, h2 in halves:
        if xmm is not None:
            h1, h2 = residualize(h1, xmm), residualize(h2, xmm)
        r, n = vec_pearson(h1, h2)
        if r is not None:
            rs.append(r)
    if not rs:
        return None
    m = mean(rs)
    return m if full_length else sb(m)


# --- prediction files ---------------------------------------------------------------------------

def load_pred(path):
    doc = common.load_json(path)
    preds = {k: v.get("predicted_rate") for k, v in doc["predictions"].items()}
    raw = {k: v.get("raw_predicted_rate", v.get("predicted_rate"))
           for k, v in doc["predictions"].items()}
    samples = {k: [s for s in v.get("samples") or [] if isinstance(s, (int, float))]
               for k, v in doc["predictions"].items()}
    return preds, raw, samples


def pred_reliability(spec, targets, samples, dev, reps=REPS):
    """SB-corrected split-half reliability of the method's per-condition prediction, in scoring
    space (each half aggregated like the method does: mean over samples)."""
    usable = {k: v for k, v in samples.items() if len(v) >= 4}
    if len(usable) < 5:
        return None
    rs = []
    for _ in range(reps):
        h1, h2 = {}, {}
        for k, v in usable.items():
            o = v[:]
            rng.shuffle(o)
            h = len(o) // 2
            h1[k], h2[k] = sum(o[:h]) / h, sum(o[h:2 * h]) / h
        v1, v2 = (scored_vecs(spec, targets, hh, dev)[1] for hh in (h1, h2))
        r, n = vec_pearson(v1, v2)
        if r is not None and n >= 5:
            rs.append(r)
    return sb(mean(rs)) if rs else None


# --- main ---------------------------------------------------------------------------------------

for ev in EVALS:
    spec = get_spec(ev)
    manifest = splits.load_manifest(ev)
    slugs = sorted(Path(p).parent.name for p in glob.glob(f"results/{ev}/*/targets.json"))
    actual, dev_keys, tdoc = {}, {}, {}
    for slug in slugs:
        tdoc[slug] = common.load_json(f"results/{ev}/{slug}/targets.json")["targets"]
        dev_keys[slug] = splits.split_keys(spec, tdoc[slug], manifest, "dev")
        actual[slug] = actual_vec(spec, tdoc[slug], dev_keys[slug])
    xmm = {}
    for slug in slugs:
        others = [actual[o] for o in slugs if o != slug]
        keys = set(actual[slug])
        for o in others:
            keys &= set(o)
        xmm[slug] = {k: mean(o[k] for o in others) for k in keys}

    print(f"\n{'=' * 100}\n{ev}  (dev, {len(slugs)} models)\n{'=' * 100}")

    # 1a. cross-model similarity of measured behavior
    print("\n[1a] Behavioral similarity: r between models' measured dev vectors")
    short = [s.replace("-low", "").replace("llama-", "l")[:14] for s in slugs]
    print(" " * 26 + "  ".join(f"{s:>14}" for s in short))
    for i, a in enumerate(slugs):
        row = []
        for b in slugs:
            r, _ = vec_pearson(actual[a], actual[b]) if a != b else (None, 0)
            row.append(fmt(r, 14) if a != b else " " * 13 + ".")
        print(f"{short[i]:>25} " + "  ".join(row))

    # 1b. decomposition: reliability, xmm, residual reliability
    print("\n[1b] Target decomposition per model")
    print(f"{'model':<28} {'rel':>6} {'ceil':>6} {'r_xmm':>6} {'xmm/ceil':>8} "
          f"{'best1':>6} {'resid_rel':>9} {'resid_ceil':>10}")
    halves_cache = {}
    for slug in slugs:
        halves, full = rep_halves(ev, spec, slug, tdoc[slug], dev_keys[slug])
        halves_cache[slug] = (halves, full)
        rel = halves_reliability(halves, full)
        ceil = rel ** 0.5 if rel and rel > 0 else None
        r_x, _ = vec_pearson(actual[slug], xmm[slug])
        best1 = max((vec_pearson(actual[slug], actual[o])[0] for o in slugs if o != slug),
                    key=lambda v: v if v is not None else -9)
        rrel = halves_reliability(halves, full, xmm=xmm[slug])
        rceil = rrel ** 0.5 if rrel and rrel > 0 else None
        frac = (r_x / ceil) if r_x is not None and ceil else None
        print(f"{slug:<28} {fmt(rel)} {fmt(ceil)} {fmt(r_x)} {fmt(frac, 8)} "
              f"{fmt(best1)} {fmt(rrel, 9)} {fmt(rceil, 10)}")

    # 2+3. per-method: leaderboard r, partial r | xmm, prediction reliability, disattenuated r
    print("\n[2,3] Methods: dev r | partial r given xmm | pred split-half rel | disattenuated r")
    fnames = sorted({Path(f).name for f in glob.glob(f"results/{ev}/*/predictions/*.json")})
    for fname in fnames:
        stem = Path(fname).stem
        cells = []
        for slug in slugs:
            p = Path(f"results/{ev}/{slug}/predictions/{fname}")
            if not p.exists():
                cells.append(None)
                continue
            preds, raws, samp = load_pred(p)
            a, pv = scored_vecs(spec, tdoc[slug], preds, dev_keys[slug])
            r, _ = vec_pearson(a, pv)
            pr = partial_r(actual[slug], pv, xmm[slug]) if stem != "cross_model_mean" else None
            prel = pred_reliability(spec, tdoc[slug], samp, dev_keys[slug])
            trel = halves_reliability(*halves_cache[slug])
            dis = None
            if r is not None and prel and prel > 0.1 and trel and trel > 0.1:
                dis = r / (prel * trel) ** 0.5
            cells.append((r, pr, prel, dis))
        print(f"  {stem}")
        for slug, c in zip(slugs, cells):
            if c is None:
                continue
            r, pr, prel, dis = c
            print(f"    {slug:<26} r={fmt(r)}  partial={fmt(pr)}  "
                  f"pred_rel={fmt(prel)}  disatt={fmt(dis)}")

    # 4. identity transfer for elicited (non-label-trained) methods
    print("\n[4] Identity transfer: rows = prediction source model, cols = target model "
          "(diag should beat off-diag iff the method carries SELF-knowledge)")
    for fname in fnames:
        stem = Path(fname).stem
        if stem in LABEL_TRAINED and stem != "cot_flip":
            continue
        pvecs = {}
        for slug in slugs:
            p = Path(f"results/{ev}/{slug}/predictions/{fname}")
            if not p.exists():
                continue
            preds, raws, _ = load_pred(p)
            use = raws if stem == "cot_flip" else preds
            _, pv = scored_vecs(spec, tdoc[slug], use, dev_keys[slug])
            if len(set(pv.values())) > 1:
                pvecs[slug] = pv
        if len(pvecs) < 3:
            continue
        diag, off = [], []
        for a in pvecs:
            for b in slugs:
                r, n = vec_pearson(pvecs[a], actual[b])
                if r is None or n < 5:
                    continue
                (diag if a == b else off).append(abs(r) if stem == "cot_flip" else r)
        if diag and off:
            note = " (|r|: sign is fold-learned)" if stem == "cot_flip" else ""
            print(f"  {stem:<24} diag mean {fmt(mean(diag))} ({len(diag)})   "
                  f"off-diag mean {fmt(mean(off))} ({len(off)}){note}")

    del halves_cache

# 5. own-behavior cross-eval baseline (single-model, non-introspective)
print(f"\n{'=' * 100}\n[5] Own-behavior cross-eval baseline (single-model)\n{'=' * 100}")
spec_s, spec_c = get_spec("sycophancy_pushback"), get_spec("capability_mmlu")
man_s, man_c = splits.load_manifest("sycophancy_pushback"), splits.load_manifest("capability_mmlu")
slugs = sorted(Path(p).parent.name for p in glob.glob("results/sycophancy_pushback/*/targets.json"))
acts, actc = {}, {}
for slug in slugs:
    ts = common.load_json(f"results/sycophancy_pushback/{slug}/targets.json")["targets"]
    tc = common.load_json(f"results/capability_mmlu/{slug}/targets.json")["targets"]
    acts[slug] = actual_vec(spec_s, ts, splits.split_keys(spec_s, ts, man_s, "dev"))
    actc[slug] = actual_vec(spec_c, tc, splits.split_keys(spec_c, tc, man_c, "dev"))
print(f"{'model':<28} {'flip~own_err':>12} {'partial|xmm':>12} {'acc~own(1-flip)':>16} {'r_xmm(syco)':>12}")
for slug in slugs:
    own_err = {k: 1 - v for k, v in actc[slug].items()}
    r1, _ = vec_pearson(acts[slug], own_err)
    others = [acts[o] for o in slugs if o != slug]
    keys = set(acts[slug])
    for o in others:
        keys &= set(o)
    xmm_s = {k: mean(o[k] for o in others) for k in keys}
    p1 = partial_r(acts[slug], own_err, xmm_s)
    r2, _ = vec_pearson(actc[slug], {k: 1 - v for k, v in acts[slug].items()})
    rx, _ = vec_pearson(acts[slug], xmm_s)
    print(f"{slug:<28} {fmt(r1, 12)} {fmt(p1, 12)} {fmt(r2, 16)} {fmt(rx, 12)}")
