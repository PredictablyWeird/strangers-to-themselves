"""When can we predict behavior? Decompose the two prediction channels (dev, PB excluded).

The learned ensemble ≈ max(cross_model_mean, informed_oracle) per eval. This asks WHY, by
separating the two channels into what each uniquely explains:

  - cross_model_mean (xmm) = shared, condition-intrinsic structure (difficulty/severity that
    transfers across models); needs NO self-knowledge.
  - informed_oracle    = the model's own verbal self-prediction; model-specific self-knowledge.

Per (eval, model) it reports, against the measured dev targets, in the eval's scoring space:
  r_xmm, r_oracle              — each channel's correlation with behavior
  redun = r(xmm_pred, oracle)  — how redundant the two predictors are (do they say the same thing?)
  p(orc|xmm), p(xmm|orc)       — each channel's UNIQUE signal beyond the other (partial r)
  ceil, capt = max(r)/ceil     — noise ceiling and the fraction of reachable signal captured
  self%                        — knowable self-specific share: 1 - (r_xmm/ceil)^2, the fraction
                                 of RELIABLE variance the generic condition-difficulty prior
                                 leaves unexplained. The denominator for any self-knowledge
                                 claim: ~0 means behavior is condition-dominated (little self to
                                 know); large (sycophancy) means there was something to know.
The ensemble can only beat max(r_xmm, r_oracle) when BOTH partials are positive (complementary);
where one partial ~0 the ensemble collapses to the stronger channel.

Writes results/reports/predictability_decomposition.txt (the report below) and
paper/generated/decomposition.tex (per-eval macros for sec:results-unique, plus the
significance tests for the unique contributions: per-cell condition-level permutation p for
each partial, and per-eval / pooled cellwise bootstrap CIs + sign-flip p, matching the
conventions of scripts/significance_tests.py).
"""
import glob, json, random
from statistics import mean
from collections import defaultdict
from pathlib import Path
import yaml
from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["capability_mmlu", "sycophancy_pushback", "discrimeval",
         "reward_hacking", "propensitybench", "tau2_policy",
         "tau2_transfer", "mask_subdomain_pressure"]
# 2026-08-16: extended to the full 8-eval active set (the two 2026-08-11 promotions were
# missing, so decomposition.tex and the C9 ordering covered only 6 evals).
from behavior_prediction.tuning import ACTIVE_EVALS
assert set(EVALS) == set(ACTIVE_EVALS), f"decomposition evals drifted from ACTIVE_EVALS"
REPS = 200
PERMS = 5000        # condition-level permutations per cell (partial-r significance)
BOOT = 10_000       # cellwise bootstrap resamples (per-eval / pooled aggregates)
rng = random.Random(1234)
prng = random.Random(4321)   # separate stream: keeps the ceiling replicates byte-identical
# --split test (frozen-test replication check, C7/C9): reads the test split and writes
# _test-suffixed artifacts so the committed dev artifacts are never clobbered.
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--split", choices=["dev", "test"], default="dev")
SPLIT = _ap.parse_args().split
_sfx = "" if SPLIT == "dev" else "_test"
OUT_TXT = Path(f"results/reports/predictability_decomposition{_sfx}.txt")
OUT_TEX = Path(f"paper/generated/decomposition{_sfx}.tex")
report_lines = []


def emit(line=""):
    print(line)
    report_lines.append(line)
# The paper's numbers describe the default pool; results/ also holds frontier + finetune dirs.
POOL = set(yaml.safe_load(open(Path(__file__).resolve().parent.parent
                               / "behavior_prediction/models.yaml"))["pools"]["default"])


def sb(r):
    return None if r is None else (2 * r) / (1 + r) if r > -1 else None


def fmt(x, w=7):
    return (f"{x:+.2f}".rjust(w) if x is not None else "--".rjust(w))


def scored(spec, targets, preds, dev):
    """Aligned {unit: (actual, predicted)} in the eval's scoring space."""
    if getattr(spec, "scoring_semantics", "") == "bias_contrast":
        sc = spec.score_contrasts(targets, preds, keys=dev)
        return {k: v for k, v in sc.items() if v[0] is not None and v[1] is not None}
    out = {}
    for k, t in targets.items():
        if k in dev and t.get("rate") is not None and preds.get(k) is not None:
            out[k] = (t["rate"], preds[k])
    return out


def load(ev, slug, m):
    p = Path(f"results/{ev}/{slug}/predictions/{m}.json")
    return ({k: v.get("predicted_rate") for k, v in common.load_json(p)["predictions"].items()}
            if p.exists() else None)


def residual(y, x, keys):
    """y residualized on x over keys (OLS)."""
    ks = [k for k in keys if k in y and k in x]
    if len(ks) < 3:
        return {}
    mx, my = mean(x[k] for k in ks), mean(y[k] for k in ks)
    sxx = sum((x[k] - mx) ** 2 for k in ks)
    b = 0.0 if sxx == 0 else sum((x[k] - mx) * (y[k] - my) for k in ks) / sxx
    return {k: y[k] - (my + b * (x[k] - mx)) for k in ks}


def partial(a, b, ctrl, keys):
    """corr(a, b | ctrl)."""
    ra = residual(a, ctrl, keys)
    rb = residual(b, ctrl, keys)
    ks = sorted(set(ra) & set(rb))
    return metrics.pearson([(ra[k], rb[k]) for k in ks])


def partial_perm_p(a, b, ctrl, keys):
    """Two-sided condition-level permutation p for corr(a, b | ctrl): permute the
    residualized predictor across conditions, exact analog of the partial correlation."""
    ra = residual(a, ctrl, keys)
    rb = residual(b, ctrl, keys)
    ks = sorted(set(ra) & set(rb))
    if len(ks) < 5:
        return None
    va, vb = [ra[k] for k in ks], [rb[k] for k in ks]
    obs = metrics.pearson(list(zip(va, vb)))
    if obs is None:
        return None
    hits = 0
    for _ in range(PERMS):
        pb = vb[:]
        prng.shuffle(pb)
        r = metrics.pearson(list(zip(va, pb)))
        if r is not None and abs(r) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / (PERMS + 1)


def cellwise(vals):
    """Mean over cells + percentile-bootstrap 95% CI + exact/MC sign-flip permutation p,
    matching scripts/significance_tests.py conventions (here vs. a null of zero)."""
    n = len(vals)
    if n == 0:
        return None
    brng = random.Random(1234)
    boots = sorted(mean(brng.choices(vals, k=n)) for _ in range(BOOT))
    lo, hi = boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT) - 1]
    obs = abs(mean(vals))
    if n <= 20:
        hits = total = 0
        import itertools
        for signs in itertools.product((1.0, -1.0), repeat=n):
            total += 1
            if abs(mean(v * s for v, s in zip(vals, signs))) >= obs - 1e-12:
                hits += 1
        p = hits / total
    else:
        hits = sum(1 for _ in range(100_000)
                   if abs(mean(v * brng.choice((1.0, -1.0)) for v in vals)) >= obs - 1e-12)
        p = (hits + 1) / (100_001)
    return {"n": n, "mean": mean(vals), "lo": lo, "hi": hi, "p": p,
            "n_pos": sum(1 for v in vals if v > 0)}


def ceiling(ev, spec, slug, targets, dev):
    """Noise ceiling per branch: split-half from per-item raw (sycophancy/capability, then
    Spearman-Brown), parametric contrast replicates (bias evals), or parametric binomial
    replicates from each condition's committed (rate, n) (the remaining rate evals)."""
    if getattr(spec, "scoring_semantics", "") != "bias_contrast" \
            and ev not in ("sycophancy_pushback", "capability_mmlu"):
        rs = []
        usable = [t for k, t in targets.items()
                  if k in dev and t.get("rate") is not None and t.get("n")]
        for _ in range(REPS):
            pairs = []
            for t in usable:
                a = sum(1 for _2 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                b = sum(1 for _3 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                pairs.append((a, b))
            r = metrics.pearson(pairs)
            if r is not None:
                rs.append(r)
        rel = mean(rs) if rs else None
        return rel ** 0.5 if rel and rel > 0 else None
    if ev in ("sycophancy_pushback", "capability_mmlu"):
        raw = json.load(open(f"results/{ev}/{slug}/behavior_raw.json"))
        by = defaultdict(list)
        for e in raw["results"].values():
            if ev == "sycophancy_pushback" and e.get("first_idx") == e.get("answer_idx"):
                by[e["subject"]].append(1 if e.get("flip") else 0)
            elif ev == "capability_mmlu" and e.get("n_answered"):
                by[e["subject"]].append(1 if e.get("n_correct") else 0)
        rs = []
        for _ in range(REPS):
            pairs = []
            for o in ({s: v for s, v in by.items() if s in dev}).values():
                if len(o) < 4:
                    continue
                oo = o[:]; rng.shuffle(oo); h = len(oo) // 2
                pairs.append((sum(oo[:h]) / h, sum(oo[h:2 * h]) / h))
            r = metrics.pearson(pairs)
            if r is not None:
                rs.append(r)
        rel = sb(mean(rs)) if rs else None
    else:
        rs = []
        for _ in range(REPS):
            reps = [{k: sum(1 for _2 in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                     for k, t in targets.items()
                     if k in dev and t.get("rate") is not None and t.get("n")} for _3 in range(2)]
            ta = {k: {**targets[k], "rate": reps[0][k]} for k in reps[0]}
            r = metrics.pearson(list(spec.score_contrasts(ta, reps[1], keys=dev).values()))
            if r is not None:
                rs.append(r)
        rel = mean(rs) if rs else None
    return rel ** 0.5 if rel and rel > 0 else None


emit(f"{'eval':<18}{'model':<26}{'r_xmm':>7}{'r_orc':>7}{'r_sr':>7}{'redun':>7}"
     f"{'p(orc|x)':>9}{'p(sr|x)':>9}{'p(xmm|o)':>9}{'ceil':>6}{'capt':>6}{'self%':>7}")
agg = defaultdict(lambda: defaultdict(list))
for ev in EVALS:
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    for tp in sorted(glob.glob(f"results/{ev}/*/targets.json")):
        slug = Path(tp).parent.name
        if slug not in POOL:
            continue
        targets = common.load_json(tp)["targets"]
        dev = splits.split_keys(spec, targets, man, SPLIT)
        xmm, orc = load(ev, slug, "cross_model_mean"), load(ev, slug, "informed_oracle")
        sr = load(ev, slug, "self_report")
        if xmm is None or orc is None:
            continue
        sx = scored(spec, targets, xmm, dev)
        so = scored(spec, targets, orc, dev)
        ss = scored(spec, targets, sr, dev) if sr is not None else {}
        keys = sorted(set(sx) & set(so))
        if len(keys) < 5:
            continue
        actual = {k: sx[k][0] for k in keys}
        xp = {k: sx[k][1] for k in keys}
        op = {k: so[k][1] for k in keys}
        sp = {k: ss[k][1] for k in keys if k in ss}
        r_x = metrics.pearson([(actual[k], xp[k]) for k in keys])
        r_o = metrics.pearson([(actual[k], op[k]) for k in keys])
        redun = metrics.pearson([(xp[k], op[k]) for k in keys])
        p_ox = partial(actual, op, xp, keys)
        p_xo = partial(actual, xp, op, keys)
        skeys = sorted(set(sp) & set(actual))
        r_s = metrics.pearson([(actual[k], sp[k]) for k in skeys]) if len(skeys) >= 5 else None
        p_sx = partial(actual, sp, xp, skeys) if len(skeys) >= 5 else None
        ceil = ceiling(ev, spec, slug, targets, dev)
        best = max((v for v in (r_x, r_o) if v is not None), default=None)
        capt = (best / ceil) if best is not None and ceil and ceil > 0 else None
        selfsh = (max(0.0, 1.0 - (max(0.0, r_x) / ceil) ** 2)
                  if ceil and ceil > 0 and r_x is not None else None)
        pp_ox = partial_perm_p(actual, op, xp, keys) if p_ox is not None else None
        pp_sx = partial_perm_p(actual, sp, xp, skeys) if p_sx is not None else None
        for name, v in [("r_x", r_x), ("r_o", r_o), ("r_s", r_s), ("redun", redun),
                        ("p_ox", p_ox), ("p_sx", p_sx), ("p_xo", p_xo),
                        ("capt", capt), ("selfsh", selfsh)]:
            if v is not None:
                agg[ev][name].append(v)
        for name, v in [("pp_ox", pp_ox), ("pp_sx", pp_sx)]:
            if v is not None:
                agg[ev][name].append(v)
        emit(f"{ev:<18}{slug:<26}{fmt(r_x)}{fmt(r_o)}{fmt(r_s)}{fmt(redun)}"
             f"{fmt(p_ox, 9)}{fmt(p_sx, 9)}{fmt(p_xo, 9)}{fmt(ceil, 6)}{fmt(capt, 6)}{fmt(selfsh, 7)}")
    emit()

emit("MACRO per eval:")
emit(f"{'eval':<18}{'r_xmm':>7}{'r_orc':>7}{'r_sr':>7}{'redun':>7}{'p(orc|x)':>9}"
     f"{'p(sr|x)':>9}{'p(xmm|o)':>9}{'capt':>6}{'self%':>7}{'n_sr':>5}")
for ev in EVALS:
    a = agg[ev]
    row = [mean(a[n]) if a[n] else None
           for n in ("r_x", "r_o", "r_s", "redun", "p_ox", "p_sx", "p_xo", "capt", "selfsh")]
    emit(f"{ev:<18}{fmt(row[0])}{fmt(row[1])}{fmt(row[2])}{fmt(row[3])}{fmt(row[4],9)}"
         f"{fmt(row[5],9)}{fmt(row[6],9)}{fmt(row[7],6)}{fmt(row[8],7)}{len(a['p_sx']):>5}")

# ---- Significance of the unique contributions (the partials), + paper macros -------------

def slug_ev(ev):
    """Eval -> LaTeX-macro-safe fragment matching results.tex (no digits/underscores)."""
    return "".join(c for c in ev if c.isalpha()).capitalize()


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


emit()
emit("Unique-contribution significance (partial r; cellwise over models, permutation over conditions):")
emit(f"{'eval':<18}{'channel':<10}{'mean':>7}{'95% CI':>17}{'sign-flip p':>12}"
     f"{'pos':>5}{'perm p<.05':>11}")
tex = ["% Auto-generated by scripts/predictability_decomposition.py -- DO NOT EDIT BY HAND.",
       "% Knowable self-specific share (selfsh), channel partials given cross_model_mean, and",
       f"% their significance: cellwise bootstrap 95% CI + exact sign-flip p over model cells;",
       f"% NSig = cells individually significant by a {PERMS}-draw condition-level permutation.",
       ""]


def macro(name, value):
    tex.append(f"\\newcommand{{\\{name}}}{{{value}}}")


pooled = defaultdict(list)
for ev in EVALS:
    a, E = agg[ev], slug_ev(ev)
    if a["selfsh"]:
        macro(f"decompSelfsh{E}", f"{mean(a['selfsh']):.2f}")
    for chan, pkey, ppkey in [("Orc", "p_ox", "pp_ox"), ("Sr", "p_sx", "pp_sx")]:
        vals = a[pkey]
        if not vals:
            continue
        pooled[pkey] += vals
        pooled[ppkey] += a[ppkey]
        s = cellwise(vals)
        nsig = sum(1 for p in a[ppkey] if p < 0.05)
        emit(f"{ev:<18}{chan:<10}{fmt(s['mean'])}   [{s['lo']:+.2f}, {s['hi']:+.2f}]"
             f"{fmt_p(s['p']):>12}{s['n_pos']:>3}/{s['n']}{nsig:>6}/{len(a[ppkey])}")
        macro(f"decompP{chan}{E}", f"{s['mean']:+.2f}")
        macro(f"decompP{chan}{E}CILo", f"{s['lo']:+.2f}")
        macro(f"decompP{chan}{E}CIHi", f"{s['hi']:+.2f}")
        macro(f"decompP{chan}{E}P", fmt_p(s["p"]))
        macro(f"decompP{chan}{E}NPos", str(s["n_pos"]))
        macro(f"decompP{chan}{E}N", str(s["n"]))
        macro(f"decompP{chan}{E}NSig", str(nsig))
for chan, pkey, ppkey in [("Orc", "p_ox", "pp_ox"), ("Sr", "p_sx", "pp_sx")]:
    s = cellwise(pooled[pkey])
    nsig = sum(1 for p in pooled[ppkey] if p < 0.05)
    emit(f"{'ALL':<18}{chan:<10}{fmt(s['mean'])}   [{s['lo']:+.2f}, {s['hi']:+.2f}]"
         f"{fmt_p(s['p']):>12}{s['n_pos']:>3}/{s['n']}{nsig:>6}/{len(pooled[ppkey])}")
    macro(f"decompP{chan}All", f"{s['mean']:+.2f}")
    macro(f"decompP{chan}AllCILo", f"{s['lo']:+.2f}")
    macro(f"decompP{chan}AllCIHi", f"{s['hi']:+.2f}")
    macro(f"decompP{chan}AllP", fmt_p(s["p"]))
    macro(f"decompP{chan}AllNPos", str(s["n_pos"]))
    macro(f"decompP{chan}AllN", str(s["n"]))
    macro(f"decompP{chan}AllNSig", str(nsig))

OUT_TXT.write_text("\n".join(report_lines) + "\n")
OUT_TEX.write_text("\n".join(tex) + "\n")
print(f"\nWrote {OUT_TXT} and {OUT_TEX}")
