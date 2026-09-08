"""Learned sign-flipping applied to EVERY method (appendix experiment; no new API calls).

The reasoned self-assessment's only learned parameter is the sign of its fit-split
correlation, applied out-of-fold with a leak-free per-fold z-transform. This sweep applies
the identical mechanism to every other method's committed predictions: per CV fold (the same
leave-one-group-out folds the trained methods use), learn sign = sign(fit-split r) and the
fit-split prediction mean/sd, transform held-out predictions to 0.5 + 0.1 * sign * z, then
score the pooled out-of-fold values. Compared against the method's raw dev r on the same
cells.

Covers the seven rate evaluations of the final 8-eval suite (DiscrimEval's scored quantity is
already signed, and the mechanism's home method opts out there). Default pool, dev split,
deterministic. METHODS is pinned to DEFAULT_METHODS (asserted below) so the sweep can never
drift from the final method set.

Writes results/reports/learned_flip_sweep.txt and paper/generated/table_learned_flip.tex.
"""
from pathlib import Path
from statistics import mean, pstdev

import yaml

from behavior_prediction import common, method_names, metrics, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import DEFAULT_METHODS
from behavior_prediction.tuning import ACTIVE_EVALS

EVALS = ["propensitybench", "capability_mmlu", "sycophancy_pushback", "reward_hacking",
         "tau2_policy", "tau2_transfer", "mask_subdomain_pressure"]
EVAL_HEAD = {"propensitybench": "PropB", "capability_mmlu": "Capab.",
             "sycophancy_pushback": "Syco.", "reward_hacking": "RewHack",
             "tau2_policy": "$\\tau^2$", "tau2_transfer": "$\\tau^2$-tr",
             "mask_subdomain_pressure": "MASK"}
METHODS = ["self_report", "generic_report", "value", "pairwise",
           "few_shot", "few_shot_other", "llm_prediction", "informed_oracle",
           "generic_oracle", "oracle_report_mean", "oracle_pairwise",
           "behavioral_sampling", "informed_sampling", "cross_model_mean", "report_mean"]
POOL = list(yaml.safe_load(open("behavior_prediction/models.yaml"))["pools"]["default"])
SCALE = 0.1

# Fail loudly if the benchmark's method/eval selection ever drifts from this sweep's lists.
assert set(METHODS) == set(DEFAULT_METHODS), \
    f"METHODS out of sync with DEFAULT_METHODS: {set(METHODS) ^ set(DEFAULT_METHODS)}"
assert set(EVALS) == set(ACTIVE_EVALS) - {"discrimeval"}, \
    f"EVALS out of sync with ACTIVE_EVALS minus discrimeval: {set(EVALS) ^ (set(ACTIVE_EVALS) - {'discrimeval'})}"
OUT_TXT = Path("results/reports/learned_flip_sweep.txt")
OUT_TEX = Path("paper/generated/table_learned_flip.tex")
OUT_MACROS = Path("paper/generated/learned_flip.tex")

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def tuned(ev, m):
    try:
        return common.load_json(Path(f"results/{ev}/tuning.json"))["methods"][m]["best_setting"]
    except Exception:
        return m


def cell(ev, slug, method):
    """(raw pooled dev r, flipped pooled OOF r) for one (eval, model, method), or None."""
    spec = get_spec(ev)
    man = splits.load_manifest(ev)
    tp = Path(f"results/{ev}/{slug}/targets.json")
    pp = Path(f"results/{ev}/{slug}/predictions/{tuned(ev, method)}.json")
    if not tp.exists() or not pp.exists():
        return None
    targets = common.load_json(tp)["targets"]
    pdoc = common.load_json(pp)["predictions"]
    preds = {k: v.get("predicted_rate") for k, v in pdoc.items()
             if isinstance(v, dict) and v.get("predicted_rate") is not None}
    dev = set(splits.split_keys(spec, targets, man, "dev"))
    scored = {k: (t["rate"], preds[k]) for k, t in targets.items()
              if k in dev and t.get("rate") is not None and preds.get(k) is not None}
    if len(scored) < 8:
        return None
    raw_r = metrics.pearson(list(scored.values()))
    if raw_r is None:
        return None
    pooled = []
    for fit_keys, score_keys in splits.folds(spec, targets, man):
        fit = [scored[k] for k in fit_keys if k in scored]
        if len(fit) < 3:
            # no usable fit split: pass predictions through untransformed (sign +1)
            pooled += [scored[k] for k in score_keys if k in scored]
            continue
        r = metrics.pearson(fit)
        sign = -1.0 if (r is not None and r < 0) else 1.0
        xs = [p for _, p in fit]
        mx, sx = mean(xs), pstdev(xs)
        for k in score_keys:
            if k not in scored:
                continue
            a, p = scored[k]
            z = (p - mx) / sx if sx > 1e-9 else 0.0
            pooled.append((a, max(0.0, min(1.0, 0.5 + SCALE * sign * z))))
    flip_r = metrics.pearson(pooled) if len(pooled) >= 8 else None
    if flip_r is None:
        return None
    return raw_r, flip_r


def main() -> int:
    per = {}   # (method, eval) -> (mean raw, mean flip, n models)
    for method in METHODS:
        for ev in EVALS:
            rs = [c for slug in POOL if (c := cell(ev, slug, method)) is not None]
            if rs:
                per[(method, ev)] = (mean(r for r, _ in rs), mean(f for _, f in rs), len(rs))

    emit(f"{'method':<22}" + "".join(f"{EVAL_HEAD[e]:>16}" for e in EVALS)
         + f"{'macro raw':>11}{'macro flip':>11}")
    emit(f"{'':<22}" + "".join(f"{'raw   flip':>16}" for e in EVALS))
    tex = ["% Auto-generated by scripts/learned_flip_sweep.py -- DO NOT EDIT BY HAND.",
           "\\begin{table}[h]", "  \\centering",
           "  \\caption{The learned-sign mechanism applied to every method (dev split,"
           " default pool; seven rate evaluations --- DiscrimEval's scored quantity is already"
           " signed). Each cell: raw dev $r$ $\\to$ pooled out-of-fold $r$ after the per-fold"
           " sign + z-transform learned per CV fold (raw\\,/\\,flipped). The transform can only add"
           " signal where a method's polarity is model-specific; elsewhere its per-fold"
           " normalization costs a little.}",
           "  \\label{tab:learned-flip}", "  \\scriptsize",
           "  \\setlength{\\tabcolsep}{1.2pt}",
           "  \\begin{tabular}{P{0.14\\linewidth}" + "c" * len(EVALS) + "cc}", "    \\toprule",
           "    Method & " + " & ".join(EVAL_HEAD[e] for e in EVALS)
           + " & Macro raw & Macro flip \\\\", "    \\midrule"]
    for method in METHODS:
        row_cells, raws, flips = [], [], []
        for ev in EVALS:
            if (method, ev) in per:
                r, f, _ = per[(method, ev)]
                row_cells.append(f"{r:+.2f}\\,/\\,{f:+.2f}")
                raws.append(r)
                flips.append(f)
            else:
                row_cells.append("--")
        if not raws:
            continue
        emit(f"{method:<22}" + "".join(
            f"{per[(method, ev)][0]:+.2f} {per[(method, ev)][1]:+.2f}".rjust(16)
            if (method, ev) in per else f"{'--':>16}" for ev in EVALS)
            + f"{mean(raws):>+11.2f}{mean(flips):>+11.2f}")
        tex.append(f"    {method_names.display(method)} & " + " & ".join(row_cells)
                   + f" & {mean(raws):+.2f} & {mean(flips):+.2f} \\\\")
    tex += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    # app:abl-settings quotes individual raw->flip cells and asserts what the macros do, so
    # export both rather than leaving the prose to be re-checked against the table by eye.
    mac = ["% Auto-generated by scripts/learned_flip_sweep.py -- DO NOT EDIT BY HAND.",
           "% \\lflip<Method><Eval>Raw/Flip = one cell; \\lflip<Method>MacroRaw/MacroFlip = the",
           "% method's macro before/after the learned sign; \\lflipNRising = how many methods the",
           "% transform helps overall (the prose's exception count)."]
    rising = []
    for method in METHODS:
        mtag = "".join(w.capitalize() for w in method.split("_"))
        raws = [per[(method, ev)][0] for ev in EVALS if (method, ev) in per]
        flips = [per[(method, ev)][1] for ev in EVALS if (method, ev) in per]
        if not raws:
            continue
        for ev in EVALS:
            if (method, ev) not in per:
                continue
            etag = "".join(w.capitalize() for w in ev.split("_"))
            mac += [f"\\newcommand{{\\lflip{mtag}{etag}Raw}}{{{per[(method, ev)][0]:+.2f}}}",
                    f"\\newcommand{{\\lflip{mtag}{etag}Flip}}{{{per[(method, ev)][1]:+.2f}}}"]
        mac += [f"\\newcommand{{\\lflip{mtag}MacroRaw}}{{{mean(raws):+.2f}}}",
                f"\\newcommand{{\\lflip{mtag}MacroFlip}}{{{mean(flips):+.2f}}}"]
        if mean(flips) > mean(raws):
            rising.append(method)
    mac.append(f"\\newcommand{{\\lflipNRising}}{{{len(rising)}}}")
    mac.append("% methods whose macro rises under the learned sign: "
               + (", ".join(rising) or "none"))

    OUT_TXT.write_text("\n".join(lines) + "\n")
    OUT_TEX.write_text("\n".join(tex) + "\n")
    OUT_MACROS.write_text("\n".join(mac) + "\n")
    print(f"\nWrote {OUT_TXT}, {OUT_TEX} and {OUT_MACROS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
