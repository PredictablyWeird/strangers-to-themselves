"""Self-serving level bias: does naming the model as the subject shift its predicted rates in
the direction that flatters it?  (Descriptive analysis added 2026-08-23, NOT pre-registered.)

For each (eval, model) cell that survives the noise-ceiling filter, compute the signed level bias
(mean predicted - mean actual, in rate units) of the self-framed method and its generic-subject
control at their tuned settings, on identical conditions, and the paired difference
self - generic.  Two tiers: abstract (self_report vs generic_report) and item-informed
(informed_oracle vs generic_oracle).

Evaluations are labeled by what a HIGH rate means so the direction can be read:
  harm   -- higher rate = the model misbehaves more (flattering direction: negative shift)
  credit -- higher rate = the model looks better (Capability accuracy; flattering: positive)
  handoff -- tau2 transfer (handing the task to a human; not a harm, not a credit)
  none   -- DiscrimEval's signed contrast has no valence
Headline statistics are over the HARM cells only; the others are reported per eval.

Writes paper/generated/selfserving.tex (macros) and table_selfserving.tex (appendix table).  Deterministic (seed 1234).
Usage: PYTHONPATH=. python scripts/selfserving_bias.py [--split test]
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import yaml

from behavior_prediction import ceilings, common, results_io, splits
from behavior_prediction.evaluate import _level_bias, tuning_path
from behavior_prediction.evals import get_spec
from behavior_prediction.selection import _corr

SEED, BOOT = 1234, 10_000
EVALS = ["sycophancy_pushback", "discrimeval", "capability_mmlu", "reward_hacking",
         "tau2_policy", "tau2_transfer", "propensitybench", "mask_subdomain_pressure",
         "agentic_misalignment"]
VALENCE = {"sycophancy_pushback": "harm", "reward_hacking": "harm", "tau2_policy": "harm",
           "propensitybench": "harm", "mask_subdomain_pressure": "harm",
           "agentic_misalignment": "harm", "tau2_transfer": "handoff",
           "capability_mmlu": "credit", "discrimeval": "none"}
SHORT = {"sycophancy_pushback": "Sycophancy", "discrimeval": "DiscrimEval",
         "capability_mmlu": "Capability", "reward_hacking": "Reward hacking",
         "tau2_policy": "$\\tau^2$ policy", "tau2_transfer": "$\\tau^2$ transfer",
         "propensitybench": "PropensityBench", "mask_subdomain_pressure": "MASK",
         "agentic_misalignment": "Agentic misalignment"}
TIERS = {"Abs": ("self_report", "generic_report"), "Inf": ("informed_oracle", "generic_oracle")}
OUT = Path("paper/generated/selfserving.tex")
OUT_TAB = Path("paper/generated/table_selfserving.tex")


def collect(split: str):
    pool = yaml.safe_load(open("behavior_prediction/models.yaml"))["pools"]["default"]
    rows: dict[tuple[str, str], list[tuple[str, float, float]]] = {}  # (tier, eval) -> [(model, b_self, b_gen)]
    for ev in EVALS:
        spec, man = get_spec(ev), splits.load_manifest(ev)
        tp = tuning_path(ev)
        best = ({b: m["best_setting"] for b, m in common.load_json(tp)["methods"].items()
                 if m.get("best_setting")} if tp.exists() else {})
        for ms in pool:
            loaded = results_io.load_cell(ev, ms)
            if not loaded:
                continue
            targets, methods = loaded
            try:
                keys = splits.split_keys(spec, targets, man, split)
            except KeyError:
                continue
            ceil = ceilings.estimate(spec, targets, keys)
            if not ceilings.is_reliable(ceil, ceilings.MIN_RELIABILITY ** 0.5):
                continue
            for tier, (a, b) in TIERS.items():
                pa, pb = methods.get(best.get(a, a)), methods.get(best.get(b, b))
                if pa is None or pb is None:
                    continue
                ks = [k for k in pa if k in pb and pa[k] is not None and pb[k] is not None]
                pa2, pb2 = {k: pa[k] for k in ks}, {k: pb[k] for k in ks}
                if _corr(spec, targets, pa2, keys) is None or _corr(spec, targets, pb2, keys) is None:
                    continue
                ba, bb = _level_bias(spec, targets, pa2, keys), _level_bias(spec, targets, pb2, keys)
                if ba is None or bb is None:
                    continue
                rows.setdefault((tier, ev), []).append((ms, ba, bb))
    return rows


def paired(d: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(d)
    boots = np.array([rng.choice(d, n, replace=True).mean() for _ in range(BOOT)])
    obs = abs(d.mean())
    if n <= 20:
        cnt = sum(abs((d * s).mean()) >= obs - 1e-12
                  for s in itertools.product((1, -1), repeat=n))
        p = cnt / 2 ** n
    else:
        signs = rng.choice((1, -1), size=(100_000, n))
        p = float((np.abs((d * signs).mean(1)) >= obs - 1e-12).mean())
    return {"mean": d.mean(), "lo": np.percentile(boots, 2.5), "hi": np.percentile(boots, 97.5),
            "p": p, "n": n, "npos": int((d > 0).sum()), "nneg": int((d < 0).sum())}


def fmt(x): return f"{x:+.2f}"
def fmtp(p): return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    rows = collect(args.split)
    lines = ["% Auto-generated by scripts/selfserving_bias.py -- DO NOT EDIT BY HAND.",
             f"% Self-framed minus generic-subject signed level bias, {args.split} split, default pool,",
             "% ceiling-filtered cells. Descriptive analysis (added 2026-08-23, not pre-registered)."]
    table = []
    for tier, (a, b) in TIERS.items():
        harm = []
        per_eval = {}
        for ev in EVALS:
            cells = rows.get((tier, ev), [])
            if not cells:
                continue
            bs = np.array([c[1] for c in cells]); bg = np.array([c[2] for c in cells])
            per_eval[ev] = (len(cells), bs.mean(), bg.mean(), (bs - bg).mean())
            if VALENCE[ev] == "harm":
                harm += list(bs - bg)
            key = "".join(ch for ch in ev.replace("_", " ").title() if ch.isalpha())
            lines.append(f"\\newcommand{{\\ssb{tier}Self{key}}}{{{fmt(bs.mean())}}}")
            lines.append(f"\\newcommand{{\\ssb{tier}Gen{key}}}{{{fmt(bg.mean())}}}")
            lines.append(f"\\newcommand{{\\ssb{tier}Diff{key}}}{{{fmt((bs - bg).mean())}}}")
        st = paired(np.array(harm))
        harm_self = np.mean([per_eval[e][1] for e in per_eval if VALENCE[e] == "harm"])
        harm_gen = np.mean([per_eval[e][2] for e in per_eval if VALENCE[e] == "harm"])
        # count of valenced evals whose mean shift has the flattering sign
        flat = sum(1 for e in per_eval if VALENCE[e] in ("harm", "handoff") and per_eval[e][3] < 0) \
            + sum(1 for e in per_eval if VALENCE[e] == "credit" and per_eval[e][3] > 0)
        nval = sum(1 for e in per_eval if VALENCE[e] != "none")
        lines += [f"\\newcommand{{\\ssb{tier}HarmSelf}}{{{fmt(harm_self)}}}",
                  f"\\newcommand{{\\ssb{tier}HarmGen}}{{{fmt(harm_gen)}}}",
                  f"\\newcommand{{\\ssb{tier}HarmDiff}}{{{fmt(st['mean'])}}}",
                  f"\\newcommand{{\\ssb{tier}HarmCILo}}{{{fmt(st['lo'])}}}",
                  f"\\newcommand{{\\ssb{tier}HarmCIHi}}{{{fmt(st['hi'])}}}",
                  f"\\newcommand{{\\ssb{tier}HarmP}}{{{fmtp(st['p'])}}}",
                  f"\\newcommand{{\\ssb{tier}HarmN}}{{{st['n']}}}",
                  f"\\newcommand{{\\ssb{tier}HarmNNeg}}{{{st['nneg']}}}",
                  f"\\newcommand{{\\ssb{tier}FlatteringEvals}}{{{flat}}}",
                  f"\\newcommand{{\\ssb{tier}ValencedEvals}}{{{nval}}}",
                  f"% {tier}: harm-eval cells self-generic = {st['mean']:+.3f} [{st['lo']:+.3f},{st['hi']:+.3f}] "
                  f"p={st['p']:.4f} n={st['n']} ({st['nneg']} negative); flattering sign on {flat}/{nval} valenced evals"]
        table.append((tier, per_eval))
    # appendix table
    t = ["\\begin{table}[t]", "  \\centering",
         "  \\caption{Self-serving shift of predicted levels (" + args.split + " split, tuned settings): "
         "mean signed bias (predicted $-$ actual, rate units) of the self-framed question and of the "
         "identical question about ``capable AI agents in general'', on the same cells, and their "
         "difference. ``High rate means'' labels the direction that would flatter the model: for a harm a "
         "flattering shift is negative, for accuracy positive. Descriptive analysis, not "
         "pre-registered. Auto-generated (\\texttt{paper/generated/table\\_selfserving.tex}).}",
         "  \\label{tab:selfserving}", "  \\footnotesize",
         "  \\setlength{\\tabcolsep}{4pt}",
         "  \\begin{tabular}{llcccccc}", "    \\toprule",
         "    & & \\multicolumn{3}{c}{abstract question} & \\multicolumn{3}{c}{verbatim items} \\\\",
         "    \\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
         "    Evaluation & high rate means & self & generic & self$-$generic & self & generic & self$-$generic \\\\",
         "    \\midrule"]
    labels = {"harm": "misbehavior", "credit": "accuracy (credit)", "handoff": "hand-off to human",
              "none": "signed contrast (none)"}
    per = {tier: pe for tier, pe in table}
    for ev in EVALS:
        cells = [per[t_].get(ev) for t_ in TIERS]
        if not any(cells):
            continue
        row = f"    {SHORT[ev]} & {labels[VALENCE[ev]]}"
        for c in cells:
            row += (f" & {fmt(c[1])} & {fmt(c[2])} & {fmt(c[3])}" if c else " & -- & -- & --")
        t.append(row + " \\\\")
    t += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    OUT.write_text("\n".join(lines) + "\n")
    OUT_TAB.write_text("% Auto-generated by scripts/selfserving_bias.py -- DO NOT EDIT BY HAND.\n" + "\n".join(t) + "\n")
    print("\n".join(l for l in lines if l.startswith("%")))
    print("\n".join(t))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
