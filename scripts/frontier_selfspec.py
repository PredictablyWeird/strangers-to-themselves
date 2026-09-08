"""Scale and reasoning analysis on the merged pool: does capability buy self-knowledge?

Instruments, all computed from committed artifacts on the dev split:
  * per-tier means of the key methods (small tier vs frontier-off vs frontier-low);
  * the direct subject control at every setting: item-informed self-prediction vs its
    generic-subject variant (self minus generic per cell);
  * identity transfer -- each frontier setting's item-informed predictions scored against its
    OWN behavior (diag) vs each small-tier model's behavior (off); advantage = diag - off;
  * partial correlation given the cross-model mean (unique signal net of the pool prior);
  * reasoning deltas -- r(low) - r(off) per base model and method, pooled over shared evals.

Cells restricted to those scored by the merged evaluation (results/reports/evaluation.json).

Writes results/reports/frontier_selfspec.txt, paper/generated/frontier_selfspec.tex (macros)
and paper/generated/table_scale.tex (the scale-analysis table).
"""
import json
from pathlib import Path
from statistics import mean

import yaml

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["discrimeval", "propensitybench", "capability_mmlu", "sycophancy_pushback",
         "reward_hacking", "tau2_policy"]
FRONTIER = ["claude-sonnet-5-off", "claude-sonnet-5-low", "gpt-5.5-off", "gpt-5.5-low",
            "deepseek-v4-pro-off", "deepseek-v4-pro-low"]
PAIRS = [("claude-sonnet-5", "Sonnet"), ("gpt-5.5", "Gpt"), ("deepseek-v4-pro", "Ds")]
SMALL = list(yaml.safe_load(open("behavior_prediction/models.yaml"))["pools"]["small"])
REASON_METHODS = ["self_report", "generic_report", "pairwise", "few_shot",
                  "informed_oracle", "generic_oracle", "cross_model_mean"]
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--split", choices=["dev", "test"], default="dev")
SPLIT = _ap.parse_args().split
_sfx = "" if SPLIT == "dev" else "_test"
OUT_TXT = Path(f"results/reports/frontier_selfspec{_sfx}.txt")
OUT_TEX = Path(f"paper/generated/frontier_selfspec{_sfx}.tex")
OUT_TAB = Path(f"paper/generated/table_scale{_sfx}.tex")

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def tuned(ev, m):
    try:
        return common.load_json(Path(f"results/{ev}/tuning.json"))["methods"][m]["best_setting"]
    except Exception:
        return m


def preds(ev, slug, fname):
    p = Path(f"results/{ev}/{slug}/predictions/{fname}.json")
    if not p.exists():
        return {}
    return {k: v.get("predicted_rate") for k, v in common.load_json(p)["predictions"].items()
            if isinstance(v, dict) and v.get("predicted_rate") is not None}


def preds_scored(spec, ev, slug, fname):
    """Predictions in the eval's scoring space (bias evals: differenced to contrast grain)."""
    pv = preds(ev, slug, fname)
    if getattr(spec, "scoring_semantics", "") != "bias_contrast":
        return pv
    t = common.load_json(Path(f"results/{ev}/{slug}/targets.json"))["targets"]
    dev = set(splits.split_keys(spec, t, splits.load_manifest(ev), SPLIT))
    sc = spec.score_contrasts(t, pv, keys=dev)
    contrast_pv = {k: p for k, (_, p) in sc.items() if p is not None}
    return contrast_pv if contrast_pv else pv


def actuals(spec, ev, slug):
    p = Path(f"results/{ev}/{slug}/targets.json")
    if not p.exists():
        return {}
    t = common.load_json(p)["targets"]
    dev = set(splits.split_keys(spec, t, splits.load_manifest(ev), SPLIT))
    if getattr(spec, "scoring_semantics", "") == "bias_contrast":
        sc = spec.score_contrasts(t, {k: v.get("rate") for k, v in t.items()}, keys=dev)
        return {k: a for k, (a, _) in sc.items() if a is not None}
    return {k: v["rate"] for k, v in t.items() if k in dev and v.get("rate") is not None}


def residual(y, x, keys):
    ks = [k for k in keys if k in y and k in x]
    if len(ks) < 3:
        return {}
    mx, my = mean(x[k] for k in ks), mean(y[k] for k in ks)
    sxx = sum((x[k] - mx) ** 2 for k in ks)
    b = 0.0 if sxx == 0 else sum((x[k] - mx) * (y[k] - my) for k in ks) / sxx
    return {k: y[k] - (my + b * (x[k] - mx)) for k in ks}


def partial(a, b, ctrl, keys):
    ra, rb = residual(a, ctrl, keys), residual(b, ctrl, keys)
    ks = sorted(set(ra) & set(rb))
    return metrics.pearson([(ra[k], rb[k]) for k in ks])


def fmt(x, w=7):
    return (f"{x:+.2f}".rjust(w) if x is not None else "--".rjust(w))


def main() -> int:
    doc = json.load(open("results/reports/evaluation.json"))
    scored = {(c["eval"], c["model"], c["method"]) for c in doc["cells"]}
    cell_r = {(c["eval"], c["model"], c["method"]): c["r"] for c in doc["cells"]}

    def ok(ev, slug, method="informed_oracle"):
        return (ev, slug, method) in scored

    # ---- identity transfer + partial + self-vs-generic per frontier setting ---------------
    def analyze(slug, donors):
        acc = {"diag": [], "off": [], "adv": [], "part": [], "sg": []}
        for ev in EVALS:
            if not ok(ev, slug):
                continue
            spec = get_spec(ev)
            pv = preds_scored(spec, ev, slug, tuned(ev, "informed_oracle"))
            own = actuals(spec, ev, slug)
            ks = sorted(set(pv) & set(own))
            diag = metrics.pearson([(own[k], pv[k]) for k in ks]) if len(ks) >= 3 else None
            offs = []
            for donor in donors:
                if donor == slug or not ok(ev, donor):
                    continue
                dact = actuals(spec, ev, donor)
                dks = sorted(set(pv) & set(dact))
                r = metrics.pearson([(dact[k], pv[k]) for k in dks]) if len(dks) >= 3 else None
                if r is not None:
                    offs.append(r)
            off = mean(offs) if offs else None
            xmm = preds_scored(spec, ev, slug, tuned(ev, "cross_model_mean"))
            pks = sorted(set(pv) & set(own) & set(xmm))
            part = partial(own, pv, xmm, pks) if len(pks) >= 5 else None
            adv = (diag - off) if diag is not None and off is not None else None
            sg = None
            if (ev, slug, "generic_oracle") in scored and (ev, slug, "informed_oracle") in scored:
                sg = cell_r[(ev, slug, "informed_oracle")] - cell_r[(ev, slug, "generic_oracle")]
            emit(f"{slug:<24}{ev:<20}{fmt(diag)}{fmt(off)}{fmt(adv)}{fmt(part, 10)}{fmt(sg, 9)}")
            for key, v in (("diag", diag), ("off", off), ("adv", adv), ("part", part),
                           ("sg", sg)):
                if v is not None:
                    acc[key].append(v)
        return acc

    per_setting = {}
    emit(f"{'setting':<24}{'eval':<20}{'diag':>7}{'off':>7}{'adv':>7}{'p(io|xmm)':>10}"
         f"{'self-gen':>9}")
    for slug in FRONTIER:
        acc = analyze(slug, SMALL)
        per_setting[slug] = {k: (mean(v) if v else None) for k, v in acc.items()} | {
            "n": len(acc["adv"])}
        emit()
    small_acc = {"diag": [], "off": [], "adv": [], "part": [], "sg": []}
    for slug in SMALL:
        acc = analyze(slug, SMALL)
        for k in small_acc:
            small_acc[k] += acc[k]
    small_ref = {k: (mean(v) if v else None) for k, v in small_acc.items()}
    emit("SMALL-TIER REFERENCE (within-tier transfer): "
         f"diag {fmt(small_ref['diag'])}  off {fmt(small_ref['off'])}  "
         f"adv {fmt(small_ref['adv'])}  part {fmt(small_ref['part'])}  "
         f"self-gen {fmt(small_ref['sg'])}")
    emit()

    off_tier = [per_setting[s] for s in FRONTIER if s.endswith("-off")]
    low_tier = [per_setting[s] for s in FRONTIER if s.endswith("-low")]

    def tier_mean(tier, key):
        vs = [t[key] for t in tier if t[key] is not None]
        return mean(vs) if vs else None

    emit("TIER MEANS (reasoning off vs low):")
    for key in ("diag", "off", "adv", "part", "sg"):
        emit(f"  {key:<6} off {fmt(tier_mean(off_tier, key))}   low {fmt(tier_mean(low_tier, key))}")

    # ---- matched-composition tier comparison (four text evals, all models) ----------------
    import random as _random
    TEXT = ["sycophancy_pushback", "discrimeval", "capability_mmlu", "reward_hacking"]

    def text_mean(m):
        rs = [cell_r[(e, m, "informed_oracle")] for e in TEXT
              if (e, m, "informed_oracle") in cell_r]
        return mean(rs) if rs else None
    ftxt = [v for m in FRONTIER if (v := text_mean(m)) is not None]
    stxt = [v for m in SMALL if (v := text_mean(m)) is not None]
    _rng = _random.Random(1234)
    boots = sorted(mean(_rng.choices(ftxt, k=len(ftxt))) - mean(_rng.choices(stxt, k=len(stxt)))
                   for _ in range(10000))
    lo, hi = boots[249], boots[9749]
    emit(f"MATCHED tier comparison (informed, four text evals): frontier {mean(ftxt):+.3f} "
         f"small {mean(stxt):+.3f} diff {mean(ftxt)-mean(stxt):+.3f} 95% CI [{lo:+.3f},{hi:+.3f}]")
    emit()

    # ---- reasoning deltas per base model and method ---------------------------------------
    emit()
    emit("REASONING DELTAS r(low) - r(off), pooled over evals scored at both settings:")
    emit(f"{'method':<20}" + "".join(f"{p[1]:>9}" for p in PAIRS) + f"{'mean':>9}")
    reason = {}
    for method in REASON_METHODS:
        row = []
        for base, tag in PAIRS:
            ds = [cell_r[(ev, f"{base}-low", method)] - cell_r[(ev, f"{base}-off", method)]
                  for ev in EVALS
                  if (ev, f"{base}-low", method) in cell_r
                  and (ev, f"{base}-off", method) in cell_r]
            row.append(mean(ds) if ds else None)
        vals = [v for v in row if v is not None]
        reason[method] = row + [mean(vals) if vals else None]
        emit(f"{method:<20}" + "".join(fmt(v, 9) for v in reason[method]))

    # ---- macros + scale table -------------------------------------------------------------
    macros = ["% Auto-generated by scripts/frontier_selfspec.py -- DO NOT EDIT BY HAND.",
              "% Scale/reasoning analysis on the merged pool: identity transfer, partials,",
              "% self-vs-generic, and reasoning deltas; scored cells only."]
    short = {"claude-sonnet-5-off": "SonnetOff", "claude-sonnet-5-low": "SonnetLow",
             "gpt-5.5-off": "GptOff", "gpt-5.5-low": "GptLow",
             "deepseek-v4-pro-off": "DsOff", "deepseek-v4-pro-low": "DsLow"}
    keytag = (("diag", "Diag"), ("off", "Off"), ("adv", "Adv"), ("part", "Part"),
              ("sg", "SelfGen"))
    for slug in FRONTIER:
        for key, tag in keytag:
            v = per_setting[slug][key]
            if v is not None:
                macros.append(f"\\newcommand{{\\frsel{tag}{short[slug]}}}{{{v:+.2f}}}")
    for tier, name in ((off_tier, "OffTier"), (low_tier, "LowTier")):
        for key, tag in keytag:
            v = tier_mean(tier, key)
            if v is not None:
                macros.append(f"\\newcommand{{\\frsel{tag}{name}}}{{{v:+.2f}}}")
    for key, tag in keytag:
        if small_ref[key] is not None:
            macros.append(f"\\newcommand{{\\frsel{tag}Small}}{{{small_ref[key]:+.2f}}}")
    macros.append(f"\\newcommand{{\\frselIoTextFrontier}}{{{mean(ftxt):+.2f}}}")
    macros.append(f"\\newcommand{{\\frselIoTextSmall}}{{{mean(stxt):+.2f}}}")
    macros.append(f"\\newcommand{{\\frselIoTextDiff}}{{{mean(ftxt)-mean(stxt):+.2f}}}")
    macros.append(f"\\newcommand{{\\frselIoTextCILo}}{{{lo:+.2f}}}")
    macros.append(f"\\newcommand{{\\frselIoTextCIHi}}{{{hi:+.2f}}}")
    txt_macro = {"claude-sonnet-5-off": "SonnetOff", "claude-sonnet-5-low": "SonnetLow",
                 "gpt-5.5-off": "GptOff", "gpt-5.5-low": "GptLow",
                 "deepseek-v4-pro-off": "DsOff", "deepseek-v4-pro-low": "DsLow",
                 "qwen3.7-plus-low": "Qwen", "gpt-5.4-nano-low": "Nano",
                 "llama-3.3-70b": "Llama", "llama-4-maverick": "Maverick",
                 "deepseek-v4-flash-low": "Flash", "gemini-3.1-flash-lite-low": "Gemini"}
    for m, tag in txt_macro.items():
        v = text_mean(m)
        if v is not None:
            macros.append(f"\\newcommand{{\\frselIoText{tag}}}{{{v:+.2f}}}")
    for method, row in reason.items():
        tag = "".join(w.capitalize() for w in method.split("_"))
        if row[-1] is not None:
            macros.append(f"\\newcommand{{\\rdelta{tag}}}{{{row[-1]:+.2f}}}")
        for (base, ptag), v in zip(PAIRS, row[:-1]):
            if v is not None:
                macros.append(f"\\newcommand{{\\rdelta{tag}{ptag}}}{{{v:+.2f}}}")

    # scale table: tier x instrument
    mm = {(x["method"], x["model"]): x["mean_r"] for x in doc["per_method_model"]}

    def tier_method_mean(models, meth):
        vs = [mm[(meth, m)] for m in models if (meth, m) in mm]
        return mean(vs) if vs else None
    fr_off = [s for s in FRONTIER if s.endswith("-off")]
    fr_low = [s for s in FRONTIER if s.endswith("-low")]
    rows = [("Small tier", SMALL, small_ref, small_ref),
            ("Frontier (off)", fr_off, {k: tier_mean(off_tier, k) for k, _ in keytag},
             None),
            ("Frontier (low)", fr_low, {k: tier_mean(low_tier, k) for k, _ in keytag},
             None)]
    tab = ["% Auto-generated by scripts/frontier_selfspec.py -- DO NOT EDIT BY HAND.",
           "\\begin{table}[t]", "  \\centering",
           "  \\caption{Scale analysis (" + SPLIT + " split; tier means over scored cells). Left:"
           " mean $r$ of the key methods. Right: the self-specificity instruments for"
           " item-informed prediction --- self minus generic-subject ($\\Delta$self--gen),"
           " identity-transfer advantage (own minus small-tier behavior), and the partial"
           " $r$ given the cross-model mean. Capability moves the left columns; no"
           " instrument on the right moves with it.}",
           "  \\label{tab:scale}", "  \\scriptsize", "  \\setlength{\\tabcolsep}{2pt}",
           "  \\begin{tabular}{lcccc|ccc}", "    \\toprule",
           "    Tier & self-rep. & item-inf. & generic & x-model &"
           " $\\Delta$s--g & transfer & partial \\\\", "    \\midrule"]
    for name, models, inst, _ in rows:
        cells = [tier_method_mean(models, m) for m in
                 ("self_report", "informed_oracle", "generic_oracle", "cross_model_mean")]
        vals = [inst.get("sg"), inst.get("adv"), inst.get("part")]
        tab.append(f"    {name} & " + " & ".join(
            f"${v:+.2f}$" if v is not None else "--" for v in cells + vals) + " \\\\")
    tab += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]

    OUT_TXT.write_text("\n".join(lines) + "\n")
    OUT_TEX.write_text("\n".join(macros) + "\n")
    OUT_TAB.write_text("\n".join(tab) + "\n")
    print(f"\nWrote {OUT_TXT}, {OUT_TEX} and {OUT_TAB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
