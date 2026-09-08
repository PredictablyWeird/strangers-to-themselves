#!/usr/bin/env python3
"""Compare predicted misalignment rates against actual behavior (single model).

Loads the target file (actual rates) and every prediction file in the predictions directory,
then prints and writes a markdown table with, per condition, the actual rate and each method's
predicted rate + absolute error, plus per-method MAE overall and by scenario, and a breakdown
by each swept axis.

**Extension point:** each ``*.json`` in ``--predictions-dir`` that conforms to the shared
prediction schema (see ``common.py``) becomes one method = one column group. Use ``--eval`` to
select which eval's ordering/axes to use (default: agentic_misalignment).

Examples
--------
    .venv/bin/python compare_predictions.py
    .venv/bin/python compare_predictions.py --targets results/targets_cv.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from behavior_prediction import common
from behavior_prediction import metrics
from behavior_prediction.evals import add_eval_arg, resolve_spec_from_argv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_eval_arg(p)
    p.add_argument("--targets", default="results/targets.json", help="target file (default: results/targets.json)")
    p.add_argument("--predictions-dir", default="results/predictions",
                   help="dir of prediction files (default: results/predictions)")
    p.add_argument("--out", default="results/comparison.md", help="output markdown path (default: results/comparison.md)")
    p.add_argument("--results-root", default="results",
                   help="results root (default: results)")
    return p.parse_args()


def fmt_rate(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def axis_sorted(axis: str, values: list, spec) -> list:
    order = spec.report_axis_order.get(axis)
    if order:
        return sorted(values, key=lambda v: order.index(v) if v in order else 99)
    return sorted(values, key=lambda v: str(v))


def mean_or_none(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def main() -> int:
    args = parse_args()
    spec = resolve_spec_from_argv()

    target_doc = common.load_json(args.targets)
    metric = target_doc["metric"]
    target_model = target_doc["model"]
    targets = target_doc["targets"]

    # Order condition keys by scenario, then the eval's secondary axis, then key.
    sec = spec.report_secondary_axis
    sec_order = spec.report_axis_order.get(sec, []) if sec else []

    def order_key(item: tuple[str, dict[str, Any]]) -> tuple:
        key, t = item
        scen = t.get("scenario", "")
        sv = (t.get("condition") or {}).get(sec) if sec else None
        si = spec.scenarios.index(scen) if scen in spec.scenarios else 99
        gi = sec_order.index(sv) if sv in sec_order else 99
        return (si, gi, key)

    keys = [k for k, _ in sorted(targets.items(), key=order_key)]
    actual = {k: targets[k].get("rate") for k in keys}
    scen_of = {k: targets[k].get("scenario", "") for k in keys}
    label_of = {k: spec.condition_label(targets[k]["condition"]) for k in keys}

    # --- load prediction methods ---
    pred_files = sorted(Path(args.predictions_dir).glob("*.json"))
    methods: list[dict[str, Any]] = []
    warnings: list[str] = []
    for pf in pred_files:
        doc = common.load_json(pf)
        if doc.get("metric") != metric:
            warnings.append(
                f"{pf.name}: predicts metric={doc.get('metric')!r} but targets are "
                f"metric={metric!r} -- included anyway, interpret with care."
            )
        if doc.get("model") != target_model:
            warnings.append(
                f"{pf.name}: model={doc.get('model')!r} differs from target model "
                f"{target_model!r}."
            )
        dpreds = doc.get("predictions", {})
        preds = {k: dpreds.get(k, {}).get("predicted_rate") for k in keys}
        missing = [k for k in keys if preds[k] is None]
        if missing:
            warnings.append(f"{pf.name}: no prediction for {len(missing)}/{len(keys)} conditions.")
        methods.append({"label": doc.get("method", pf.stem), "preds": preds})

    # --- build markdown ---
    lines: list[str] = []
    lines.append(f"# Prediction vs. actual behavior ({metric})")
    lines.append("")
    lines.append(f"- **Model:** `{target_model}`")
    lines.append(f"- **Metric:** `{metric}` (actual rates from {args.targets})")
    lines.append(f"- **Conditions:** {len(keys)}")
    lines.append("")

    header = ["Condition", f"Actual ({metric})"]
    for m in methods:
        header += [f"{m['label']} pred", f"{m['label']} |err|"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    errs_all: dict[str, list[float]] = {m["label"]: [] for m in methods}
    errs_scen: dict[str, dict[str, list[float]]] = {m["label"]: {} for m in methods}
    for k in keys:
        row = [label_of[k], fmt_rate(actual[k])]
        for m in methods:
            pred = m["preds"][k]
            row.append(fmt_rate(pred))
            if pred is None or actual[k] is None:
                row.append("n/a")
            else:
                err = abs(pred - actual[k])
                errs_all[m["label"]].append(err)
                errs_scen[m["label"]].setdefault(scen_of[k], []).append(err)
                row.append(f"{err * 100:.1f} pts")
        lines.append("| " + " | ".join(row) + " |")

    mae_row = ["**MAE (all)**", ""]
    for m in methods:
        errs = errs_all[m["label"]]
        mae_row += ["", (f"**{mean(errs) * 100:.1f} pts**" if errs else "n/a")]
    lines.append("| " + " | ".join(mae_row) + " |")
    lines.append("")

    # --- overall metrics: MAE + correlation per method, vs constant baselines ---
    def fmt_pts(x: float | None) -> str:
        return "n/a" if x is None else f"{x * 100:.1f}"

    def fmt_corr(r: float | None) -> str:
        return "— (constant)" if r is None else f"{r:+.2f}"

    lines.append("## Overall metrics")
    lines.append("")
    lines.append("| Method | MAE (pts) | Correlation |")
    lines.append("|---|---|---|")
    for m in methods:
        lines.append(f"| {m['label']} | {fmt_pts(metrics.mae(targets, m['preds']))} "
                     f"| {fmt_corr(metrics.correlation(targets, m['preds']))} |")
    for blabel, bconst in metrics.BASELINES.items():
        lines.append(f"| _baseline: {blabel}_ | {fmt_pts(metrics.baseline_mae(targets, bconst))} | — |")
    lines.append("")
    lines.append("- **Correlation** = Pearson r between predicted and actual rate across conditions "
                 "(higher = better). `— (constant)` means the method's predictions have no variance "
                 "(it outputs the same number for every condition), so correlation is undefined.")
    lines.append("- **Baselines** = MAE of a predictor that always outputs that fixed rate; a method "
                 "that doesn't beat the best baseline carries no condition-level signal.")
    lines.append("")

    # per-scenario MAE
    lines.append("## MAE by scenario (percentage points)")
    lines.append("")
    sh = ["Scenario"] + [m["label"] for m in methods]
    lines.append("| " + " | ".join(sh) + " |")
    lines.append("|" + "|".join(["---"] * len(sh)) + "|")
    for scen in [s for s in spec.scenarios if any(scen_of[k] == s for k in keys)]:
        row = [scen]
        for m in methods:
            errs = errs_scen[m["label"]].get(scen, [])
            row.append(f"{mean(errs) * 100:.1f}" if errs else "n/a")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- breakdown by each swept axis, aggregated over scenarios ---
    cond_of = {k: targets[k].get("condition") or {} for k in keys}
    swept_axes = [ax for ax in spec.report_candidate_axes
                  if len({cond_of[k].get(ax) for k in keys}) > 1]
    for ax in swept_axes:
        lines.append(f"## By {ax} (mean across {len(spec.scenarios)} scenarios)")
        lines.append("")
        head = [ax, "n", f"Actual ({metric})"]
        for m in methods:
            head += [f"{m['label']} pred", f"{m['label']} MAE"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        values = axis_sorted(ax, list({cond_of[k].get(ax) for k in keys}), spec)
        for val in values:
            grp = [k for k in keys if cond_of[k].get(ax) == val]
            actual_mean = mean_or_none([actual[k] for k in grp if actual[k] is not None])
            row = [str(val), str(len(grp)), fmt_rate(actual_mean)]
            for m in methods:
                pvals = [m["preds"][k] for k in grp if m["preds"][k] is not None]
                errs = [abs(m["preds"][k] - actual[k]) for k in grp
                        if m["preds"][k] is not None and actual[k] is not None]
                row.append(fmt_rate(mean_or_none(pvals)))
                row.append(f"{mean(errs) * 100:.1f} pts" if errs else "n/a")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"Wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
