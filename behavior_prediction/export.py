#!/usr/bin/env python3
"""Machine-readable export of the evaluation leaderboard.

``bp-evaluate`` renders markdown + HTML for humans; this module turns the *same* computed object
(``evaluate.compute()`` -> the ``c`` dict) into artifacts a build pipeline / the paper can consume:

  * ``build_results_doc(c, split)`` -> a flat, JSON-serializable dict (cells, per-method macro
    means + bootstrap CIs, per-eval best method).
  * ``write_json(doc, path)``       -> the canonical machine-readable artifact.
  * ``write_latex_macros(doc, path)`` -> one ``\\newcommand`` per headline number so the paper can
    ``\\input`` it instead of hand-copying figures.

Nothing here calls a model or re-scores anything; it is a pure reshaping of ``c``.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from behavior_prediction import method_names, results_io


def build_results_doc(c: dict[str, Any], split: str) -> dict[str, Any]:
    """Reshape ``evaluate.compute()`` output into a flat, JSON-serializable results document.

    Primary metric is the mean correlation (``mean_r``); the secondary ceiling-normalized score
    (``r/ceiling``, fraction of reachable signal captured) is aggregated by **median**
    (``median_norm``) — a median is robust to the heavy tail a ratio picks up near small ceilings.
    Weak cells removed by :func:`evaluate.compute` never enter either aggregate; they are carried
    through as ``dropped_weak`` so the exclusion is auditable, alongside the per-cell ``ceilings``."""
    cells, methods, evals, models = c["cells"], c["methods"], c["evals"], c["models"]
    cells_norm = c.get("cells_norm", {})
    ceiling_map = c.get("ceilings", {})
    settings = c["settings"]

    def cell_ceiling(ev: str, ms: str) -> float | None:
        rec = ceiling_map.get((ev, ms))
        return rec["ceiling"] if rec else None

    # Flat per-cell records: correlation + ceiling + normalized score per (eval, model, method).
    cell_records = [
        {"eval": ev, "model": ms, "method": m, "setting": settings.get((ev, m), m), "r": r,
         "ceiling": cell_ceiling(ev, ms), "r_norm": cells_norm.get((ev, ms, m))}
        for (ev, ms, m), r in sorted(cells.items())
    ]

    # Per-method macro mean over its (eval, model) cells (primary), with a bootstrap CI; plus the
    # median ceiling-normalized score over the same cells (secondary).
    cells_mae = c.get("cells_mae", {})
    cells_bias = c.get("cells_bias", {})
    per_method = []
    for m in methods:
        vals = [r for (ev, ms, mm), r in cells.items() if mm == m]
        if not vals:
            continue
        ci = results_io.bootstrap_cells(vals)
        norms = [v for (ev, ms, mm), v in cells_norm.items() if mm == m]
        maes = [v for (ev, ms, mm), v in cells_mae.items() if mm == m]
        biases = [v for (ev, ms, mm), v in cells_bias.items() if mm == m]
        per_method.append({
            "method": m,
            "mean_r": mean(vals),
            "ci_lo": ci[0] if ci else None,
            "ci_hi": ci[1] if ci else None,
            "median_norm": median(norms) if norms else None,
            # Calibration (rate-emitting methods only; see MethodAdapter.calibrated).
            "mae": mean(maes) if maes else None,
            "bias": mean(biases) if biases else None,
            "n": len(vals),
            "n_norm": len(norms),
        })

    # Per (method, eval) mean correlation over models — fills the leaderboard table's body.
    per_method_eval = []
    for m in methods:
        for ev in evals:
            vals = [r for (e, ms, mm), r in cells.items() if e == ev and mm == m]
            per_method_eval.append({"method": m, "eval": ev,
                                    "mean_r": mean(vals) if vals else None})

    # Per (method, model) mean correlation over evals — fills the method×model table body.
    per_method_model = []
    for m in methods:
        for ms in models:
            vals = [r for (e, mms, mm), r in cells.items() if mms == ms and mm == m]
            per_method_model.append({"method": m, "model": ms,
                                     "mean_r": mean(vals) if vals else None})

    # Per-eval best method: marginalize over models, then take the highest-correlation method.
    per_eval_best = []
    for ev in evals:
        scored = []
        for m in methods:
            vals = [r for (e, ms, mm), r in cells.items() if e == ev and mm == m]
            if vals:
                scored.append((m, mean(vals)))
        if scored:
            best_m, best_r = max(scored, key=lambda mr: mr[1])
            per_eval_best.append({"eval": ev, "best_method": best_m, "r": best_r})

    return {
        "split": split,
        "models": models,
        "methods": methods,
        "evals": evals,
        "dropped": c["dropped"],
        "dropped_weak": c.get("dropped_weak", []),
        "filter_weak": c.get("filter_weak", False),
        "min_ceiling": c.get("min_ceiling", 0.7),
        "missing_tuning": c["missing_tuning"],
        "ceilings": [{"eval": ev, "model": ms, **(rec or {})}
                     for (ev, ms), rec in sorted(ceiling_map.items())],
        "cells": cell_records,
        "per_method": per_method,
        "per_method_eval": per_method_eval,
        "per_method_model": per_method_model,
        "per_eval_best": per_eval_best,
    }


def write_json(doc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")


# --- LaTeX macros -------------------------------------------------------------------------------

def _camel(s: str) -> str:
    """``self_report`` / ``llama-3.3-70b`` -> ``selfReport`` / ``llama3370b`` (macro-name safe)."""
    parts = [p for p in s.replace("-", "_").replace(".", "_").split("_") if p]
    if not parts:
        return "x"
    head, *tail = parts
    name = head.lower() + "".join(p.capitalize() for p in tail)
    # LaTeX command names must be letters only; drop any remaining non-letters.
    name = "".join(ch for ch in name if ch.isalpha())
    return name or "x"


def _macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def latex_macros(doc: dict[str, Any]) -> str:
    """Build the ``\\newcommand`` body. Numbers are formatted with a sign and 2 decimals."""
    lines = [
        "% Auto-generated by behavior_prediction.export — DO NOT EDIT BY HAND.",
        f"% split: {doc['split']}  models: {len(doc['models'])}  methods: {len(doc['methods'])}",
        "",
    ]

    def num(x: float | None) -> str:
        return f"{x:+.2f}" if x is not None else "NA"

    # Per-method macro mean + CI (primary) and median ceiling-normalized score (secondary),
    # e.g. \resultSelfReportMeanR, \resultSelfReportCILo/Hi, \resultSelfReportMedianNorm.
    for row in doc["per_method"]:
        base = "result" + _camel(row["method"]).capitalize()
        lines.append(_macro(base + "MeanR", num(row["mean_r"])))
        lines.append(_macro(base + "CILo", num(row["ci_lo"])))
        lines.append(_macro(base + "CIHi", num(row["ci_hi"])))
        lines.append(_macro(base + "MedianNorm", num(row.get("median_norm"))))
        # Calibration secondary metrics (NA for ranking-score methods).
        mae, bias = row.get("mae"), row.get("bias")
        lines.append(_macro(base + "Mae", f"{mae:.2f}" if mae is not None else "NA"))
        lines.append(_macro(base + "Bias", num(bias)))
        lines.append(_macro(base + "N", str(row["n"])))

    lines.append("")
    # Per (method, eval) cell, e.g. \resultSelfreportDiscrimeval — fills the leaderboard table.
    for row in doc["per_method_eval"]:
        base = "result" + _camel(row["method"]).capitalize() + _camel(row["eval"]).capitalize()
        lines.append(_macro(base, num(row["mean_r"])))

    lines.append("")
    # Per-eval best method + its correlation, e.g. \resultDiscrimevalBestMethod / BestR.
    for row in doc["per_eval_best"]:
        base = "result" + _camel(row["eval"]).capitalize()
        lines.append(_macro(base + "BestMethod", row["best_method"].replace("_", r"\_")))
        lines.append(_macro(base + "BestR", num(row["r"])))

    return "\n".join(lines) + "\n"


def write_latex_macros(doc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(latex_macros(doc))


# --- Method x model table -----------------------------------------------------------------------

def _tex(s: str) -> str:
    """Escape a name for LaTeX text (underscores are the only special char in our method/model ids)."""
    return s.replace("_", r"\_")


def method_model_table(doc: dict[str, Any]) -> str:
    """A complete, self-contained ``table`` (method rows x model columns, mean r over evals, plus a
    row mean). Emitted as a full ``tabular`` rather than fixed macros because the model set is
    dynamic — the paper just ``\\input``s it."""
    methods, models = doc["methods"], doc["models"]
    # (method, model) -> mean_r lookup from the flat per_method_model list.
    cell = {(row["method"], row["model"]): row["mean_r"] for row in doc["per_method_model"]}

    def num(x: float | None) -> str:
        return f"{x:+.2f}" if x is not None else "--"

    lines = [
        "% Auto-generated by behavior_prediction.export — DO NOT EDIT BY HAND.",
        r"\begin{table}[t]",
        r"  \centering",
        (r"  \caption{Prediction correlation by method and model (" + doc["split"] + r" split; mean over evaluations, "
         r"each method at its tuned best setting). Higher = better ranking; \texttt{--} = no cell "
         r"for that (method, model). Auto-generated (\texttt{paper/generated/table\_method\_model.tex}).}"),
        r"  \label{tab:method-model}",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{2pt}",
        r"  \begin{tabular}{l" + "c" * len(models) + "c}",
        r"    \toprule",
        "    Method & " + " & ".join(r"\rotatebox{90}{" + _tex(ms) + "}" for ms in models)
        + r" & Mean \\",
        r"    \midrule",
    ]
    for m in methods:
        present = [cell[(m, ms)] for ms in models if cell.get((m, ms)) is not None]
        row_mean = num(mean(present)) if present else "--"
        cells = " & ".join(num(cell.get((m, ms))) for ms in models)
        lines.append(f"    {method_names.display(m)} & {cells} & {row_mean} " + r"\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_method_model_table(doc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(method_model_table(doc))


# --- Calibration (MAE / signed bias) appendix table -----------------------------------------

def calibration_table(doc: dict[str, Any]) -> str:
    """Appendix table of the calibration secondary metrics: per-method mean absolute error and
    mean signed bias over its scored (eval, model) cells. Rate-emitting methods only — ranking
    scores (pairwise variants, ensembles) carry no absolute scale and render as ``--``."""
    lines = [
        "% Auto-generated by behavior_prediction.export — DO NOT EDIT BY HAND.",
        r"\begin{table}[t]",
        r"  \centering",
        (r"  \caption{Calibration of the rate-emitting methods (" + doc["split"] + r" split, tuned settings): mean "
         r"absolute error and mean signed bias (predicted $-$ actual, in rate units) over each "
         r"method's scored (evaluation, model) cells. Negative bias = systematic understatement. "
         r"Ranking-score methods (paired comparisons, ensembles) have no absolute scale "
         r"(\texttt{--}). Auto-generated (\texttt{paper/generated/table\_bias.tex}).}"),
        r"  \label{tab:bias}",
        r"  \footnotesize",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Method & MAE & signed bias & cells \\",
        r"    \midrule",
    ]
    def num(x):
        return f"{x:+.2f}" if x is not None else "--"
    for r in sorted(doc["per_method"], key=lambda r: (r.get("mae") is None, r.get("mae") or 0)):
        mae = f"{r['mae']:.2f}" if r.get("mae") is not None else "--"
        lines.append(f"    {method_names.display(r['method'])} & {mae} & {num(r.get('bias'))} "
                     f"& {r['n']} " + r"\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_calibration_table(doc: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(calibration_table(doc))
