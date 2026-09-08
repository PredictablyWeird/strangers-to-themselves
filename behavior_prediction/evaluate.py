#!/usr/bin/env python3
"""Evaluation: score the tuned best setting of each method on the eval datasets.

This replaces the old reports (``bp-report`` / ``bp-overview`` / leaderboard). It does **not**
select settings — selection happens in ``bp-tune`` and is frozen in ``results/<eval>/tuning.json``.
Here we take each method's ``best_setting``, load its cached predictions for every model, and score
them on the requested split (default **dev**; ``--split test`` for the final, held-out run).

One markdown leaderboard is emitted to ``results/reports/evaluation.md`` (one row per method, using
its tuned setting): per-method global mean ± bootstrap CI, method×eval and method×model matrices,
and the best method per (model, eval). Correlation is the primary metric (constant predictions count
as 0, no-overlap cells dropped); aggregation is macro over model×eval cells. Calls no models.

Which models share one table is a **pool** (``--pool``, declared under ``pools:`` in models.yaml;
``--pool all`` scores everything on disk). The ``default`` pool is the model set the paper's numbers
describe, and it alone writes ``evaluation.*`` plus ``paper/generated/*.tex``; every other pool
writes ``evaluation_<pool>.*`` and no paper file. So::

    bp-evaluate                              # the paper's pool -> canonical artifacts
    bp-evaluate --pool introspection         # base-vs-finetuned -> evaluation_introspection.*
    bp-evaluate --pool default_plus_introspection   # both, in one table
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median
from typing import Any

from behavior_prediction import ceilings, common, metrics, results_io, selection, splits
from behavior_prediction import methods as method_registry
from behavior_prediction.evals import REGISTRY as EVALS, get_spec
# ACTIVE_EVALS is single-sourced in tuning.py (2026-08-11) so a bare `bp-evaluate` and a bare
# `bp-tune` can never drift apart: the set MUST match the committed canonical artifacts
# (evaluation.* + paper/generated/*.tex), or a default run silently rewrites the paper's
# numbers from a subset (which happened 2026-07-24 when a stale list lingered here).
from behavior_prediction.tuning import ACTIVE_EVALS, tuning_path



def _level_bias(spec, targets: dict[str, Any], preds: dict[str, float | None],
                keys: set[str] | None = None) -> float | None:
    """Signed level bias: mean(predicted) - mean(actual), in the eval's scoring space.

    The companion to MAE and the more diagnostic of the two here: a method can rank conditions
    well while sitting systematically low (the denial pattern — self-reports of a harmful
    behavior cluster near 0 against a double-digit measured rate) or systematically high. MAE
    conflates that offset with per-condition error; this isolates it."""
    ps = metrics.scored_pairs(spec, targets, preds, keys)
    if not ps:
        return None
    return mean(p for _, p in ps) - mean(a for a, _ in ps)


#: Methods with cached predictions that the reports no longer present. cot_flip was retired
#: as a standalone method 2026-08-09: asking for reasoning is a setting, and its
#: fold-learned sign is a tuning mechanism -- both now live in the learned-sign appendix
#: sweep (scripts/learned_flip_sweep.py), which covers every method.
REPORT_EXCLUDE = {"cot_flip"}


def compute(eval_names: list[str], model_names: list[str] | None, split: str, *,
            filter_weak: bool = True, min_ceiling: float = ceilings.MIN_RELIABILITY ** 0.5,
            ceiling_reps: int = ceilings.CEILING_REPS) -> dict[str, Any]:
    """Score each method's tuned best setting on ``split`` for every available (eval, model).

    Correlation (``cells``) is the primary metric. Each (eval, model) cell also carries a parametric
    **noise ceiling** (:mod:`behavior_prediction.ceilings`); ``cells_norm`` holds the secondary
    ceiling-normalized score ``r / ceiling`` (fraction of reachable signal captured). When
    ``filter_weak`` is set (default), cells whose ceiling POINT estimate is below ``min_ceiling``
    (sqrt(0.5): reliability under 0.5 — less than half the target variance is signal) are dropped from *both* metrics and recorded in
    ``dropped_weak`` (the ground truth there carries no reliably-measurable signal, so no predictor
    can be validated); the decision uses only the targets, never a method's score.

    Returns ``{"cells", "cells_norm", "ceilings", "methods", "evals", "models", "settings",
    "dropped", "dropped_weak", "missing_tuning", "filter_weak", "min_ceiling"}`` where
    ``cells[(eval, model, method)] = correlation``, ``cells_norm`` is keyed the same, and
    ``ceilings[(eval, model)]`` is a serialized :class:`~behavior_prediction.ceilings.Ceiling`."""
    cells: dict[tuple[str, str, str], float] = {}
    cells_norm: dict[tuple[str, str, str], float] = {}
    ceiling_map: dict[tuple[str, str], dict[str, Any] | None] = {}
    cells_mae: dict[tuple[str, str, str], float] = {}
    cells_bias: dict[tuple[str, str, str], float] = {}
    settings: dict[tuple[str, str], str] = {}
    dropped, missing_tuning = 0, []
    dropped_weak: list[dict[str, Any]] = []
    for ev in eval_names:
        tpath = tuning_path(ev)
        if not tpath.exists():
            missing_tuning.append(ev)
            continue
        tdoc = common.load_json(tpath)
        best = {base: m["best_setting"] for base, m in tdoc.get("methods", {}).items()
                if m.get("best_setting") and base not in REPORT_EXCLUDE}
        if not best:
            continue
        spec = get_spec(ev)
        manifest = splits.load_manifest(ev)
        for ms in (model_names or results_io.discover_models(ev)):
            loaded = results_io.load_cell(ev, ms)
            if not loaded:
                continue
            targets, methods = loaded
            try:
                keys = splits.split_keys(spec, targets, manifest, split)
            except KeyError:
                print(f"[skip {ev}/{ms}] targets incompatible with current split_unit; "
                      f"re-run bp-targets")
                continue
            ceil = ceilings.estimate(spec, targets, keys, reps=ceiling_reps)
            ceiling_map[(ev, ms)] = ceilings.as_record(ceil)
            if filter_weak and not ceilings.is_reliable(ceil, min_ceiling):
                dropped_weak.append({
                    "eval": ev, "model": ms,
                    "ceiling": ceil.ceiling if ceil else None,
                    "ceiling_lo": ceil.ceiling_lo if ceil else None,
                    "n_conditions": ceil.n_conditions if ceil else 0,
                })
                continue
            for base, setting_id in best.items():
                preds = methods.get(setting_id)
                if preds is None:
                    continue
                r = selection._corr(spec, targets, preds, keys)
                settings[(ev, base)] = setting_id
                if r is None:
                    dropped += 1
                    continue
                cells[(ev, ms, base)] = r
                norm = ceilings.normalize(r, ceil)
                if norm is not None:
                    cells_norm[(ev, ms, base)] = norm
                # Calibration (secondary): only for methods whose predicted_rate is on the eval's
                # own scale — ranking scores (pairwise, cot_flip, the z-scored ensembles, the
                # anchor-calibrated in-context regressors) have no meaningful absolute error.
                adapter = method_registry.REGISTRY.get(base)
                if adapter is not None and getattr(adapter, "calibrated", True):
                    m = metrics.mae_spec(spec, targets, preds, keys)
                    if m is not None:
                        cells_mae[(ev, ms, base)] = m
                    b = _level_bias(spec, targets, preds, keys)
                    if b is not None:
                        cells_bias[(ev, ms, base)] = b
    return {
        "cells": cells,
        "cells_norm": cells_norm,
        "cells_mae": cells_mae,
        "cells_bias": cells_bias,
        "ceilings": ceiling_map,
        "methods": sorted({m for _, _, m in cells}),
        "evals": [e for e in eval_names if any(c[0] == e for c in cells)],
        "models": sorted({m for _, m, _ in cells}),
        "settings": settings,
        "dropped": dropped,
        "dropped_weak": dropped_weak,
        "missing_tuning": missing_tuning,
        "filter_weak": filter_weak,
        "min_ceiling": min_ceiling,
    }


def _col_means(cells, methods, cols, col_of) -> dict[str, dict[str, float | None]]:
    """method -> {col: mean r over matching cells}."""
    out: dict[str, dict[str, float | None]] = {}
    for m in methods:
        out[m] = {}
        for col in cols:
            vals = [r for (e, s, mm), r in cells.items()
                    if mm == m and col_of(e, s) == col]
            out[m][col] = mean(vals) if vals else None
    return out


def _best(cells, ev, ms, methods) -> tuple[str, float] | None:
    cands = [(m, cells[(ev, ms, m)]) for m in methods if (ev, ms, m) in cells]
    return max(cands, key=lambda mc: mc[1]) if cands else None


def render_markdown(c: dict[str, Any], split: str) -> str:
    cells, methods, evals, models = c["cells"], c["methods"], c["evals"], c["models"]
    cells_norm, settings = c.get("cells_norm", {}), c["settings"]
    out: list[str] = ["# Behavior-prediction evaluation", ""]
    if c["missing_tuning"]:
        out += [f"> No `tuning.json` for: {', '.join(c['missing_tuning'])} — run `bp-tune` first.", ""]
    out.append(f"Each method is scored at its **tuned best setting** (frozen in `tuning.json`), on "
               f"the **{split}** split. Correlation primary; constant predictions = 0; macro over "
               f"model×eval cells. Evals: {', '.join(evals) or '—'}; models: {len(models)}; "
               f"methods: {len(methods)}. Cells dropped (no data): {c['dropped']}.")
    if c.get("filter_weak"):
        out.append(f"Weak-signal cells removed (target reliability < 0.5, i.e. ceiling < "
                   f"{c.get('min_ceiling', 0.71):.2f}): {len(c.get('dropped_weak', []))}. "
                   f"The secondary **r/ceil** column is the ceiling-normalized score (fraction of "
                   f"reachable signal captured), aggregated by **median**.")
    out.append("")

    # Weak cells excluded up front: their ground truth has no reliably-measurable spread, so no
    # method can be validated there. Listed explicitly — never silently dropped.
    if c.get("dropped_weak"):
        out += ["## Dropped — weak signal (excluded from all metrics)", "",
                "| Eval | Model | ceiling | CI lower | conditions |", "|---|---|---|---|---|"]
        for d in sorted(c["dropped_weak"], key=lambda d: (d["eval"], d["model"])):
            cval = f"{d['ceiling']:+.2f}" if d["ceiling"] is not None else "undefined"
            lo = f"{d['ceiling_lo']:+.2f}" if d["ceiling_lo"] is not None else "—"
            out.append(f"| {d['eval']} | {d['model']} | {cval} | {lo} | {d['n_conditions']} |")
        out.append("")

    # Which setting each method used (per eval), so the rows are interpretable.
    if settings:
        out += ["## Tuned settings used", "", "| Method | " + " | ".join(evals) + " |",
                "|" + "---|" * (len(evals) + 1)]
        for m in methods:
            row = " | ".join(settings.get((e, m), "—") for e in evals)
            out.append(f"| {m} | {row} |")
        out.append("")

    out += [f"## Per method — global ({split})", "",
            f"| Method | mean {split} r | 95% CI | median r/ceil | cells |",
            "|---|---|---|---|---|"]
    for m in methods:
        vals = [r for (e, s, mm), r in cells.items() if mm == m]
        if not vals:
            continue
        ci = results_io.bootstrap_cells(vals)
        cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "—"
        norms = [v for (e, s, mm), v in cells_norm.items() if mm == m]
        nm = f"{median(norms):+.2f}" if norms else "—"
        out.append(f"| {m} | {mean(vals):+.3f} | {cis} | {nm} | {len(vals)} |")
    out.append("")

    # Calibration (secondary): are the ACTUAL RATES predicted, not just their ordering? Only
    # rate-emitting methods appear (see MethodAdapter.calibrated). `bias` is signed
    # mean(predicted) - mean(actual): negative = systematic under-prediction (the denial pattern).
    cells_mae, cells_bias = c.get("cells_mae", {}), c.get("cells_bias", {})
    if cells_mae:
        out += [f"## Calibration — do the predicted rates match the measured ones? ({split})", "",
                "| Method | mean abs. error | mean signed bias | cells |", "|---|---|---|---|"]
        for m in methods:
            ms_ = [v for (e, s_, mm), v in cells_mae.items() if mm == m]
            bs = [v for (e, s_, mm), v in cells_bias.items() if mm == m]
            if not ms_:
                continue
            out.append(f"| {m} | {mean(ms_):.3f} | {mean(bs):+.3f} | {len(ms_)} |")
        out.append("")
        out += ["### Calibration per method × eval (mean signed bias)", "",
                "| Method | " + " | ".join(evals) + " |", "|" + "---|" * (len(evals) + 1)]
        for m in methods:
            if not any(mm == m for (_, _, mm) in cells_bias):
                continue
            row = []
            for ev in evals:
                vs = [v for (e, s_, mm), v in cells_bias.items() if mm == m and e == ev]
                row.append(f"{mean(vs):+.3f}" if vs else "—")
            out.append(f"| {m} | " + " | ".join(row) + " |")
        out.append("")

    def matrix(title, cols, col_of):
        means = _col_means(cells, methods, cols, col_of)
        out.append(f"## {title}\n")
        out.append("| Method | " + " | ".join(cols) + " | mean |")
        out.append("|" + "---|" * (len(cols) + 2))
        for m in methods:
            present = [v for v in means[m].values() if v is not None]
            row = " | ".join(f"{means[m][col]:+.2f}" if means[m][col] is not None else "—"
                             for col in cols)
            out.append(f"| {m} | {row} | {mean(present):+.2f} |" if present
                       else f"| {m} | {row} | — |")
        out.append("")

    matrix("Per method × eval (mean over models)", evals, lambda e, s: e)
    matrix("Per method × model (mean over evals)", models, lambda e, s: s)

    # Same grid, ceiling-normalized: the fraction of reachable signal captured per eval. 1.00 = at
    # the noise ceiling. Median over models (a ratio is heavy-tailed near small ceilings).
    if cells_norm:
        out.append("## Per method × eval — ceiling-normalized (median r/ceil over models)\n")
        out.append("| Method | " + " | ".join(evals) + " | median |")
        out.append("|" + "---|" * (len(evals) + 2))
        for m in methods:
            vals = {}
            for ev in evals:
                ns = [v for (e, s, mm), v in cells_norm.items() if mm == m and e == ev]
                vals[ev] = median(ns) if ns else None
            present = [v for v in vals.values() if v is not None]
            row = " | ".join(f"{vals[ev]:+.2f}" if vals[ev] is not None else "—" for ev in evals)
            out.append(f"| {m} | {row} | {median(present):+.2f} |" if present
                       else f"| {m} | {row} | — |")
        out.append("")

    out += [f"## Per model — best method per eval ({split})", "",
            "| Model | " + " | ".join(evals) + " | mean |", "|" + "---|" * (len(evals) + 2)]
    for ms in models:
        present, row = [], []
        for ev in evals:
            b = _best(cells, ev, ms, methods)
            row.append(f"{b[1]:+.2f} ({b[0]})" if b else "—")
            if b:
                present.append(b[1])
        mean_s = f"{mean(present):+.2f}" if present else "—"
        out.append(f"| {ms} | " + " | ".join(row) + f" | {mean_s} |")
    out.append("")
    return "\n".join(out) + "\n"


def build(eval_names: list[str], model_names: list[str] | None, split: str = "dev") -> str:
    return render_markdown(compute(eval_names, model_names, split), split)


def render_html(c: dict[str, Any], split: str, pool: str = "default") -> str:
    """Cross-eval evaluation overview: the same matrices as the markdown, colour-coded, linking to
    each eval's detailed report (``<eval>.html``)."""
    from behavior_prediction.report_html import esc, corr_style, fmt_corr, _CSS
    cells, methods, evals, models = c["cells"], c["methods"], c["evals"], c["models"]
    p = ["<!doctype html><html><head><meta charset='utf-8'>",
         f"<title>Behavior-prediction evaluation ({esc(pool)})</title>"
         f"<style>{_CSS}</style></head><body>",
         f"<h1>Behavior-prediction evaluation <span class='muted'>({esc(split)} split, "
         f"{esc(pool)} pool)</span></h1>",
         '<div class="muted">Each method scored at its <b>tuned best setting</b> (frozen in '
         f'<code>tuning.json</code>), on the <b>{esc(split)}</b> split. Constant predictions count '
         f'as 0; macro over model×eval cells. Dropped: {c["dropped"]}.</div>']
    if c["missing_tuning"]:
        p.append(f'<div class="muted">No tuning.json for: {esc(", ".join(c["missing_tuning"]))} '
                 "— run <code>bp-tune</code> first.</div>")
    if c.get("filter_weak"):
        p.append(f'<div class="muted">Weak-signal cells removed (target reliability &lt; 0.5, i.e. ceiling &lt; '
                 f'{c.get("min_ceiling", 0.71):.2f}): {len(c.get("dropped_weak", []))}.</div>')
    p.append("<div>Detailed per-eval reports: "
             + " · ".join(f'<a href="{esc(e)}.html">{esc(e)}</a>' for e in evals) + "</div>")
    if c.get("dropped_weak"):
        rows = "".join(
            f'<tr><td class="lbl">{esc(d["eval"])}</td><td class="lbl">{esc(d["model"])}</td>'
            f'<td class="num">{d["ceiling"]:+.2f}</td><td class="num">{d["ceiling_lo"]:+.2f}</td>'
            f'<td class="num">{d["n_conditions"]}</td></tr>'
            for d in sorted(c["dropped_weak"], key=lambda d: (d["eval"], d["model"]))
            if d["ceiling"] is not None)
        p.append("<h2>Dropped — weak signal</h2><table><tr><th>eval</th><th>model</th>"
                 "<th>ceiling</th><th>CI lower</th><th>conditions</th></tr>" + rows + "</table>")

    def hcell(r):
        return (f'<td class="num" style="{corr_style(r)}">{fmt_corr(r)}</td>'
                if r is not None else '<td class="num">—</td>')

    # Per-method global: primary mean r (+CI) next to the secondary ceiling-normalized median,
    # so the "fraction of reachable signal captured" is visible without opening the markdown.
    cells_norm = c.get("cells_norm", {})
    if methods:
        body = []
        for m in methods:
            vals = [r for (e, s, mm), r in cells.items() if mm == m]
            if not vals:
                continue
            ci = results_io.bootstrap_cells(vals)
            cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "—"
            norms = [v for (e, s, mm), v in cells_norm.items() if mm == m]
            nm = median(norms) if norms else None
            body.append(f'<tr><td class="lbl">{esc(m)}</td>{hcell(mean(vals))}'
                        f'<td class="num">{cis}</td>{hcell(nm)}'
                        f'<td class="num">{len(vals)}</td></tr>')
        p.append(f"<h2>Per method — global ({esc(split)})</h2>"
                 '<div class="muted">Primary: mean r over model×eval cells. Secondary: '
                 "<b>median r/ceil</b> — the ceiling-normalized score, i.e. the fraction of the "
                 "noise-ceiling-reachable correlation the method captures.</div>"
                 "<table><tr><th>method</th><th>mean r</th><th>95% CI</th>"
                 "<th>median r/ceil</th><th>cells</th></tr>" + "".join(body) + "</table>")

    def matrix(title, cols, col_of):
        head = "".join(f"<th>{esc(x)}</th>" for x in cols)
        body = []
        def col_mean(m, col):
            rs = [r for (e, s, mm), r in cells.items()
                  if mm == m and col_of(e, s) == col]
            return mean(rs) if rs else None

        for m in methods:
            vals = {col: col_mean(m, col) for col in cols}
            present = [v for v in vals.values() if v is not None]
            tail = hcell(mean(present)) if present else '<td class="num">—</td>'
            body.append(f'<tr><td class="lbl">{esc(m)}</td>'
                        + "".join(hcell(vals[col]) for col in cols) + tail + "</tr>")
        p.append(f"<h2>{esc(title)}</h2><table><tr><th>method</th>{head}<th>mean</th></tr>"
                 + "".join(body) + "</table>")

    matrix("Per method × eval (mean over models)", evals, lambda e, s: e)
    matrix("Per method × model (mean over evals)", models, lambda e, s: s)

    # The same method × eval grid in the ceiling-normalized space: how much of the *reachable*
    # signal each method captures per eval. Aggregated by median (a ratio is heavy-tailed).
    if cells_norm:
        head = "".join(f"<th>{esc(e)}</th>" for e in evals)
        body = []
        for m in methods:
            vals = {}
            for ev in evals:
                ns = [v for (e, s, mm), v in cells_norm.items() if mm == m and e == ev]
                vals[ev] = median(ns) if ns else None
            present = [v for v in vals.values() if v is not None]
            tail = hcell(median(present)) if present else '<td class="num">—</td>'
            body.append(f'<tr><td class="lbl">{esc(m)}</td>'
                        + "".join(hcell(vals[ev]) for ev in evals) + tail + "</tr>")
        p.append("<h2>Per method × eval — ceiling-normalized (median r/ceil over models)</h2>"
                 '<div class="muted">1.00 = at the noise ceiling (no headroom left); values may '
                 "exceed 1.00 where a cell's r overshoots the noisy ceiling estimate.</div>"
                 f"<table><tr><th>method</th>{head}<th>median</th></tr>"
                 + "".join(body) + "</table>")

    # Per-model breakdown: one method × eval matrix per model — the raw cells behind the two
    # averaged matrices above (a dash = that combo has no scored predictions for this model).
    p.append("<h2>Per model — method × eval</h2>")
    for ms in models:
        head = "".join(f"<th>{esc(e)}</th>" for e in evals)
        body = []
        for m in methods:
            vals = {e: cells.get((e, ms, m)) for e in evals}
            present = [v for v in vals.values() if v is not None]
            tail = hcell(mean(present)) if present else '<td class="num">—</td>'
            body.append(f'<tr><td class="lbl">{esc(m)}</td>'
                        + "".join(hcell(vals[e]) for e in evals) + tail + "</tr>")
        p.append(f"<h3>{esc(ms)}</h3><table><tr><th>method</th>{head}<th>mean</th></tr>"
                 + "".join(body) + "</table>")
    p.append("</body></html>")
    return "\n".join(p)


#: The one pool whose artifacts are canonical: it owns ``evaluation.*`` and the paper's macros.
DEFAULT_POOL = "default"


def write_reports(eval_names: list[str], model_names: list[str] | None, split: str,
                  md_out: str, pool: str = DEFAULT_POOL, **compute_kw: Any) -> dict[str, Any]:
    """Write the markdown + the HTML overview + a detailed per-eval HTML report (restricted to each
    method's tuned best setting). ``compute_kw`` forwards the weak-cell filter options to
    :func:`compute`. Returns the computed result.

    Only the ``default`` pool owns the canonical artifacts (``evaluation.html``/``.json``, the
    ``paper/generated/*.tex`` macros, and the per-eval ``<eval>.html`` back-links). Any other pool
    writes ``evaluation_<pool>.*`` and writes no paper file, so scoring a subset of models can never
    silently overwrite the numbers the paper cites."""
    from behavior_prediction import export, report_html
    c = compute(eval_names, model_names, split, **compute_kw)
    canonical = pool == DEFAULT_POOL
    suffix = "" if canonical else f"_{pool}"
    reports = Path(common.results_root()) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    Path(md_out).parent.mkdir(parents=True, exist_ok=True)
    Path(md_out).write_text(render_markdown(c, split))
    (reports / f"evaluation{suffix}.html").write_text(render_html(c, split, pool))

    # Machine-readable artifacts for the build pipeline / the paper (not for humans):
    # a flat JSON of every number, plus LaTeX \newcommand macros the paper can \input.
    doc = export.build_results_doc(c, split)
    export.write_json(doc, reports / f"evaluation{suffix}.json")
    if canonical:
        export.write_latex_macros(doc, Path("paper") / "generated" / "results.tex")
        export.write_method_model_table(doc, Path("paper") / "generated" / "table_method_model.tex")
        export.write_calibration_table(doc, Path("paper") / "generated" / "table_bias.tex")

    # The per-eval reports cover every model on disk (they ignore the pool), so they are written
    # once, by the canonical run. A pool run links back to its own overview instead.
    if canonical:
        for ev in c["evals"]:
            best_ids = {sid for (e, _b), sid in c["settings"].items() if e == ev}
            try:
                report_html.build_eval_report(
                    str(common.results_root()), get_spec(ev), str(reports / f"{ev}.html"),
                    split=split, methods=best_ids, overview_link="evaluation.html")
            except SystemExit as e:
                print(f"[eval-report {ev}] {e}")
    return c


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals", default=",".join(ACTIVE_EVALS),
                   help=f"comma-separated eval names (default: {','.join(ACTIVE_EVALS)})")
    p.add_argument("--pool", default=DEFAULT_POOL,
                   help=f"named model set from models.yaml `pools:` (default: {DEFAULT_POOL}; "
                        f"'{common.DISCOVER_POOL}' = every model with predictions on disk). Only "
                        "the default pool writes evaluation.* and the paper's LaTeX macros; other "
                        "pools write evaluation_<pool>.*")
    p.add_argument("--models", default=None,
                   help="comma-separated model slugs, overriding --pool (implies a non-default "
                        "pool named 'custom' unless --pool is given)")
    p.add_argument("--split", choices=["dev", "test"], default="dev",
                   help="which split to score (default: dev — keep test frozen until the final run)")
    p.add_argument("--allow-test", action="store_true",
                   help="required to score the frozen TEST split (see guard below)")
    p.add_argument("--no-filter-weak", action="store_true",
                   help="keep weak-signal cells (default: drop cells whose noise-ceiling point "
                        "estimate is below --min-ceiling)")
    p.add_argument("--min-ceiling", type=float, default=ceilings.MIN_RELIABILITY ** 0.5,
                   help="a cell is kept only if its noise-ceiling POINT estimate reaches this "
                        "(default sqrt(0.5) ~= 0.707: target reliability at least 0.5 — half "
                        "the variance is signal; grounded, not a free constant)")
    p.add_argument("--ceiling-reps", type=int, default=ceilings.CEILING_REPS,
                   help=f"bootstrap reps for the noise-ceiling estimate (default "
                        f"{ceilings.CEILING_REPS})")
    p.add_argument("--out", default=None,
                   help="markdown output path (default: results/reports/evaluation[_<pool>].md)")
    args = p.parse_args()

    # behavior_raw.json is gitignored scratch and can silently belong to a different sampling
    # run than the committed targets.json; item-level analyses joining the two would then be
    # scrambled. Cheap reconstruction check — prints a loud banner only when something is stale.
    from behavior_prediction import rawsync
    rawsync.warn_if_stale()

    # ---------------------------------------------------------------------------------------------
    # TEST-SPLIT GUARD. The test split is the frozen, held-out generalization set. Scoring it
    # "uses up" the held-out evaluation and biases all later method comparisons. DO NOT run the
    # test split — keep everything on dev — unless the user has EXPLICITLY instructed a final
    # test-split run. When they have, pass --allow-test (or set BP_ALLOW_TEST=1).
    # ---------------------------------------------------------------------------------------------
    import os
    if args.split == "test" and not (args.allow_test or os.environ.get("BP_ALLOW_TEST") == "1"):
        raise SystemExit(
            "Refusing to score the frozen TEST split. This is the held-out generalization set and "
            "must only be run on explicit user instruction. Re-run with --allow-test (or "
            "BP_ALLOW_TEST=1) if that is truly intended; otherwise use --split dev.")

    eval_names = [e.strip() for e in args.evals.split(",") if e.strip() in EVALS]

    # --models is an ad-hoc pool: name it so it gets its own artifacts rather than overwriting the
    # canonical ones. An explicit --pool always wins as the artifact name.
    if args.models:
        pool = args.pool if args.pool != DEFAULT_POOL else "custom"
        models = [common.model_slug(m.strip()) for m in args.models.split(",")]
    else:
        pool = args.pool
        try:
            models = common.model_pool(pool)
        except KeyError as e:
            raise SystemExit(e.args[0]) from e  # args[0], not str(e): KeyError repr adds quotes

    suffix = "" if pool == DEFAULT_POOL else f"_{pool}"
    reports = Path(common.results_root()) / "reports"
    md_out = args.out or str(reports / f"evaluation{suffix}.md")

    c = write_reports(eval_names, models, args.split, md_out, pool,
                      filter_weak=not args.no_filter_weak, min_ceiling=args.min_ceiling,
                      ceiling_reps=args.ceiling_reps)
    print(render_markdown(c, args.split))
    written = [md_out, str(reports / f"evaluation{suffix}.html"),
               str(reports / f"evaluation{suffix}.json")]
    if pool == DEFAULT_POOL:
        written += [str(reports / f"{e}.html") for e in c["evals"]]
        written += ["paper/generated/results.tex", "paper/generated/table_method_model.tex"]
        # Downstream paper artifacts derived from evaluation.json — regenerate in lockstep so
        # they can never go stale against the canonical numbers. Both are
        # deterministic re-reads of the just-written reports; failures warn rather than abort
        # (the core artifacts above are already on disk). significance_tests also reads
        # evaluation_frontier.json and keeps its last output when that pool is stale.
        import subprocess, sys
        for script in ("scripts/spearman_robustness.py", "scripts/significance_tests.py"):
            r = subprocess.run([sys.executable, script])
            if r.returncode != 0:
                print(f"WARNING: {script} failed (exit {r.returncode}); its paper/generated "
                      "output may be stale")
            else:
                written.append(script.replace("scripts/", "paper/generated/")
                               .replace("spearman_robustness.py", "spearman.tex")
                               .replace("significance_tests.py", "stats.tex"))
    print("Wrote " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
