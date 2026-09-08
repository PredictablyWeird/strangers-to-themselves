"""Self-contained HTML rendering for the behavior-prediction reports.

A rendering *library* (no CLI of its own) used by ``bp-evaluate`` and ``bp-tune``. Auto-discovers
every model for an eval under the ``results/<eval>/<slug>/`` layout, reads each model's targets +
prediction methods, and renders the detailed per-eval HTML:

- **Overview**: actual mean rate by scenario per model, and self-prediction MAE/correlation per
  method per model.
- **Per model**: the full per-condition breakdown (actual vs. each method's prediction, error
  colour-coded green→red), per-scenario MAE, a per-axis summary, and collapsible example prompts.

``build_eval_report`` takes an optional ``methods`` filter so the evaluation report shows only the
tuned best settings. ``corr_style``/``fmt_corr`` (lifted from the old overview) colour the
leaderboard matrices. No external dependencies; everything is inlined.
"""

from __future__ import annotations

import html
from pathlib import Path
from statistics import mean
from typing import Any

from behavior_prediction import common
from behavior_prediction import elicitation
from behavior_prediction import metrics
from behavior_prediction import splits

# Preferred column order for methods (others appended alphabetically).
_METHOD_ORDER = ["self_report", "value", "value-aggregate", "behavioral_sampling",
                 "llm_prediction", "train_scenario_mean"]
# Error (fraction) at which the heat scale saturates to full red.
_ERR_SATURATE = 0.6
# Error (fraction) at/under which a prediction is flagged as "works".
_WORKS_THRESHOLD = 0.10


def corr_style(r: float | None) -> str:
    """Background colour for a correlation cell: red (≤0) → yellow (~0.5) → green (1)."""
    if r is None:
        return "background:#eee"
    x = min(max(r, 0.0), 1.0)
    hue = 120 * x  # 0=red -> 120=green
    return f"background:hsl({hue:.0f},72%,82%)"


def fmt_corr(r: float | None) -> str:
    return "—" if r is None else f"{r:+.2f}"


# --- discovery & loading -------------------------------------------------------


def discover_models(results_dir: str, spec) -> list[tuple[Path, Path]]:
    """Return (targets.json, predictions_dir) per model for ``spec`` from the
    ``results/<eval>/<slug>/`` layout."""
    found: list[tuple[Path, Path]] = []
    base = Path(results_dir) / spec.name
    if base.exists():
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            if (sub / "targets.json").exists():
                found.append((sub / "targets.json", sub / "predictions"))
    return found


def load_model(targets_path: Path, predictions_dir: Path,
               keep: set[str] | None = None) -> dict[str, Any]:
    """Load a model's targets + prediction methods. ``keep`` restricts to those method ids (the
    tuned best settings for the evaluation report); ``None`` keeps every prediction file."""
    doc = common.load_json(targets_path)
    methods: list[dict[str, Any]] = []
    if predictions_dir.exists():
        for pf in sorted(predictions_dir.glob("*.json")):
            pdoc = common.load_json(pf)
            label = pdoc.get("method", pf.stem)
            if keep is not None and label not in keep:
                continue
            # Pull one real generated scenario per condition so the report can show an actual prompt
            # for behavioral_sampling instead of a placeholder. Scenarios live inline in the
            # prediction file (the benchmark adapter path) and/or in a gitignored transcripts sidecar
            # (the CLI runner splits them out); read both. Keyed by condition/contrast key AND by the
            # entry's scenario/category, so a per-cell report key can fall back to its category (the
            # comparative DiscrimEval path generates one case set per category). Best-effort, first
            # source wins; empty when neither carries scenarios (e.g. a fresh, lean checkout).
            ex_scen: dict[str, str] = {}

            def _add(key: str | None, scs: list) -> None:
                if scs and key and key not in ex_scen:
                    ex_scen[key] = scs[0]

            for ck, v in pdoc.get("predictions", {}).items():
                scs = v.get("scenarios") or []
                _add(ck, scs)
                _add(v.get("scenario"), scs)
            tpath = predictions_dir.parent / "transcripts" / pf.name
            if tpath.exists():
                tdoc = common.load_json(tpath)
                for ck, tr in (tdoc.get("transcripts") or {}).items():
                    _add(ck, tr.get("scenarios") or [])
            methods.append({
                "label": label,
                "base_method": pdoc.get("base_method", label),
                "hyperparameters": pdoc.get("hyperparameters", {}),
                "preds": {k: v.get("predicted_rate")
                          for k, v in pdoc.get("predictions", {}).items()},
                "example_scenarios": ex_scen,
            })
    return {
        "model": doc["model"],
        # The actual results-dir name (reasoning-aware slug). Re-deriving it from the full model
        # string is lossy for reasoning variants (several shortcuts share one provider string), so
        # carry it through for any code that needs to find this model's directory.
        "slug": targets_path.parent.name,
        "metric": doc["metric"],
        "targets": doc["targets"],
        "methods": methods,
    }


def short_model(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("openrouter/") else name


def ordered_keys(targets: dict[str, Any], spec, keys: set[str] | None = None) -> list[str]:
    sec = spec.report_secondary_axis
    order = spec.report_axis_order.get(sec, []) if sec else []

    def k(item: tuple[str, dict[str, Any]]) -> tuple:
        key, t = item
        scen = t.get("scenario", "")
        si = spec.scenarios.index(scen) if scen in spec.scenarios else 99
        sv = (t.get("condition") or {}).get(sec) if sec else None
        gi = order.index(sv) if sv in order else 99
        return (si, gi, key)
    items = [(kk, t) for kk, t in targets.items() if keys is None or kk in keys]
    return [key for key, _ in sorted(items, key=k)]


def method_specs(models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """label -> {base_method, hyperparameters, example_scenarios}. A label's base/hyperparameters
    are model-independent (first occurrence wins); ``example_scenarios`` (real generated scenarios
    per condition, for the prompt exhibits) are merged across models so one model's transcripts can
    supply the example even if another model's are missing."""
    specs: dict[str, dict[str, Any]] = {}
    for mod in models:
        for m in mod["methods"]:
            entry = specs.setdefault(m["label"], {
                "base_method": m.get("base_method", m["label"]),
                "hyperparameters": m.get("hyperparameters", {}),
                "example_scenarios": {}})
            for ck, sc in (m.get("example_scenarios") or {}).items():
                entry["example_scenarios"].setdefault(ck, sc)
    return specs


def method_labels(models: list[dict[str, Any]]) -> list[str]:
    specs = method_specs(models)

    def key(l: str) -> tuple:
        base = specs[l]["base_method"]
        return (_METHOD_ORDER.index(base) if base in _METHOD_ORDER else 99, base, l)

    return sorted(specs, key=key)


# --- formatting helpers --------------------------------------------------------


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.0f}%"


def err_style(err: float | None) -> str:
    if err is None:
        return "background:#eee"
    x = min(max(err, 0.0) / _ERR_SATURATE, 1.0)
    hue = 120 * (1 - x)  # 120=green -> 0=red
    return f"background:hsl({hue:.0f},72%,82%)"


def mean_or_none(xs: list[float]) -> float | None:
    return mean(xs) if xs else None


def split_keys_for(spec, targets: dict[str, Any], split: str = "dev") -> set[str] | None:
    """Condition keys assigned to ``split`` (dev/test) for this eval, so the report lists and scores
    only that split — matching the leaderboard/overview view and keeping the frozen test split out of
    the dev report. ``None`` when there is no split manifest → use everything."""
    try:
        man = splits.load_manifest(spec.name)
    except SystemExit:
        return None
    return splits.split_keys(spec, targets, man, split)


def split_dev_keys(spec, targets: dict[str, Any]) -> set[str] | None:
    """Dev-split keys (kept for the cross-eval overview, which always reports dev)."""
    return split_keys_for(spec, targets, "dev")


def is_bias(spec) -> bool:
    return getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast"


def _in(keys: set[str] | None, k: str) -> bool:
    return keys is None or k in keys


def actual_by_scenario(targets: dict[str, Any], spec,
                       keys: set[str] | None = None) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for s in spec.scenarios:
        rates = [t["rate"] for k, t in targets.items()
                 if _in(keys, k) and t.get("scenario") == s and t.get("rate") is not None]
        out[s] = mean_or_none(rates)
    return out


def method_mae(targets: dict[str, Any], preds: dict[str, float | None],
               scenario: str | None = None, keys: set[str] | None = None) -> float | None:
    errs = []
    for k, t in targets.items():
        if not _in(keys, k) or (scenario and t.get("scenario") != scenario):
            continue
        a, p = t.get("rate"), preds.get(k)
        if a is not None and p is not None:
            errs.append(abs(p - a))
    return mean_or_none(errs)


# --- HTML building -------------------------------------------------------------

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;color:#222;}
h1{font-size:22px;} h2{font-size:18px;margin-top:34px;border-bottom:2px solid #ddd;padding-bottom:4px;}
h3{font-size:15px;margin-top:22px;color:#444;}
table{border-collapse:collapse;margin:10px 0 18px;font-size:13px;}
th,td{border:1px solid #ccc;padding:4px 8px;text-align:center;}
th{background:#f4f4f4;position:sticky;top:0;}
td.lbl{text-align:left;font-weight:600;white-space:nowrap;}
td.num{font-variant-numeric:tabular-nums;}
.muted{color:#888;font-size:12px;}
.works{font-weight:700;}
.legend{font-size:12px;margin:6px 0;}
.swatch{display:inline-block;width:14px;height:14px;border:1px solid #aaa;vertical-align:middle;margin:0 3px;}
details{margin:6px 0;} summary{cursor:pointer;font-weight:600;}
pre{white-space:pre-wrap;background:#f7f7f7;border:1px solid #ddd;padding:10px;border-radius:4px;max-width:920px;font-size:12px;line-height:1.45;}
.tldr{background:#eef4fb;border:1px solid #b8d2ec;border-radius:6px;padding:12px 16px;margin:12px 0;max-width:920px;font-size:13px;line-height:1.5;}
.tldr b{color:#1a4d80;}
"""


def esc(s: Any) -> str:
    return html.escape(str(s))


def methods_legend_html(models: list[dict[str, Any]]) -> str:
    """Legend decoding each method id into its base method + hyperparameters (generic over the
    per-method knob set, so it adapts as methods add/drop hyperparameters)."""
    specs = method_specs(models)
    rows = []
    for l in method_labels(models):
        hp = specs[l]["hyperparameters"]
        settings = ", ".join(f"{k}={v}" for k, v in hp.items() if k != "runs") or "—"
        rows.append(f'<tr><td class="lbl">{esc(l)}</td><td>{esc(specs[l]["base_method"])}</td>'
                    f'<td>{esc(settings)}</td></tr>')
    return ('<h3>Methods</h3>'
            '<table><tr><th>method id</th><th>base</th>'
            f'<th>settings</th></tr>{"".join(rows)}</table>')


def overview_html(models: list[dict[str, Any]], labels: list[str], spec,
                  split: str = "dev") -> str:
    rows = []
    metric = (models[0]["metric"] if models else spec.default_metric)
    bias = is_bias(spec)
    unit = "bias" if bias else "rate"
    dev = {id(mod): split_keys_for(spec, mod["targets"], split) for mod in models}
    # Only the scenarios that have at least one in-split condition (so a dev report omits test-only
    # categories entirely instead of printing empty columns).
    scen_keys = {id(mod): {mod["targets"][k]["scenario"] for k in (dev[id(mod)] or mod["targets"])}
                 for mod in models}
    shown_scenarios = [s for s in spec.scenarios if any(s in scen_keys[id(m)] for m in models)]

    # Table 1: actual mean behavior by scenario (descriptive — the measured rate, not a prediction).
    rows.append(f"<h3>Actual mean {esc(metric)} rate by scenario ({esc(split)})</h3>")
    head = "".join(f"<th>{esc(s)}</th>" for s in shown_scenarios) + "<th>overall</th>"
    body = []
    for mod in models:
        dk = dev[id(mod)]
        abs_ = actual_by_scenario(mod["targets"], spec, dk)
        overall = mean_or_none([t["rate"] for k, t in mod["targets"].items()
                                if _in(dk, k) and t.get("rate") is not None])
        cells = "".join(f'<td class="num">{pct(abs_[s])}</td>' for s in shown_scenarios)
        body.append(f'<tr><td class="lbl">{esc(short_model(mod["model"]))}</td>{cells}'
                    f'<td class="num"><b>{pct(overall)}</b></td></tr>')
    rows.append(f'<table><tr><th>model</th>{head}</tr>{"".join(body)}</table>')

    # Table 2: self-prediction MAE per method, with the trivial baseline for scale. For a bias eval
    # this is the per-contrast GAP error and the baseline is "predict zero bias"; both are scored on
    # the chosen split so the numbers match the cross-eval overview.
    rows.append(f"<h3>Self-prediction accuracy — {unit} MAE on {esc(split)} (lower = better)</h3>")
    base_cols = ["predict 0 bias"] if bias else list(metrics.BASELINES)
    head = "".join(f"<th>{esc(l)}</th>" for l in labels)
    head += "".join(f'<th class="muted">{esc(b)}</th>' for b in base_cols)
    body = []
    for mod in models:
        dk = dev[id(mod)]
        cells = []
        for l in labels:
            m = next((mm for mm in mod["methods"] if mm["label"] == l), None)
            mae = metrics.mae_spec(spec, mod["targets"], m["preds"], dk) if m else None
            cells.append(f'<td class="num">{"—" if mae is None else f"{mae*100:.1f}"}</td>')
        consts = [0.0] if bias else list(metrics.BASELINES.values())
        for c in consts:
            bmae = metrics.baseline_mae_spec(spec, mod["targets"], c, dk)
            cells.append(f'<td class="num muted">{"—" if bmae is None else f"{bmae*100:.1f}"}</td>')
        body.append(f'<tr><td class="lbl">{esc(short_model(mod["model"]))}</td>{"".join(cells)}</tr>')
    rows.append(f'<table><tr><th>model</th>{head}</tr>{"".join(body)}</table>')
    rows.append(f'<div class="muted">MAE in percentage points on the <b>{esc(split)}</b> split. '
                + ('The greyed column is the trivial <b>predict-zero-bias</b> baseline (assume no '
                   'discrimination); a method that doesn\'t beat it carries no signal about which '
                   'groups are favored.' if bias else
                   'The greyed columns are <b>constant baselines</b> — always 0%/50%/100%; a method '
                   'that doesn\'t beat the best of them carries no condition-level signal.') + '</div>')

    # Table 3: self-prediction correlation per method (reveals constant / non-tracking predictions).
    rows.append(f"<h3>Self-prediction accuracy — {unit} correlation with actual on {esc(split)} "
                "(higher = better)</h3>")
    head = "".join(f"<th>{esc(l)}</th>" for l in labels)
    body = []
    for mod in models:
        dk = dev[id(mod)]
        cells = []
        for l in labels:
            m = next((mm for mm in mod["methods"] if mm["label"] == l), None)
            r = metrics.corr_spec(spec, mod["targets"], m["preds"], dk) if m else None
            cells.append(f'<td class="num">{"—" if r is None else f"{r:+.2f}"}</td>')
        body.append(f'<tr><td class="lbl">{esc(short_model(mod["model"]))}</td>{"".join(cells)}</tr>')
    rows.append(f'<table><tr><th>model</th>{head}</tr>{"".join(body)}</table>')
    rows.append('<div class="muted">Pearson correlation between predicted and actual '
                + (f'<b>{unit}</b> (per-contrast signed gap) ' if bias else f'<b>{unit}</b> ')
                + f'across conditions, on the {esc(split)} split — the same number the cross-eval '
                'overview shows. <b>“—”</b> = undefined (the method\'s predictions have no variance, '
                'i.e. it outputs essentially the same value for every condition).</div>')
    return "\n".join(rows)


def condition_table_html(mod: dict[str, Any], labels: list[str], spec,
                         split_keys: set[str] | None = None) -> str:
    targets = mod["targets"]
    keys = ordered_keys(targets, spec, split_keys)
    present = [l for l in labels if any(m["label"] == l for m in mod["methods"])]
    methmap = {m["label"]: m["preds"] for m in mod["methods"]}

    head = '<th>condition</th><th>actual</th>'
    for l in present:
        head += f'<th>{esc(l)}<br>pred</th><th>{esc(l)}<br>|err|</th>'
    body = []
    for k in keys:
        t = targets[k]
        a = t.get("rate")
        label = spec.condition_label(t["condition"])
        cells = f'<td class="lbl">{esc(label)}</td><td class="num">{pct(a)}</td>'
        for l in present:
            p = methmap[l].get(k)
            err = abs(p - a) if (p is not None and a is not None) else None
            works = ' works' if (err is not None and err <= _WORKS_THRESHOLD) else ''
            errtxt = "—" if err is None else f"{err*100:.0f}"
            mark = " ✓" if works else ""
            cells += (f'<td class="num">{pct(p)}</td>'
                      f'<td class="num{works}" style="{err_style(err)}">{errtxt}{mark}</td>')
        body.append(f"<tr>{cells}</tr>")
    return f'<table><tr>{head}</tr>{"".join(body)}</table>'


def scenario_mae_html(mod: dict[str, Any], labels: list[str], spec,
                      split_keys: set[str] | None = None) -> str:
    present = [l for l in labels if any(m["label"] == l for m in mod["methods"])]
    methmap = {m["label"]: m["preds"] for m in mod["methods"]}
    head = "".join(f"<th>{esc(l)}</th>" for l in present)
    body = []
    for s in [s for s in spec.scenarios
              if any(_in(split_keys, k) and t.get("scenario") == s
                     for k, t in mod["targets"].items())]:
        cells = ""
        for l in present:
            mae = method_mae(mod["targets"], methmap[l], scenario=s, keys=split_keys)
            cells += f'<td class="num">{"—" if mae is None else f"{mae*100:.1f}"}</td>'
        body.append(f'<tr><td class="lbl">{esc(s)}</td>{cells}</tr>')
    return f'<table><tr><th>scenario</th>{head}</tr>{"".join(body)}</table>'


def axis_breakdown_html(mod: dict[str, Any], labels: list[str], spec,
                        split_keys: set[str] | None = None) -> str:
    targets = {k: v for k, v in mod["targets"].items() if _in(split_keys, k)}
    present = [l for l in labels if any(m["label"] == l for m in mod["methods"])]
    methmap = {m["label"]: m["preds"] for m in mod["methods"]}
    cond_of = {k: (targets[k].get("condition") or {}) for k in targets}
    swept = [ax for ax in spec.report_candidate_axes
             if len({cond_of[k].get(ax) for k in targets}) > 1]
    out = []
    for ax in swept:
        out.append(f"<h3>By {esc(ax)} (rollup across conditions)</h3>")
        head = f'<th>{esc(ax)}</th><th>n</th><th>actual</th>'
        for l in present:
            head += (f'<th>{esc(l)}<br>pred</th><th>{esc(l)}<br>MAE</th>'
                     f'<th>{esc(l)}<br>r</th>')
        vals = list({cond_of[k].get(ax) for k in targets})
        ordr = spec.report_axis_order.get(ax)
        vals.sort(key=lambda v: (ordr.index(v) if ordr and v in ordr else 99, str(v)))
        body = []
        for val in vals:
            grp = [k for k in targets if cond_of[k].get(ax) == val]
            a_mean = mean_or_none([targets[k]["rate"] for k in grp if targets[k]["rate"] is not None])
            cells = f'<td class="lbl">{esc(val)}</td><td class="num">{len(grp)}</td><td class="num">{pct(a_mean)}</td>'
            for l in present:
                pv = [methmap[l].get(k) for k in grp if methmap[l].get(k) is not None]
                errs = [abs(methmap[l][k] - targets[k]["rate"]) for k in grp
                        if methmap[l].get(k) is not None and targets[k]["rate"] is not None]
                mae = mean_or_none(errs)
                # Within-group Pearson r (predicted vs actual across this group's conditions); "—"
                # when undefined (e.g. a group with <2 scored conditions or a flat prediction).
                ps = [(targets[k]["rate"], methmap[l][k]) for k in grp
                      if methmap[l].get(k) is not None and targets[k].get("rate") is not None]
                r = metrics.pearson(ps)
                cells += (f'<td class="num">{pct(mean_or_none(pv))}</td>'
                          f'<td class="num" style="{err_style(mae)}">'
                          f'{"—" if mae is None else f"{mae*100:.0f}"}</td>'
                          f'<td class="num" style="{corr_style(r)}">{fmt_corr(r)}</td>')
            body.append(f"<tr>{cells}</tr>")
        out.append(f'<table><tr>{head}</tr>{"".join(body)}</table>')
    return "\n".join(out)


def _prompt_item_html(item_label: str, item_key: str, slots: dict[str, str]) -> str:
    """One expandable item showing a method's labeled prompt(s) for one condition/contrast."""
    multi = len(slots) > 1
    blocks = []
    for slabel, text in slots.items():
        if multi:
            blocks.append(f'<div class="muted" style="margin:8px 0 2px">{esc(slabel)}</div>')
        blocks.append(f"<pre>{esc(text)}</pre>")
    return (f'<details style="margin-left:18px"><summary>{esc(item_label)} '
            f'<span class="muted">({esc(item_key)})</span></summary>{"".join(blocks)}</details>')


def _prompt_group_html(title: str, items: list[tuple[str, str, dict[str, str]]]) -> str:
    inner = "".join(_prompt_item_html(lbl, key, slots) for lbl, key, slots in items)
    return f'<details><summary>{esc(title)} — {len(items)} entries</summary>{inner}</details>'


def _prompt_groups_html(title: str,
                        groups: list[tuple[str, list[tuple[str, str, dict[str, str]]]]]) -> str:
    """Like ``_prompt_group_html`` but with one nested ``<details>`` per group label (e.g. MASK's
    domains), mirroring how DiscrimEval's list clusters by category."""
    total = sum(len(items) for _, items in groups)
    blocks = []
    for glabel, items in groups:
        inner = "".join(_prompt_item_html(lbl, key, slots) for lbl, key, slots in items)
        blocks.append(f'<details style="margin-left:14px"><summary>{esc(glabel)} — '
                      f'{len(items)} entries</summary>{inner}</details>')
    return (f'<details><summary>{esc(title)} — {total} entries in {len(groups)} groups</summary>'
            f'{"".join(blocks)}</details>')


def _contrast_label(ckey: str) -> str:
    """Readable label for a bias contrast key ``cat/axis/value-vs-base`` -> ``cat · axis value-vs-base``."""
    parts = ckey.split("/")
    return f"{parts[0]} · {' '.join(parts[1:])}" if len(parts) >= 3 else ckey


def prompts_html(models: list[dict[str, Any]], spec, split: str = "dev") -> str:
    """Collapsible 'Example prompts' section: the behavioral prompt the model saw, plus the exact
    elicitation prompt(s) every prediction method asks — including multi-prompt methods (e.g.
    behavioral_sampling's generation + grading) and eval-specific ones (e.g. DiscrimEval's
    direct-contrast prompt). Restricted to the reported ``split`` so test prompts stay hidden."""
    merged: dict[str, Any] = {}
    for mod in models:
        for k, t in mod["targets"].items():
            merged.setdefault(k, t)
    keys = ordered_keys(merged, spec, split_keys_for(spec, merged, split))
    specs = method_specs(models)

    out = ["<h2>Example prompts</h2>",
           '<div class="muted">Hidden by default; click to expand. '
           '<b>behavioral</b> = the full prompt the model actually saw in the eval (what produces '
           'the measured rates). The rest are the <b>elicitation</b> prompts each prediction method '
           'asks; a method with several fixed prompts shows each labeled step. The condition-'
           'specific framing is injected per condition — expand and compare to see what changes.</div>']

    # 1. Behavioral (eval) prompts the model actually saw. If the spec supplies a group label
    # (e.g. MASK's domain), cluster them into nested sections (a few examples per group) like
    # DiscrimEval's category list; otherwise show the flat one-entry-per-condition list.
    beh_items = [(spec.condition_label(merged[k]["condition"]), k,
                  spec.behavioral_prompt(merged[k]["condition"])) for k in keys]
    beh_groups = [spec.behavioral_group(merged[k]["condition"]) for k in keys]
    if any(g is not None for g in beh_groups):
        per_group_cap = 6  # a few representative examples per group keeps the section readable
        grouped: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
        totals: dict[str, int] = {}
        for item, g in zip(beh_items, beh_groups):
            g = g or "unlabelled"
            totals[g] = totals.get(g, 0) + 1
            grouped.setdefault(g, [])
            if len(grouped[g]) < per_group_cap:
                grouped[g].append(item)
        group_list = [
            (g if totals[g] <= per_group_cap else f"{g} (showing {per_group_cap} of {totals[g]})",
             grouped[g]) for g in sorted(grouped)]
        out.append(_prompt_groups_html("behavioral (eval)", group_list))
    else:
        out.append(_prompt_group_html("behavioral (eval)", beh_items))

    # 2. One group per prediction method that exposes a prompt (skips e.g. derived-only methods).
    # On a bias_contrast eval the introspective methods predict per *contrast* (a comparison), not
    # per cell — so list one entry per contrast (the actual comparative prompt this setting sends),
    # NOT one per individual-group cell.
    bias = is_bias(spec)
    scenarios = sorted({merged[k]["condition"]["scenario"] for k in keys}) if bias else None
    # A handful of labelled training pairs from the reported split, to fill the trained methods'
    # (llm_prediction, few_shot) training block in their exhibits (see example_prompt_exhibit):
    # (condition, rate) cases normally; (contrast key, label, signed gap) contrasts on a bias eval,
    # where both work at the contrast grain like every other method. The bias entries carry the
    # contrast KEY as well as its label because few_shot re-renders each contrast's actual prompt
    # (it shows them as conversation turns), which the label alone can't identify.
    if bias:
        sub = {k: merged[k] for k in keys}
        gaps = spec.score_contrasts(sub, {k: t.get("rate") for k, t in sub.items()})
        clabel = {c["key"]: c["label"]
                  for c in (spec.condition_contrast(merged[k]["condition"]) for k in keys) if c}
        llm_train = [(ck, clabel[ck], a) for ck, (a, _) in gaps.items() if ck in clabel][:8]
    else:
        llm_train = [(merged[k]["condition"], merged[k]["rate"]) for k in keys
                     if merged[k].get("rate") is not None][:8]
    for l in method_labels(models):
        base, hp = specs[l]["base_method"], specs[l]["hyperparameters"]
        items: list[tuple[str, str, dict[str, str]]] = []
        if bias and base in elicitation.PROMPT_BUILDERS:
            cprompts = spec.comparative_bias_prompts(scenarios, base_method=base, params=hp) or {}
            items = [(_contrast_label(ckey), ckey,
                      {"order 1": o["o0"],
                       "order 2 (options swapped; parsed delta negated)": o["o1"]})
                     for ckey, o in cprompts.items()]
        else:
            ex_scen = specs[l].get("example_scenarios", {})
            for k in keys:
                cond = merged[k]["condition"]
                # Every method that reaches here on a bias eval predicts per CONTRAST (the oracles,
                # llm_prediction, behavioral_sampling's paired variant), so the baseline cell — which
                # anchors the contrasts but is not itself predicted, and has no exhibit — is skipped
                # rather than dropping the whole group.
                contrast = spec.condition_contrast(cond) if bias else None
                if bias and contrast is None:
                    continue
                # behavioral_sampling transcripts are keyed by condition (PB) or by category (the
                # comparative DiscrimEval path); fall back to the condition's scenario so both resolve.
                sc = ex_scen.get(k) or ex_scen.get(cond.get("scenario"))
                # For the trained in-context methods, show a real training block (sampled pairs
                # minus this target).
                tex = None
                if base in ("llm_prediction", "few_shot", "few_shot_other"):
                    tex = ([(ck, lbl, g) for ck, lbl, g in llm_train if ck != contrast["key"]][:6]
                           if bias else [(c, r) for c, r in llm_train if c is not cond][:6])
                # few_shot_other's system turn names the model under test; this section is shared
                # across the report's models, so show it for the first (the only one, in a
                # single-model report) — the exhibit's label states which.
                slots = elicitation.example_prompt_exhibit(base, spec, cond, hp,
                                                           example_scenario=sc, train_examples=tex,
                                                           model=models[0]["model"])
                if slots is None:       # derived method (cross_model_mean): no prompt of its own
                    items = []
                    break
                # List the entry under what the method actually predicts: on a bias eval the
                # contrast (the gap it forecasts), not the individual-group cell.
                items.append((_contrast_label(contrast["key"]), contrast["key"], slots)
                             if contrast is not None else (spec.condition_label(cond), k, slots))
        if items:
            out.append(_prompt_group_html(l, items))

    # 3. Eval-specific exhibits (split-restricted), for methods that aren't standard prompt builders.
    for ex in spec.extra_prompt_exhibits(split):
        out.append(_prompt_group_html(ex["title"], ex["items"]))

    return "\n".join(out)


def build_eval_report(results_dir: str, spec, out_path: str, *, split: str = "dev",
                      methods: set[str] | None = None, overview_link: str | None = None) -> str:
    """Detailed per-eval HTML (the report the old ``bp-report`` produced), written to ``out_path``.

    ``methods`` restricts to a set of method ids — ``bp-evaluate`` passes the tuned best settings so
    the report shows exactly what was evaluated; ``None`` shows every prediction on disk.
    ``overview_link`` adds a back-link to the cross-eval evaluation page."""
    found = discover_models(results_dir, spec)
    if not found:
        raise SystemExit(f"No models found for eval {spec.name!r} under {results_dir!r} "
                         "(need targets.json).")
    models = [load_model(tf, pd, keep=methods) for tf, pd in found]
    models = [m for m in models if m["methods"]]
    if not models:
        raise SystemExit(f"No predictions for the requested methods under {spec.name!r}.")
    labels = method_labels(models)
    metric_set = {m["metric"] for m in models}
    metric = metric_set.pop() if len(metric_set) == 1 else "/".join(sorted(metric_set))
    keys_of = {id(mod): split_keys_for(spec, mod["targets"], split) for mod in models}

    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>{esc(spec.report_title)}</title><style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>{esc(spec.report_title)} <span class='muted'>({esc(split)} split)</span></h1>")
    if overview_link:
        parts.append(f'<div class="muted"><a href="{esc(overview_link)}">← evaluation overview</a></div>')
    elic_methods = [l for l in labels if l in elicitation.PROMPT_BUILDERS]
    parts.append(spec.report_tldr_html(metric, elic_methods))
    parts.append(f'<div class="muted">Metric: <b>{esc(metric)}</b> · {len(models)} models · '
                 f'split: <b>{esc(split)}</b> (only {esc(split)} conditions are listed/scored; '
                 f'the other split is hidden) · tuned settings: {esc(", ".join(labels))}</div>')
    parts.append('<div class="legend">|err| heat: '
                 '<span class="swatch" style="background:hsl(120,72%,82%)"></span> 0 pts'
                 '<span class="swatch" style="background:hsl(60,72%,82%)"></span> ~30 pts'
                 '<span class="swatch" style="background:hsl(0,72%,82%)"></span> ≥60 pts'
                 f' · ✓ = prediction within {int(_WORKS_THRESHOLD*100)} pts of actual.</div>')

    parts.append("<h2>Overview</h2>")
    parts.append(methods_legend_html(models))
    parts.append(overview_html(models, labels, spec, split))

    bias = is_bias(spec)
    for mod in models:
        sk = keys_of[id(mod)]
        parts.append(f"<h2>{esc(short_model(mod['model']))}</h2>")
        parts.append(spec.report_extra_html(mod["slug"], results_dir, split))
        if bias:
            # The bias eval is scored ONLY on contrasts (per-(category, axis, value) gaps), which
            # the report_extra_html table above shows. The per-condition *rate* tables below would
            # report accuracy on individual cell rates (e.g. the age=80 rate), which we don't score.
            continue
        if getattr(spec, "report_per_condition", True):
            parts.append("<h3>Per-condition (all sweep values)</h3>")
            parts.append(condition_table_html(mod, labels, spec, sk))
        parts.append("<h3>MAE by scenario</h3>")
        parts.append(scenario_mae_html(mod, labels, spec, sk))
        parts.append(axis_breakdown_html(mod, labels, spec, sk))

    parts.append(prompts_html(models, spec, split))
    parts.append("</body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    return str(out)


# --- tuning report (settings × prompts × performance) --------------------------

def _setting_prompts_html(spec, base_method: str, hp: dict[str, Any],
                          examples: list[tuple[str, dict[str, Any]]]) -> str:
    """Collapsible rendered prompt(s) a setting sends, for one or two example conditions — so the
    effect of each argument value is visible side by side across settings."""
    blocks = []
    for label, cond in examples:
        slots = elicitation.example_prompt_exhibit(base_method, spec, cond, hp)
        if slots is None:
            return '<div class="muted">(no prompt of its own)</div>'
        inner = "".join((f'<div class="muted" style="margin:8px 0 2px">{esc(sl)}</div>'
                         if len(slots) > 1 else "") + f"<pre>{esc(text)}</pre>"
                        for sl, text in slots.items())
        blocks.append(f'<details style="margin-left:18px"><summary>example: {esc(label)}</summary>'
                      f'{inner}</details>')
    return "".join(blocks)


def build_tuning_report(spec, tuning_doc: dict[str, Any],
                        examples: list[tuple[str, dict[str, Any]]], out_path: str) -> str:
    """HTML tuning report: per method, the settings it was tuned over — each setting's arguments,
    its aggregate + per-model dev correlation, and the actual prompt it sends — with the chosen
    best setting highlighted. Written to ``out_path``."""
    pool = tuning_doc.get("selection_pool", [])
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             f"<title>Tuning — {esc(tuning_doc.get('eval'))}</title><style>{_CSS}</style>"
             "</head><body>"]
    parts.append(f"<h1>Hyperparameter tuning — {esc(tuning_doc.get('eval'))} "
                 f"<span class='muted'>({esc(tuning_doc.get('split', 'dev'))} split)</span></h1>")
    parts.append('<div class="muted">Each method is tuned over its settings (from '
                 '<code>methods.yaml</code>); the <b>best</b> setting (highest aggregate dev '
                 'correlation over the selection pool) is highlighted and frozen in '
                 f'<code>tuning.json</code>. Selection pool: {esc(", ".join(pool)) or "—"}.</div>')

    for base, m in tuning_doc.get("methods", {}).items():
        best = m.get("best_setting")
        parts.append(f"<h2>{esc(base)} <span class='muted'>(criterion: {esc(m.get('criterion'))}; "
                     f"best: {esc(best or '—')})</span></h2>")
        head = ("<tr><th>setting</th><th>arguments</th><th>agg r</th><th># models</th>"
                + "".join(f"<th>{esc(short_model(p))}</th>" for p in pool) + "</tr>")
        rows = []
        for mid, s in m.get("settings", {}).items():
            args = ", ".join(f"{k}={v}" for k, v in (s.get("args") or {}).items()) or "(default)"
            agg = s.get("agg_r")
            star = " ★" if mid == best else ""
            per = "".join(
                f'<td class="num" style="{corr_style((s.get("per_model", {}).get(p) or {}).get("r"))}">'
                f'{fmt_corr((s.get("per_model", {}).get(p) or {}).get("r"))}</td>' for p in pool)
            rows.append(
                f'<tr><td class="lbl">{esc(mid)}{star}</td><td>{esc(args)}</td>'
                f'<td class="num" style="{corr_style(agg)}">{fmt_corr(agg)}</td>'
                f'<td class="num">{s.get("n_models", 0)}</td>{per}</tr>')
            rows.append(f'<tr><td colspan="{4 + len(pool)}">'
                        f'{_setting_prompts_html(spec, base, s.get("args") or {}, examples)}</td></tr>')
        parts.append(f"<table>{head}{''.join(rows)}</table>")

    parts.append("</body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    return str(out)
