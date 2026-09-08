#!/usr/bin/env python3
"""First-pass figures for the paper, generated from the machine-readable evaluation export.

Reads ``results/reports/evaluation.json`` (written by ``bp-evaluate`` via ``export.py``) and emits
two bar charts into ``paper/generated/``, matching the RQ2/RQ1 placeholders in ``paper/main.tex``
(``fig:results``):

  * ``fig_rq2_methods``  -- mean correlation per method (marginalized over models & evals), with
    bootstrap-CI error bars and a dashed zero line (no-ranking baseline). The outside-view prior
    (``cross_model_mean``) is drawn as a dashed horizontal line -- the bar to clear -- rather
    than a bar, and the ``oracle_xmm*`` ensembles are omitted.
  * ``fig_rq1_evals``    -- best-method correlation per eval, selected among methods that do not
    use the cross-model prior; that prior's per-eval value is overlaid as a dashed tick.

These are deliberately plain: the point is a real, regenerable figure in the pipeline, not polish.
Run as ``python -m behavior_prediction.plots`` (after ``bp-evaluate``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import random

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402

from behavior_prediction import method_names  # noqa: E402


def _cell_ci(doc: dict[str, Any], ev: str, meth: str, boot: int = 2000) -> tuple[float, float] | None:
    """Percentile-bootstrap 95% CI of the mean r over the (model) cells of one (eval, method)."""
    vals = [c["r"] for c in doc.get("cells", [])
            if c["eval"] == ev and c["method"] == meth and c.get("r") is not None]
    if len(vals) < 2:
        return None
    rng = random.Random(1234)
    boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(boot))
    return boots[int(0.025 * boot)], boots[int(0.975 * boot) - 1]


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        paths.append(p)
    plt.close(fig)
    return paths


# Methods built on the cross-model prior: excluded from the per-method bars (the ensembles) and
# from best-method selection, since the paper draws the prior as the bar the others must clear.
XMM_METHODS = {"cross_model_mean", "oracle_xmm", "oracle_xmm_learned"}


def fig_rq2_methods(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    """RQ2: mean correlation per method, with the outside-view prior as a dashed bar-to-clear."""
    rows = [r for r in doc["per_method"] if r["method"] not in XMM_METHODS]
    xmm = next((r for r in doc["per_method"] if r["method"] == "cross_model_mean"), None)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if rows:
        labels = [method_names.short(r["method"]) for r in rows]
        means = [r["mean_r"] for r in rows]
        # Asymmetric error bars from the CI; fall back to 0 where the CI is undefined.
        lo = [(r["mean_r"] - r["ci_lo"]) if r["ci_lo"] is not None else 0.0 for r in rows]
        hi = [(r["ci_hi"] - r["mean_r"]) if r["ci_hi"] is not None else 0.0 for r in rows]
        x = range(len(labels))
        ax.bar(x, means, yerr=[lo, hi], capsize=4, color="#4c72b0")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    if xmm is not None:
        if xmm["ci_lo"] is not None and xmm["ci_hi"] is not None:
            ax.axhspan(xmm["ci_lo"], xmm["ci_hi"], color="#c44e52", alpha=0.12, lw=0)
        ax.axhline(xmm["mean_r"], ls="--", lw=1.4, color="#c44e52",
                   label="cross-model behavior mean (outside view)")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_ylabel("mean correlation")
    ax.set_title(f"RQ2: prediction signal per method ({doc['split']})")
    return _save(fig, out_dir, "fig_rq2_methods")


def fig_rq1_evals(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    """RQ1: best-method correlation per eval (which behaviors are predictable).

    Best method is selected among methods that do not use the cross-model prior; the prior's own
    per-eval correlation is overlaid as a dashed tick (the outside-view bar).
    """
    per_me = doc.get("per_method_eval", [])
    evals = [r["eval"] for r in doc["per_eval_best"]]  # keep the export's eval ordering
    best: dict[str, tuple[str, float]] = {}
    xmm_by_eval: dict[str, float] = {}
    for r in per_me:
        if r.get("mean_r") is None:
            continue
        if r["method"] == "cross_model_mean":
            xmm_by_eval[r["eval"]] = r["mean_r"]
        if r["method"] in XMM_METHODS:
            continue
        if r["eval"] not in best or r["mean_r"] > best[r["eval"]][1]:
            best[r["eval"]] = (r["method"], r["mean_r"])
    rows = [(e, *best[e]) for e in evals if e in best]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if rows:
        labels = [f"{SHORT.get(e, e)}\n({method_names.short(m)})" for e, m, _ in rows]
        vals = [v for _, _, v in rows]
        cis = [_cell_ci(doc, e, m) for e, m, _ in rows]
        lo = [(v - ci[0]) if ci else 0.0 for (_, _, v), ci in zip(rows, cis)]
        hi = [(ci[1] - v) if ci else 0.0 for (_, _, v), ci in zip(rows, cis)]
        x = range(len(labels))
        ax.bar(x, vals, yerr=[lo, hi], capsize=4, color="#55a868")
        for i, (e, _, _) in enumerate(rows):
            if e in xmm_by_eval:
                ax.hlines(xmm_by_eval[e], i - 0.42, i + 0.42, ls="--", lw=1.4, color="#c44e52",
                          label="cross-model behavior mean (outside view)" if i == 0 else None)
        ax.legend(loc="upper left", fontsize=7, frameon=False)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_ylabel("best-method correlation")
    ax.set_title(f"RQ1: predictability per eval ({doc['split']})")
    return _save(fig, out_dir, "fig_rq1_evals")



# --- The three figures that carry the paper's main points -------------------------------------
# Each illustrates exactly one claim, as plainly as possible:
#   fig_headroom   how much of each evaluation is model-specific at all (RQ1)
#   fig_axes       information transforms elicitation; the subject of the question does not (RQ2)
#   fig_selfspec   the like-for-like self-advantage that survives, per evaluation

#: Per-eval knowable self-specific share and cross-model behavior agreement. Sourced from
#: scripts/generic_prior_analysis.py + predictability_decomposition.py (see their reports);
#: kept here as data so the figure regenerates without re-running those sweeps.
HEADROOM = {   # 2026-08-19: frozen TEST split, 12-model pool (predictability_decomposition
    # --split test macro rows: shared = r_xmm^2, self = self%% share of reliable variance)
    "reward_hacking":      (0.79, 0.10),
    "capability_mmlu":     (0.69, 0.04),
    "sycophancy_pushback": (0.45, 0.35),
    "mask_subdomain_pressure": (0.53, 0.38),
    "propensitybench":     (0.20, 0.62),
    "tau2_transfer":       (0.16, 0.69),
    "discrimeval":         (0.09, 0.76),
    "tau2_policy":         (0.01, 0.97),
}
SHORT = {"reward_hacking": "reward hack", "capability_mmlu": "capability",
         "sycophancy_pushback": "sycophancy", "discrimeval": "DiscrimEval",
         "propensitybench": "PropensityB", "tau2_policy": r"$\tau^2$",
         "tau2_transfer": r"$\tau^2$ transfer", "mask_subdomain_pressure": "MASK",
         "discrimeval_implicit": "DiscrimEval-impl",
         "agentic_misalignment": "agentic misal."}

#: Like-for-like identity transfer at the oracle tier (scripts/oracle_identity_transfer.py
#: --split test, 2026-08-19; frozen TEST split).
SELFSPEC = {                      # eval -> (informed diag, informed off, generic diag, generic off)
    "propensitybench":     (0.06, 0.18, 0.43, 0.31),
    "capability_mmlu":     (0.43, 0.40, 0.40, 0.43),
    "sycophancy_pushback": (0.37, 0.34, 0.43, 0.38),
    "discrimeval":         (0.20, 0.15, 0.24, 0.16),
    "reward_hacking":      (0.29, 0.28, 0.38, 0.42),
    "tau2_policy":         (0.18, 0.05, -0.11, -0.01),
}


def fig_headroom(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    """RQ1: how much of each evaluation is model-specific, i.e. could reward introspection."""
    evs = sorted(HEADROOM, key=lambda e: HEADROOM[e][1])
    share = [HEADROOM[e][1] for e in evs]
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    y = range(len(evs))
    ax.barh(list(y), share, color="#4c72b0", label="model-specific (knowable)")
    ax.barh(list(y), [1 - v for v in share], left=share, color="#d9d9d9",
            label="shared across models")
    for i, e in enumerate(evs):
        ax.text(share[i] + 0.015, i, f"{share[i]:.2f}", va="center", fontsize=7.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels([SHORT[e] for e in evs], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of reliable variance")
    ax.set_title("How much is there to know? (approximate)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    return _save(fig, out_dir, "fig_headroom")


def fig_axes(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    """fig:axes (Figure 1): information moves elicitation a long way; whether the question is
    about the model itself moves it very little. Paired bars per information tier: blue = the
    question is about the model itself, orange = the same question about AI assistants in general,
    an outside analyst (told the model's name), or other models. No title (the caption carries it); paper-size fonts.

    Layout: the information-free "abstract description" tier sits in the MIDDLE; each side adds a
    different KIND of information (left: the model's own measured history; right: the exact
    evaluation items). Hollow block arrows in the gaps point outward from the centre — "adding
    information" in two directions rather than one ordered x-axis, since the two kinds are not
    comparable amounts."""
    per = {r["method"]: r for r in doc["per_method"]}
    SELF, OTHER, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a1a", "#6b6b6b"
    groups = [
        ("+ own measured\nhistory", [("itself", "few_shot"),
                                         ("outside\nanalyst", "few_shot_other")]),
        ("abstract\ndescription", [("itself", "self_report"),
                                   ("AI in\ngeneral", "generic_report")]),
        ("+ exact\nevaluation items", [("itself", "informed_oracle"),
                                           ("AI in\ngeneral", "generic_oracle"),
                                           ("other\nmodels", "oracle_report_mean")]),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    W, STEP, GAP = 0.56, 0.84, 1.9
    x, xt, xl, seen = 0.0, [], [], set()
    edges = []  # (first bar x, last bar x) per group, for placing the gap arrows
    for gname, members in groups:
        xs, tops = [], []
        for label, method in members:
            row = per.get(method)
            if row is None or row.get("mean_r") is None:
                continue
            v = row["mean_r"]
            is_self = label == "itself"
            col = SELF if is_self else OTHER
            key = ("question is about the model itself"
                   if is_self else "same question about AI in general, an analyst, or other models")
            ax.bar(x, v, width=W, color=col, zorder=3, label=key if key not in seen else None)
            seen.add(key)
            top = v
            if row.get("ci_lo") is not None and row.get("ci_hi") is not None:
                ax.plot([x, x], [row["ci_lo"], row["ci_hi"]], color=INK, lw=1.1,
                        solid_capstyle="butt", zorder=4)
                top = row["ci_hi"]
            ax.text(x, top + 0.014, f"{v:+.2f}", ha="center", va="bottom", fontsize=11,
                    color=INK, zorder=5)
            ax.text(x, -0.018, label, ha="center", va="top", fontsize=12, color=MUTED,
                    linespacing=1.05)
            xs.append(x); tops.append(top)
            x += STEP
        if xs:
            xt.append(sum(xs) / len(xs)); xl.append(gname); edges.append((xs[0], xs[-1]))
            # Δ between the "itself" bar and its nearest not-self twin, drawn as a slim bracket.
            a, b = members[0][1], members[1][1]
            if a in per and b in per:
                d = per[b]["mean_r"] - per[a]["mean_r"]
                yb = max(tops[:2]) + 0.075
                ax.plot([xs[0], xs[0], xs[1], xs[1]], [yb - 0.012, yb, yb, yb - 0.012],
                        color=MUTED, lw=0.8, zorder=2)
                ax.text((xs[0] + xs[1]) / 2, yb + 0.008, f"$\\Delta$ {d:+.2f}", ha="center",
                        va="bottom", fontsize=10, color=MUTED)
        x += GAP
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=12, color=INK)
    ax.tick_params(axis="x", pad=46, length=0)
    ax.tick_params(axis="y", labelsize=11, colors=INK)
    ax.set_xlim(-0.55, x - GAP + 0.55)
    ax.set_ylim(-0.03, 0.62)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_ylabel("correlation with behavior", fontsize=12, color=INK)
    ax.axhline(0, lw=0.8, color=INK, zorder=2)
    ax.yaxis.grid(True, color="#e6e6e6", lw=0.7, zorder=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#bdbdbd")
    # Hollow block arrows in the gaps, pointing outward from the centre (abstract) group: each
    # side ADDS a different kind of information, so there is no single ordered information axis.
    if len(edges) == 3:
        ARROW_Y, SHAFT, HEAD_W, HEAD_L, PAD = 0.27, 0.045, 0.10, 0.30, 0.62
        for x0, x1 in ((edges[1][0] - PAD, edges[0][1] + PAD),   # centre -> left group
                       (edges[1][1] + PAD, edges[2][0] - PAD)):  # centre -> right group
            ax.arrow(x0, ARROW_Y, x1 - x0, 0.0, width=SHAFT, head_width=HEAD_W,
                     head_length=HEAD_L, length_includes_head=True, fc="white", ec=MUTED,
                     lw=1.3, zorder=2)
            ax.text((x0 + x1) / 2, ARROW_Y + HEAD_W / 2 + 0.012, "adding\ninformation",
                    ha="center", va="bottom", fontsize=10.5, color=MUTED, style="italic",
                    linespacing=1.05)
    ax.legend(fontsize=11, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              handlelength=1.0, handleheight=0.9, ncol=1, labelspacing=0.3)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.98, bottom=0.42)
    return _save(fig, out_dir, "fig_axes")


def fig_selfspec(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    """The like-for-like test: does an oracle predict ITS OWN behavior better than its peers'?"""
    evs = sorted(SELFSPEC, key=lambda e: -(SELFSPEC[e][0] - SELFSPEC[e][1]))
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = range(len(evs))
    inf = [SELFSPEC[e][0] - SELFSPEC[e][1] for e in evs]
    gen = [SELFSPEC[e][2] - SELFSPEC[e][3] for e in evs]
    w = 0.38
    ax.bar([i - w / 2 for i in x], inf, width=w, color="#4c72b0", label="asked about you")
    ax.bar([i + w / 2 for i in x], gen, width=w, color="#dd8452", label="asked about a generic agent")
    ax.axhline(0, ls="--", lw=1, color="gray")
    ax.set_xticks(list(x))
    ax.set_xticklabels([SHORT[e] for e in evs], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("self-advantage\n(own $-$ others' behavior)", fontsize=8)
    ax.set_title("Does the informed answer track the answerer specifically?", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    return _save(fig, out_dir, "fig_selfspec")


def make_figures(doc: dict[str, Any], out_dir: Path) -> list[Path]:
    return (fig_rq2_methods(doc, out_dir) + fig_rq1_evals(doc, out_dir)
            + fig_headroom(doc, out_dir) + fig_axes(doc, out_dir) + fig_selfspec(doc, out_dir))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", default="results/reports/evaluation.json",
                   help="evaluation export written by bp-evaluate")
    p.add_argument("--out-dir", default="paper/generated",
                   help="directory for the generated figures")
    args = p.parse_args()

    doc = json.loads(Path(args.json).read_text())
    paths = make_figures(doc, Path(args.out_dir))
    print("Wrote:\n  " + "\n  ".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
