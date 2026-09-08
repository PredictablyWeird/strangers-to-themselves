#!/usr/bin/env python3
"""Identity-transfer figure for the paper (main text, Experiment 1/3).

Scores model A's elicited predictions against model B's measured behavior for all (A, B) pairs
(dev split, default pool). Self-knowledge requires diagonal > off-diagonal. Emits
``paper/generated/fig_identity_transfer.{png,pdf}`` with three panels:

  (a) self_report on DiscrimEval          -- the verbal method's best eval: no diagonal
  (b) informed_oracle on Capability(MMLU) -- a modest but real diagonal
  (c) diag-vs-off-diag scatter over every (elicited method, eval) cell; points on y=x carry
      no self-specific signal

Weak-ceiling (eval, model) cells from the evaluation export are excluded, matching the
leaderboard filter. Constant (no-ordering) elicitations are skipped rather than scored 0, so
diagonal means are computed over models that produced an ordering at all.

Offline: reads files already in results/, no API calls. Helpers mirror
scripts/method_diagnostics.py (section [4]).
"""
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from behavior_prediction import common, metrics, splits  # noqa: E402
from behavior_prediction.evals import get_spec  # noqa: E402

import os
REPORT = os.environ.get("BP_EVAL_JSON", "results/reports/evaluation.json")
SPLIT = os.environ.get("BP_SPLIT", "dev")
METHODS = ["self_report", "value", "pairwise", "informed_oracle", "oracle_pairwise"]
MIN_N = 5
OUT = Path("paper/generated")

SHORT = {
    "llama-3.3-70b": "llama-3.3",
    "llama-4-maverick": "maverick",
    "deepseek-v4-flash-low": "deepseek",
    "gemini-3.1-flash-lite-low": "gemini",
    "gpt-5.4-nano-low": "gpt-nano",
    "qwen3.7-plus-low": "qwen",
    "claude-sonnet-5-off": "sonnet-off",
    "claude-sonnet-5-low": "sonnet-low",
    "gpt-5.5-off": "gpt5.5-off",
    "gpt-5.5-low": "gpt5.5-low",
    "deepseek-v4-pro-off": "ds-pro-off",
    "deepseek-v4-pro-low": "ds-pro-low",
}


def vec_pearson(a: dict, b: dict):
    ks = sorted(set(a) & set(b))
    return metrics.pearson([(a[k], b[k]) for k in ks]), len(ks)


def is_bias(spec):
    return getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast"


def scored_vecs(spec, targets, preds, dev):
    if is_bias(spec):
        sc = spec.score_contrasts(targets, preds, keys=dev)
        ok = {k: v for k, v in sc.items() if v[0] is not None and v[1] is not None}
        return {k: v[0] for k, v in ok.items()}, {k: v[1] for k, v in ok.items()}
    a = {k: t["rate"] for k, t in targets.items() if k in dev and t.get("rate") is not None}
    p = {k: v for k, v in preds.items() if k in a and v is not None}
    return {k: a[k] for k in p}, p


def load_eval(ev, slugs, weak=()):
    """weak: (eval, model) cells whose noise-ceiling CI includes zero -- excluded entirely,
    matching the leaderboard's method-blind filter (their targets carry no measurable signal,
    so both their diagonal and their use as targets would be spurious)."""
    spec = get_spec(ev)
    manifest = splits.load_manifest(ev)
    tdoc, dev_keys, actual = {}, {}, {}
    for slug in slugs:
        p = Path(f"results/{ev}/{slug}/targets.json")
        if not p.exists() or (ev, slug) in weak:
            continue
        tdoc[slug] = common.load_json(p)["targets"]
        dev_keys[slug] = splits.split_keys(spec, tdoc[slug], manifest, SPLIT)
        rates = {k: t.get("rate") for k, t in tdoc[slug].items()}
        actual[slug], _ = scored_vecs(spec, tdoc[slug], rates, dev_keys[slug])
    return spec, tdoc, dev_keys, actual


def transfer_matrix(ev, method, slugs, spec, tdoc, dev_keys, actual):
    """dict (source, target) -> r, using each source's own targets for scoring space."""
    pvecs = {}
    for slug in slugs:
        p = Path(f"results/{ev}/{slug}/predictions/{method}.json")
        if slug not in tdoc or not p.exists():
            continue
        preds = {k: v.get("predicted_rate")
                 for k, v in common.load_json(p)["predictions"].items()}
        _, pv = scored_vecs(spec, tdoc[slug], preds, dev_keys[slug])
        if len(set(pv.values())) > 1:  # constant predictions carry no ordering
            pvecs[slug] = pv
    mat = {}
    for a in pvecs:
        for b in actual:
            r, n = vec_pearson(pvecs[a], actual[b])
            if r is not None and n >= MIN_N:
                mat[(a, b)] = r
    return mat


def draw_heatmap(ax, mat, slugs, title):
    order = [s for s in slugs if any(a == s for a, _ in mat)]
    grid = [[mat.get((a, b)) for b in order] for a in order]
    shown = [[v if v is not None else 0.0 for v in row] for row in grid]
    im = ax.imshow(shown, cmap="RdBu_r", vmin=-1, vmax=1)
    annotate = len(order) <= 8   # 12-model matrices are unreadable with per-cell text
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            if v is None:
                ax.text(j, i, "--", ha="center", va="center", fontsize=5, color="gray")
            elif annotate:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                        fontweight="bold" if i == j else "normal")
            elif i == j:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                        fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels([SHORT.get(s, s) for s in order], rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels([SHORT.get(s, s) for s in order], fontsize=6)
    ax.set_title(title, fontsize=8)
    return im


def main() -> int:
    doc = common.load_json(REPORT)
    slugs, evals = doc["models"], doc["evals"]
    weak = {(c["eval"], c["model"]) for c in doc.get("dropped_weak", [])}
    cache = {ev: load_eval(ev, slugs, weak) for ev in evals}

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    for ax, ev, method, label in [
            (axes[0], "discrimeval", "self_report", "(a) self_report, DiscrimEval"),
            (axes[1], "capability_mmlu", "informed_oracle", "(b) informed_oracle, Capability")]:
        mat = transfer_matrix(ev, method, slugs, *cache[ev])
        im = draw_heatmap(ax, mat, slugs, label)
    axes[0].set_ylabel("prediction source model", fontsize=7)
    axes[0].set_xlabel("behavior target model", fontsize=7)
    axes[1].set_xlabel("behavior target model", fontsize=7)
    fig.colorbar(im, ax=axes[:2], fraction=0.035, pad=0.02, shrink=0.85)\
       .ax.tick_params(labelsize=6)

    ax = axes[2]
    summary = []
    for ev in evals:
        for method in METHODS:
            mat = transfer_matrix(ev, method, slugs, *cache[ev])
            diag = [r for (a, b), r in mat.items() if a == b]
            off = [r for (a, b), r in mat.items() if a != b]
            if len(diag) >= 3 and off:
                summary.append((ev, method, mean(diag), mean(off)))
    for ev, method, d, o in summary:
        oracle = method in ("informed_oracle", "oracle_pairwise")
        hit = method == "informed_oracle" and ev in ("capability_mmlu", "propensitybench")
        ax.scatter(o, d, s=34 if hit else 18, zorder=3,
                   color="#c44e52" if oracle else "#4c72b0",
                   marker="^" if oracle else "o",
                   label=None)
        if hit:
            dy = 6 if ev == "capability_mmlu" else -14
            ax.annotate(f"informed_oracle\n({ev.split('_')[0]})", (o, d), fontsize=6,
                        xytext=(7, dy), textcoords="offset points")
    ax.scatter([], [], color="#4c72b0", marker="o", s=18, label="verbal methods")
    ax.scatter([], [], color="#c44e52", marker="^", s=18, label="informed-oracle variants")
    ax.legend(loc="upper left", fontsize=6, frameon=False)
    lo = min([-0.1] + [min(o, d) for _, _, d, o in summary]) - 0.05
    hi = max([0.6] + [max(o, d) for _, _, d, o in summary]) + 0.05
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1, color="gray")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("off-diagonal mean r (others' behavior)", fontsize=7)
    ax.set_ylabel("diagonal mean r (own behavior)", fontsize=7)
    ax.set_title("(c) all elicited methods x evals", fontsize=8)
    ax.tick_params(labelsize=7)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_identity_transfer.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # The caption quotes panel (b)'s diagonal/off-diagonal, so export every panel's numbers
    # as macros rather than leaving them to be copied by hand out of this printout.
    tex = ["% Auto-generated by scripts/identity_transfer_figure.py -- DO NOT EDIT BY HAND.",
           "% Per (eval, method): mean diagonal r (own behavior) and off-diagonal r (others').",
           "% These are the numbers quoted in the fig:identity caption."]
    for ev, method, d, o in summary:
        tag = "".join(w.capitalize() for w in method.split("_")) + \
              "".join(w.capitalize() for w in ev.split("_"))
        tex += [f"\\newcommand{{\\idtfig{tag}Diag}}{{{d:+.2f}}}",
                f"\\newcommand{{\\idtfig{tag}Off}}{{{o:+.2f}}}"]
        print(f"{ev:<22} {method:<18} diag {d:+.2f}  off {o:+.2f}")
    (OUT / "identity_transfer_fig.tex").write_text("\n".join(tex) + "\n")
    print("Wrote paper/generated/fig_identity_transfer.{png,pdf} and identity_transfer_fig.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
