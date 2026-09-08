"""fig:overview — the hero figure, built around one concrete evaluation (Sycophancy).

Panel 1 makes "condition" concrete: one condition = one MMLU subject, with Llama-3.3-70B's
measured flip rates. Panel 2 shows what each prediction channel answers for one condition
(global_facts), all real recorded predictions. Panel 3 scores predicted vs. actual over the
34 dev subjects for the three main channels. Data is read live from results/, so the figure
regenerates with the pipeline.

Writes paper/generated/fig_overview.{png,pdf} and the caption macros in
paper/generated/hero.tex.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, FancyBboxPatch

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

SLUG = "llama-3.3-70b"
EVAL = "sycophancy_pushback"
EXAMPLE = "global_facts"

BLUE, ORANGE, GREY, RED, TEAL = "#4c72b0", "#dd8452", "#8c8c8c", "#c44e52", "#3a8a7c"


def load():
    spec = get_spec(EVAL)
    man = splits.load_manifest(EVAL)
    targets = common.load_json(f"results/{EVAL}/{SLUG}/targets.json")["targets"]
    dev = splits.split_keys(spec, targets, man, "dev")
    rates = {k: t["rate"] for k, t in targets.items() if k in dev and t.get("rate") is not None}
    preds = {}
    for m in ("self_report", "generic_report", "few_shot", "informed_oracle", "cross_model_mean"):
        p = common.load_json(f"results/{EVAL}/{SLUG}/predictions/{m}.json")["predictions"]
        preds[m] = {k: v.get("predicted_rate") for k, v in p.items()
                    if k in rates and v.get("predicted_rate") is not None}
    return rates, preds


def main() -> int:
    rates, preds = load()

    fig = plt.figure(figsize=(11.6, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.5, 1.05], wspace=0.36,
                          left=0.115, right=0.985, top=0.82, bottom=0.13)

    # ---- Panel 1: measure -----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    subjects = ["global_facts", "econometrics", "high_school_physics",
                "world_religions", "marketing"]
    subjects = [s for s in subjects if s in rates]
    vals = [rates[s] for s in subjects]
    y = range(len(subjects))
    ax1.barh(list(y), vals, color=[RED if s == EXAMPLE else GREY for s in subjects],
             height=0.62)
    for i, v in enumerate(vals):
        ax1.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=8.5)
    ax1.set_yticks(list(y))
    ax1.set_yticklabels([s.replace("_", " ") for s in subjects], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.06)
    ax1.set_xticks([0, 0.5, 1.0])
    ax1.set_xticklabels(["0%", "50%", "100%"], fontsize=8)
    ax1.set_xlabel("measured flip rate", fontsize=9)
    ax1.set_title("1. MEASURE behavior per condition\n"
                  "Sycophancy: does the model abandon a correct\n"
                  "MMLU answer when the user pushes back?\n"
                  "One condition = one MMLU subject.", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # ---- Panel 2: predict -----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1])
    ax2.set_axis_off()
    ax2.set_title(f"2. PREDICT the rate for each condition\n"
                  f"Example: \u201c{EXAMPLE.replace('_', ' ')}\u201d, measured flip rate "
                  f"{rates[EXAMPLE]:.0%}.", fontsize=9, loc="left")
    rows = [
        ("self-report", "\u201cHow often would YOU cave in on global-facts\nquestions?\u201d",
         preds["self_report"].get(EXAMPLE), GREY),
        ("generic report", "same question about \u201ca capable AI assistant\u201d",
         preds["generic_report"].get(EXAMPLE), ORANGE),
        ("history-informed self-prediction", "shown its measured rates on other subjects first",
         preds["few_shot"].get(EXAMPLE), TEAL),
        ("item-informed self-prediction",
         "\u201cHere is the exact exam question \u2026 would you cave?\u201d",
         preds["informed_oracle"].get(EXAMPLE), BLUE),
        ("cross-model mean (never asks)", "the other models' measured flip rate on this subject",
         preds["cross_model_mean"].get(EXAMPLE), RED),
    ]
    yy = 0.97
    for name, sub, v, col in rows:
        box = FancyBboxPatch((0.0, yy - 0.155), 0.80, 0.15,
                             boxstyle="round,pad=0.012", fc="#f4f4f4", ec="#cccccc",
                             transform=ax2.transAxes)
        ax2.add_patch(box)
        ax2.text(0.02, yy - 0.045, name, fontsize=8.5, fontweight="bold",
                 transform=ax2.transAxes, va="center", color=col)
        ax2.text(0.02, yy - 0.115, sub, fontsize=7.2, transform=ax2.transAxes,
                 va="center", color="#444444", style="italic")
        ax2.text(0.90, yy - 0.08, f"{v:.0%}", fontsize=11, fontweight="bold",
                 transform=ax2.transAxes, va="center", ha="center", color=col)
        yy -= 0.20

    # ---- Panel 3: score -------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2])
    ks = sorted(rates)
    series = [
        ("cross_model_mean", "cross-model mean", RED, "o"),
        ("informed_oracle", "item-informed self-pred.", BLUE, "^"),
        ("self_report", "self-report", GREY, "s"),
    ]
    panel_r = {}
    for m, label, col, marker in series:
        xs = [rates[k] for k in ks if k in preds[m]]
        ys = [preds[m][k] for k in ks if k in preds[m]]
        r = metrics.pearson(list(zip(xs, ys)))
        panel_r[m] = r
        rtxt = "$r$ undefined (constant 0)" if r is None else f"$r={r:+.2f}$"
        ax3.scatter(xs, ys, s=12, color=col, marker=marker, alpha=0.75,
                    label=f"{label}: {rtxt}")
    ax3.plot([0, 0.6], [0, 0.6], ls=":", lw=0.8, color="#bbbbbb")
    ax3.set_xlim(-0.02, 1.0); ax3.set_ylim(-0.03, 0.6)  # predictions live below 0.6
    ax3.set_xlabel("actual flip rate", fontsize=9)
    ax3.set_ylabel("predicted flip rate", fontsize=9)
    ax3.tick_params(labelsize=8)
    ax3.set_title("3. SCORE across conditions\n(34 dev subjects, this model)",
                  fontsize=9, loc="left")
    ax3.legend(fontsize=7.2, frameon=False, loc="upper left", handletextpad=0.15)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)

    out = "paper/generated"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig_overview.{ext}", dpi=170)

    # The fig:overview caption quotes panel 3's correlations, so export them as macros
    # instead of leaving the caption to drift from the figure beside it.
    tex = ["% Auto-generated by scripts/hero_figure.py -- DO NOT EDIT BY HAND.",
           "% Panel-3 correlations of the fig:overview walkthrough cell (one eval, one model);",
           "% NA where the channel is constant and r is undefined.",
           f"\\newcommand{{\\heroNConditions}}{{{len(ks)}}}"]
    for m, tag in (("cross_model_mean", "Xmm"), ("informed_oracle", "Io"),
                   ("self_report", "Sr")):
        r = panel_r.get(m)
        tex.append(f"\\newcommand{{\\hero{tag}R}}{{" + ("NA" if r is None else f"{r:+.2f}") + "}")
    Path(f"{out}/hero.tex").write_text("\n".join(tex) + "\n")
    print(f"Wrote {out}/fig_overview.png/.pdf and {out}/hero.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
