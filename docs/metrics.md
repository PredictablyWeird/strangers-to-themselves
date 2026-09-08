# Prediction targets beyond correlation — what's useful in practice (2026-07-07)

> **Design note.** Records a design decision, not results; it predates the final runs.

So far every method in this repo is scored by **Pearson `r`** between the predicted per-condition
rate and the actual per-condition rate across dev conditions (with MAE reported but treated as
"not meaningful" for the ranking-style methods). This note argues that `r` is a narrower target
than it looks, lays out prediction targets that map more directly onto the decisions a behavior
predictor is supposed to support, and draws out what those targets imply for **designing new
evals**.

The framing throughout: *pick the target by naming the downstream decision first.* The metric you
optimize silently defines what a "good" predictor is, so it should be the metric of the decision,
not a convenient statistic.

## Why correlation under-serves the practical use

Pearson `r` across conditions is **scale-free, shift-free, rank-flavored, and equal-weight**. Four
consequences bite in practice:

- **Blind to absolute level.** `r = 0.9` is compatible with predicting 2% everywhere the truth is
  40–90%. If the point is to *avoid running the expensive behavioral eval*, you must act on the
  number itself, and `r` certifies none of that. (The repo's "MAE is meaningless for ranking
  methods" convention is a statement about the *methods*, not about what's useful — it concedes
  the methods only predict ordering.)
- **Rewards the easy middle.** Safety utility lives in the tail — the few worst conditions. A
  global `r` can be high while the top-k ranking is wrong, or low while the top-k is perfect.
- **Needs a measurable spread of ground truth to even compute.** Much of the pool is already
  unmeasurable this way: deepseek/qwen sycophancy were noise-floor-limited, PropensityBench at
  n=6 is undefined (see [`noise-ceilings.md`](noise-ceilings.md)). A target that collapses to
  "undefined" on the cases you most want to screen is a weak target.
- **Symmetric.** Missing a dangerous condition and flagging a safe one cost the same under `r`. In
  deployment they don't — false negatives dominate the loss.

None of this makes `r` wrong; it makes it a **secondary diagnostic**, good for "which condition is
worse," blind to "is anything bad" and "how bad."

## Targets tied to the decision they serve

| Decision | Better target | Metric |
|---|---|---|
| "Is this eval/model worth full measurement?" (triage) | Does **any** condition cross a risk threshold? | AUROC / PR on the binary "bad" label; **recall@k** of the worst conditions |
| "Where do I spend measurement budget?" (active selection) | Quality of the **top** of the ranking, not the whole order | precision@k, NDCG, or **ranking regret** = true rate of the worst condition you'd have skipped |
| "Act without ever measuring truth" (extrapolation) | The **absolute rate**, calibrated | Brier / ECE / calibration curves (MAE brought back deliberately) |
| "Is version N+1 worse than N here?" (regression) | The **delta** across model versions per condition | sign-accuracy / AUROC on the "got worse" label |
| "Should I trust this prediction?" (abstention) | The predictor's **own confidence** | selective-prediction (coverage–risk) curves; value-of-information |

Two of these are more than metric swaps and deserve their own treatment.

### Breaking point along an intensity axis

PropensityBench **already measures this and pools it away.** The per-condition target records
`level_hist` and `mean_trigger_level` — *the pressure step at which the model caved*, out of
`PB_MAX_LEVEL = 12` — and the current pipeline collapses that to a pooled trigger rate.

Predicting the **breaking point** (a survival / threshold target: "how much pressure until it
flips") is:

- **more interpretable** — "this model holds to level 8, that one caves at level 2" is a sentence
  a stakeholder can act on, unlike "0.42 vs 0.18";
- **more robust to sampling noise** than a rate — an ordinal step is coarser and steadier than a
  fraction estimated from few samples;
- **graceful under censoring** — a condition nobody triggers is *right-censored* (survived the
  whole ladder), not a flat, unmeasurable 0. This directly attacks the "everything denies to 0"
  degeneracy we hit repeatedly (see
  [`style-and-deniable-elicitation.md`](style-and-deniable-elicitation.md)).

This is the single most underused signal already sitting in the data.

### Cross-model transfer and version-delta

The practically important generalization is **not** held-out *conditions* within a model (what
within-model held-out-condition `r` measures). It is:

- a **new model** — which is what `cross_model_mean` already leans on: "does a predictor fit on
  models A/B/C rank model D's conditions?";
- a **new version** — "does the predictor catch the conditions that regressed from N to N+1?"

These are the questions a deployment team actually asks. Note the connection to a recurring result
in this repo: the strong single-model effects (`cot_flip`, the maverick slang self-report, the
llama-3.3 randomized-response trick) are **idiosyncratic per model** and do not transfer — so a
transfer-oriented target would score them honestly (poorly) where a within-model `r`, computed
separately per model, flatters them.

## Implications for designing new evals

The prediction target and the eval design are coupled. Bake these in up front:

1. **Build in an intensity / dose axis** (like PB's pressure ladder) and **log per-item, not just
   pooled rates.** That one choice unlocks breaking-point targets, calibration, and
   censoring-aware analysis for free. Pooled-only evals foreclose all of them.
2. **Design for a measurable ground truth.** Estimate the noise ceiling and rate spread *before*
   investing (the `noise-ceilings.md` machinery). An eval whose conditions all sit at 0, or all at
   the noise floor, cannot validate any predictor — correlation or otherwise. Enough
   samples/condition and a genuine spread of true rates are eval-design *requirements*, not
   afterthoughts.
3. **Define an actionable threshold as part of the eval.** "Bad = triggers above X% / breaks
   before level L" gives the binary labels that detection and cost-weighted metrics need — and
   forces you to state what decision the eval informs.
4. **Include held-out models, and a version pair if possible**, so transfer and
   regression-detection are measurable rather than assumed.

## Recommendation

- Adopt two headline targets alongside `r`:
  - **(a) detection** — recall@k / AUROC on a thresholded "bad" label, for the triage use case;
  - **(b) calibration** — Brier / ECE on the absolute rate, for the act-without-ground-truth use
    case.

  Keep `r` as a secondary diagnostic, not the objective.
- For the **next eval built**, the highest-leverage design decision is an **explicit intensity
  axis with per-item logging**, so "predict the breaking point" becomes available. It is more
  decision-relevant and more noise-robust than any pooled rate, and PB shows the plumbing already
  exists.

### Caveat on our own methods

These targets are stricter, so some current "winners" will look worse under them. A sign-learned,
scale-free method (`cot_flip`; the slang self-reports; randomized-response) has **no calibrated
absolute level** — it would score fine on recall@k but poorly on Brier. That is a feature: it
surfaces that most methods so far predict *ordering*, and the practically valuable thing they do
not yet deliver is a *trustworthy number*.

## Concrete next steps (if pursued)

- Add `recall_at_k`, detection `AUROC`, and a `brier` / calibration score to
  `behavior_prediction/metrics.py`, reported next to `correlation` in the existing dev tables.
- Add a PB **breaking-point** target derived from `mean_trigger_level` / `level_hist`
  (survival-style, censoring-aware), and re-score the current methods against it on dev.
- Once a second version of any pooled model exists, add a **version-delta** ("got worse") label
  and score sign-accuracy / AUROC on it.
