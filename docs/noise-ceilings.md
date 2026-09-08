# Noise ceilings: how much dev correlation the targets can support (2026-07-04)

> **Exploratory note (dev split only).** The numbers below predate the final runs and the
> frozen-test evaluation; the paper's reported figures are the canonical ones.

Every method is scored as a Pearson r against **measured** per-condition rates, and those
measurements carry sampling noise. The maximum r any predictor can be *expected* to reach against
a measured target — even one that knows the true rates exactly — is `sqrt(reliability)`, where
reliability is the correlation between two independent measurements of the same conditions.
This bounds every number in the leaderboard, and where the ceiling is ~0 the eval simply does
not measure anything for that model: any observed correlation there is fit to a frozen noise
realization.

Scripts: `scripts/noise_ceiling.py` (split-half / parametric-bootstrap reliability, the primary
estimator) and `scripts/noise_ceiling_analytic.py` (independent cross-check: observed variance
minus unbiased binomial noise variance, with a bootstrap CI over conditions). The two agree
within ±0.05 everywhere. Dev split only.

## Estimation per eval

- **sycophancy_pushback / capability_mmlu**: real split-half over the per-question outcomes in
  `behavior_raw.json` (200 random within-subject splits), Spearman-Brown corrected to full
  length. Assumption-free.
- **propensitybench**: split-half over the 6 per-tactic Bernoulli outcomes (3/3 splits, SB).
  Conservative — tactic heterogeneity counts as noise. (The analytic estimator, which does not
  have that bias, agrees, so the conservatism is not driving the conclusions.)
- **discrimeval**: parametric bootstrap at the contrast grain — cell counts resampled
  ~ Binomial(n, p̂), the shared baseline cell resampled once per replicate, reliability = mean
  correlation between two replicates' gap vectors.

## Ceilings (dev, targets as measured at the time of the analysis)

| eval | llama-3.3 | maverick | deepseek | qwen |
|---|---|---|---|---|
| sycophancy_pushback (n≈20/subject) | 0.78 | 0.80 | **~0.0–0.15** ⚠ | **~0.15–0.27** ⚠ |
| capability_mmlu (n=20/subject) | 0.70–0.72 | 0.76–0.78 | 0.64–0.67 | 0.79–0.81 |
| propensitybench (n=6/condition) | **~0.0** ⚠ | 0.73–0.76 | 0.87–0.88 | ~0.38–0.43 ⚠ |
| discrimeval (n=50/cell, gap-scored) | 0.56 | 0.75 | 0.70 | 0.71 |

⚠ = unpowered cell. Analytic 90% CIs (see script output) are wide for the low cells — e.g.
deepseek sycophancy [0.00, 0.42] — but the point estimates say the between-condition variance is
sampling noise.

## Consequences

1. **Four cells are unpowered: deepseek+qwen on sycophancy, llama(+marginally qwen) on PB.**
   Deepseek flips ≤3/18 times per subject with 15 of 34 dev subjects at exactly zero (target sd
   0.05 ≈ its binomial noise); llama's PB rates sit in a sd-0.07 band on n=6 trials. Correlations
   observed against these targets — including cot_flip's deepseek sycophancy +0.35,
   llm_prediction's +0.39, and cross_model_mean's PB-llama +0.50 — are **not interpretable as
   signal**. A subtle trap these cells set: the target noise realization is *frozen*, so an
   accidental correlation persists across every fresh prediction batch and looks like a stable
   replicated result. Re-running predictions never re-rolls target noise; only re-measured
   targets (or the held-out test split) can. The earlier "sycophancy works on deepseek (+0.35)"
   reading of the cot_flip results is withdrawn on this basis
   (see docs/single-model-methods-sycophancy.md).
2. **Sycophancy for llama/maverick has real method headroom**: ceilings ~0.78–0.80 vs best
   methods +0.36/+0.45. The measurement supports far better prediction than any current method
   achieves.
3. **capability_mmlu and discrimeval are saturated by cross_model_mean** (within ~0.05 of the
   ceiling on most models: maverick capability +0.79 vs ~0.77 ceiling, deepseek +0.67 vs ~0.65,
   discrimeval similar). No method can measurably beat cross_model_mean on these evals as
   currently measured; single-model methods still have honest headroom below the ceiling.
4. **Raising the ceilings is a measurement question**: noise sd scales as 1/sqrt(n). Sycophancy
   at a ~5% flip base rate (strong models) needs n≈100+ questions/subject before per-subject
   differences resolve; PB's n=6 (one sample per pressure tactic) is the binding constraint for
   its weak cells.

## Action taken (2026-07-04)

Sycophancy behavior was re-measured at **100 questions per subject** for all four pool models
(previous: 20), overwriting `behavior_raw.json`/`targets.json`. Outcome (split-half, dev):

| sycophancy ceiling | llama-3.3 | maverick | deepseek | qwen |
|---|---|---|---|---|
| n=20 (old) | 0.78 | 0.80 | ~0.0–0.15 | ~0.15–0.27 |
| **n=100 (current)** | **0.94** | **0.94** | **0.91** | **0.83** |

So deepseek and qwen DO have real per-subject sycophancy structure — at n=20 it sat below the
noise floor, and their unpowered cells were a measurement artifact, not a property of the models.
All four sycophancy cells are now well-powered, with best methods at +0.37–0.48 against the new
targets (large honest headroom). Trained methods' committed predictions (cot_flip,
llm_prediction) were regenerated against the new targets; stateless prediction files remain
valid as elicited and were simply re-scored.

PB's weak cells (llama ~0, qwen ~0.4, n=6 samples/condition) remain the open measurement
problem, and their treatment in tuning macros and the paper (exclude vs flag) is an open
editorial decision.
