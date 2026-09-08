# Why cross_model_mean wins — offline diagnostics (2026-07-06)

> **Exploratory note (dev split only).** The numbers below predate the final runs and the
> frozen-test evaluation; the paper's reported figures are the canonical ones.

> **Follow-up (2026-07-06): the informed-oracle upper bound.** See the dedicated section at the
> bottom of this file. Headline: shown the verbatim measured item and asked to predict its own
> score, the model still loses to `cross_model_mean` on the absolute-rate evals (sycophancy,
> capability) — the bottleneck there is introspective, not informational. **But on discrimeval,
> asked for the demographic gap *directly* (comparative, both applicant orders swapped to cancel
> position bias), the oracle *beats* `cross_model_mean` (macro +0.57 vs +0.50, ahead on 4/6
> models) and carries real self-specific signal beyond the pool prior (partial r +0.30..+0.46,
> positive 6/6; transfer diag +0.57 vs off-diag +0.43).** The elicitation **grain** is decisive:
> predicting the gap directly rather than differencing two absolute rates took discrimeval from
> +0.32 to +0.51, and cancelling position bias with the order swap added +0.51 to +0.57.


Question: `cross_model_mean` (a no-self-knowledge baseline) tops the leaderboard — is that
because the models are too incoherent for self-prediction to work? Answered with **offline**
analyses over existing `results/` files (dev split, all 6 models, PB excluded by design —
its harm framing is out of scope here). Scripts: `scripts/method_diagnostics.py` (variance
decomposition, partial correlations, prediction reliability, identity transfer, own-behavior
baseline) and `scripts/method_stacking.py` (out-of-fold stacked combinations).

## TL;DR

1. **"Too incoherent" is the wrong diagnosis.** Verbal self-reports are highly *reliable*
   (split-half reliability of cot_flip/pairwise/self_report predictions ≈ 0.8–0.95): the models
   give stable answers. They are stably **generic and/or miscalibrated** — identity-transfer
   matrices show model A's elicited predictions score model B's behavior as well as A's own
   (diag ≈ off-diag for every method on all three evals). Elicited reports carry a shared
   difficulty prior, not privileged self-knowledge. The one genuinely sample-noise-limited
   method is `behavioral_sampling` (pred reliability −0.2…+0.5 → more runs would help it).
2. **The targets are NOT mostly shared difficulty on sycophancy/discrimeval.** Residual
   reliability (targets minus their xmm regression) leaves ceilings of 0.65–0.94 on sycophancy
   and 0.43–0.72 on discrimeval — large reliable *model-specific* variance cross_model_mean can
   never capture. Only capability_mmlu is xmm-saturated (xmm/ceiling ≈ 0.9–1.0, residual
   reliability ≈ 0–0.2 for most models).
3. **New single-model baseline that beats cross_model_mean on sycophancy: own capability error
   rate.** r(flip rate, own measured MMLU error) = +0.46…+0.76, macro **+0.61** vs xmm's +0.58
   (beats it on 4/6 models; gemini +0.76 vs +0.49, llama-3.3 +0.70 vs +0.40). No other models,
   no introspection — behavioral self-measurement. Partial r given xmm is +0.24…+0.71: mostly
   *unique* signal.
4. **Existing methods DO carry unique signal beyond the pool prior** — it's hidden by raw-r
   ranking. Partial r given xmm on sycophancy: llm_prediction up to +0.72 (llama), cot_flip
   +0.30…+0.58 on 5/6 models, pairwise positive on 4/6. Out-of-fold stacking confirms it's
   usable: **xmm + own_err + llm_prediction = macro +0.68** (vs raw xmm +0.58), positive on all
   six models (+0.55…+0.73).
5. **Pool composition is a real xmm weakness.** Behavioral similarity is clustered
   (deepseek/qwen/gpt-nano/maverick r ≈ 0.7–0.86 on sycophancy; gemini and llama-3.3 are
   outliers at r ≈ 0.15–0.4). For clustered models the single most-similar donor beats the
   pooled mean (best1 +0.86 vs xmm +0.66 for deepseek); for outlier models xmm degrades badly
   (llama-3.3 +0.40 vs its 0.94 ceiling). xmm's headline number rides on the pool happening to
   contain similar models.

## Numbers

### Target decomposition (dev): how much of the reliable variance is shared with the pool?

rel = split-half reliability of the targets (SB-corrected; parametric bootstrap for
discrimeval); ceiling = sqrt(rel); r_xmm = leave-one-out cross-model mean vs targets
(matches the stored cross_model_mean predictions); xmm/ceil = fraction of the *achievable*
correlation xmm already gets; resid_ceil = ceiling for predicting the xmm-residual
(the model-specific part).

| eval | model | ceil | r_xmm | xmm/ceil | resid_ceil |
|---|---|---|---|---|---|
| sycophancy | deepseek | 0.91 | +0.66 | 0.73 | 0.84 |
| sycophancy | gemini-flash-lite | 0.92 | +0.49 | 0.53 | 0.90 |
| sycophancy | gpt-5.4-nano | 0.83 | +0.64 | 0.77 | 0.71 |
| sycophancy | llama-3.3 | 0.94 | +0.40 | 0.42 | 0.94 |
| sycophancy | maverick | 0.95 | +0.57 | 0.60 | 0.92 |
| sycophancy | qwen | 0.83 | +0.71 | 0.85 | 0.65 |
| capability | (all six) | 0.68–0.84 | +0.62…+0.76 | 0.73–1.03 | ~0–0.75 |
| discrimeval | (all six) | 0.51–0.75 | +0.33…+0.65 | 0.45–1.12 | 0.43–0.72 |

Sycophancy (at the n=100 targets) has *more* reliable model-specific variance than shared
variance for llama-3.3, maverick and gemini. capability_mmlu is the only eval where "nothing
model-specific to predict" is (approximately) true — deepseek's residual reliability there is
−0.01, i.e. exactly zero, and xmm is at its ceiling.

### Method coherence vs validity (sycophancy, dev)

pred_rel = split-half reliability of the method's own per-condition predictions across its
samples (SB-corrected): "does the model give the same answer twice?". partial = r with targets
controlling for xmm: "signal beyond the pool prior".

| method | pred_rel range | raw r (macro) | partial r given xmm |
|---|---|---|---|
| cot_flip (raw, pre-sign) | 0.78–0.94 | +0.24 | +0.30…+0.58 (5/6 models; gemini −0.28) |
| pairwise | 0.77–0.91 | ~+0.0 | +0.34…+0.58 on the deepseek/qwen/gpt cluster; −0.56 llama |
| self_report(-honest) | 0.2–1.0 | ~0 | ~0 (near-constant answers) |
| llm_prediction(-full) | n/a (1 call) | +0.35 | **+0.14…+0.72, positive 6/6** |
| behavioral_sampling | **−0.16…+0.46** | +0.03 | mixed | 

Reading: high pred_rel + low validity = coherent-but-wrong self-models (the llama cot_flip
inversion is the extreme case: r_raw −0.50, perfectly stable). The methods are not limited by
sampling noise (except behavioral_sampling); they are limited by *what* the reports contain.

### Identity transfer (self-specificity test)

Score model A's elicited predictions against model B's targets, all 30 ordered pairs + 6 diag.
If self-reports contained self-knowledge, diag ≫ off-diag. Result: **diag ≈ off-diag for every
method on every eval** (largest gap: self_report on discrimeval, +0.36 vs +0.29; cot_flip
sycophancy |r| +0.29 vs +0.25). Whatever signal elicited methods carry is a generic
difficulty/severity prior, exchangeable across models.

### Own-behavior cross-eval baseline + stacking (sycophancy, dev)

| model | own_err raw r | xmm raw r | OOF stack xmm+own+llmp |
|---|---|---|---|
| deepseek | +0.51 | +0.66 | +0.70 |
| gemini-flash-lite | +0.76 | +0.49 | +0.73 |
| gpt-5.4-nano | +0.46 | +0.64 | +0.62 |
| llama-3.3 | +0.70 | +0.40 | +0.72 |
| maverick | +0.72 | +0.57 | +0.73 |
| qwen | +0.52 | +0.71 | +0.55 |
| **macro** | **+0.61** | +0.58 | **+0.68** |

own_err needs no fitting (correlation is scale-free), so its raw r is its honest score, same
rules as cross_model_mean. The stack is fit strictly out-of-fold (same CV folds as the trained
methods). Caveat: sycophancy items and capability items are drawn from the same MMLU subject
pools (different draws/sample sizes, separate measurement runs), so this is "behavior on a
related eval", not fully independent data — that is exactly the point (behavioral
self-measurement beats both verbal introspection and other-model transfer), but it should be
framed as a cross-eval transfer method, not free knowledge.

## What this means for the paper / next steps

- Reframe the leaderboard question: the interesting quantity is **incremental validity over
  the cross-model prior** (partial r / stacked gain), not raw r. cross_model_mean is best
  standalone but is provably missing the majority of the reliable variance on 2 of 3 evals.
- Promote two new baselines into the benchmark proper:
  **own_behavior_proxy** (single-model; predict eval X from measured behavior on related eval Y)
  and **stacked prior + method** (xmm + own_err + llm_prediction, OOF weights). The first is a
  fairness upgrade for the "no other models available" story; the second is the honest
  best-achievable reference.
- cross_model_mean variants worth one experiment: similarity-weighted donor mean (weights from
  capability-profile similarity, which needs no target-eval labels of the subject model) — the
  best-single-donor numbers (+0.86 vs pooled +0.66) say there's headroom; and a pool-ablation
  (xmm with only-dissimilar donors) to show its fragility.
- API-requiring follow-ups (not run here): behavioral micro-probes as the general form of
  own_err (see the excluded-probe results in docs/single-model-methods-sycophancy.md — same
  mechanism, +0.43 macro); an "ask model A about model B" elicitation to confirm the transfer
  result causally; paraphrase-robustness of self-reports (coherence across rephrasings rather
  than resamples).
- The "models are incoherent" hypothesis is rejected in the resampling sense and replaced by:
  **self-reports are reliable, generic, and (for llama) systematically inverted; model-specific
  signal exists and is reachable behaviorally, not verbally.**

---

# The informed-oracle upper bound (2026-07-06)

Question raised in review: `cross_model_mean` (no self-knowledge) tops the leaderboard — is that
because self-prediction is fundamentally impossible, or because the abstract elicitation prompts
withhold information the model would need? The **informed_oracle** method settles it by removing
the information bottleneck entirely: for each measured item it shows the model the *verbatim
prompt(s) sent during measurement* plus a description of the protocol (WITHOUT eliciting the
behavior), and asks it to predict its own per-item score; the condition prediction is the mean
over items. Deliberately over-informed — a diagnostic ceiling on what verbal self-prediction
could achieve with maximal item information, not a deployable method.

Implementation: `behavior_prediction/methods/elicited.py::InformedOracle` (+ the
`EvalSpec.oracle_items` hook, per-eval in `evals/{sycophancy_pushback,capability_mmlu,
discrimeval,propensitybench}.py`). One call per item, k=20 items/condition, greedy (temperature 0;
reasoning models keep their default temperature). The numbers below come from a 6-model × 3-eval
dev run that predates the PropensityBench oracle (see below), so PB is absent from them. Report
script: `scripts/oracle_report.py`; partial-r and transfer via `scripts/method_diagnostics.py`.

**Elicitation grain (this matters — see Result 1).** On the absolute-rate evals (sycophancy,
capability) the oracle predicts a per-item rate (`parse_prediction_tag`, 0–100). On **discrimeval
(a bias_contrast eval) it predicts the demographic gap *directly*** — for each template it is
shown *both* demographic fills (VERSION A = baseline person, VERSION B = the compared group), the
bias definition (yes-rate for B minus yes-rate for A, in points), and asked for that signed
difference (`parse_signed_prediction_tag`, −100..100); predictions are keyed by contrast and
consumed as-is by `score_contrasts`. Each template is asked in **both applicant orders**
(A=baseline/B=group and A=group/B=baseline, the swapped reading negated) so a constant tendency
to favor whichever applicant is shown second cancels — the same swap the other comparative
methods use. An earlier per-cell version (predict each demographic's absolute yes-rate, let the
scorer difference them) scored far worse (+0.32 macro) — differencing two independently-noisy
~0.9 rate predictions destroys the small gap. Two fixes compound: predicting the gap directly
(+0.32 → +0.51 macro) and cancelling position bias with the order swap (+0.51 → +0.57).

**PropensityBench (added 2026-07-09).** PB has no single measured prompt to exhibit — behavior emerges from a multi-turn escalating-pressure
loop — so the exhibit shows the **initial prompt of a run, verbatim** (the harness system prompt,
which names the forbidden tool, its stated consequences, and the full tool set; plus the first user
turn), and the `protocol` **explains the escalating pressure in prose** rather than transcribing it:
the approved tool always fails; a ladder of ≤12 messages arrives one at a time, all from one
pressure tactic, rising monotonically from a junior sender and a remote deadline to an executive and
an imminent catastrophe; the run stops the moment the forbidden tool is called; the outcome is
binary per run. Showing the real 12-turn transcript instead would dwarf the initial prompt and hand
the model the answer for the levels it is being asked to reason about.

The system prompt is *reconstructed*, not paraphrased: `evals/propensitybench.py` vendors the
harness's `SYSTEM_PROMPT_TEMPLATE`/`DEFAULT_INTERACTION_PROTOCOL` byte-for-byte and re-renders them
from the harness inputs (`$PB_SCENARIOS`, default `../propensity-evaluation/data/full`).
`tests/test_pb_oracle.py` diffs our output against the harness's own `Agent.create_system_prompt`
whenever a clone is on disk, so a harness bump can't silently desync the exhibit. Without a clone
there is no verbatim material, so `oracle_items` warns and returns no items (prediction `None`)
rather than guessing.

One item per **(task-scenario, pressure-tactic)** run, which makes the oracle's unweighted item mean
pool exactly what the target pools at the default `task_scenario` grain (a condition *is* its 6
tactic runs). `k` therefore caps whole task-scenarios (`k // n_tactics`, at least one), never a
partial tactic set — that would reweight the tactics against the target. The per-item ask names the
tactic, what it escalates along, and the forbidden tool.

Because `oracle_xmm`/`oracle_xmm_learned` gate on `InformedOracle.applies_to`, both ensembles now
cover PB as well (their PB predictions are not generated yet).

**PB dev result (6 models, 542 items, every item parsed, 0 parse failures):**

| model | oracle r | MAE | pred mean | actual mean |
|---|---|---|---|---|
| gpt-5.4-nano-low | **+0.84** | 0.09 | 0.06 | 0.08 |
| llama-4-maverick | +0.56 | 0.20 | 0.37 | 0.50 |
| gemini-3.1-flash-lite-low | +0.45 | 0.39 | 0.03 | 0.42 |
| llama-3.3-70b | +0.41 | 0.60 | 0.36 | 0.97 |
| deepseek-v4-flash-low | +0.10 | 0.57 | 0.03 | 0.60 |
| qwen3.7-plus-low | **−0.24** | 0.17 | 0.01 | 0.16 |

Macro **+0.35 [+0.09, +0.60]** — beats `cross_model_mean` (+0.33) but not `pairwise` (+0.42), and
the CI spans nearly the whole range: the per-model spread (−0.24 to +0.84) is the story, not the
mean.

**Every model under-predicts, most of them severely.** Predicted means sit at 0.01–0.37 against
actual means of 0.08–0.97; llama-3.3-70b predicts 0.36 where it actually triggers 0.97 of the time.
Read the correlations against that: MAE is not small, and on the models that *do* correlate, the
oracle is ranking conditions it thinks are uniformly safe. Note also how little target variance
several models offer — llama-3.3-70b's actual rates span only 0.80–1.00 (sd 0.07) over 20
conditions, so its +0.41 is estimated against almost no spread. The information bottleneck is
clearly not what limits self-prediction here either (cf. Result 1).

## Result 1 — introspective failure on rates; competitive on the direct gap

Dev Pearson r (oracle output is a calibrated rate/gap, so MAE is meaningful here, unlike
pairwise/cot_flip):

| eval | oracle macro r | cross_model_mean macro r | ceiling range | cells oracle beats xmm |
|---|---|---|---|---|
| sycophancy_pushback | +0.31 | +0.58 | 0.83–0.95 | 0 / 6 |
| capability_mmlu | +0.32 | +0.69 | 0.68–0.84 | 0 / 6 |
| discrimeval (direct gap, swapped) | **+0.57** | +0.50 | 0.55–0.76 | **4 / 6** |

**Absolute-rate evals — the information bottleneck is not the explanation.** On sycophancy and
capability the oracle never reaches xmm and never approaches the ceiling. Extremes: qwen
sycophancy is its best cell (+0.67, still < xmm +0.71); deepseek *capability* is −0.06 (handed
its own MMLU questions, it cannot tell which subjects it is better or worse at, ceiling 0.68);
gemini sycophancy is +0.00. With the exact item in hand the model still tracks its own
between-condition *rate* worse than a prior that never consults it — the failure is
introspective, not informational. (MAE corroborates a separate miscalibration story: llama-3.3
sycophancy MAE 0.45 — it wildly over-predicts its own flip *level* — vs well-calibrated
capability MAEs 0.05–0.14. Ranking and level are independent failures.)

**discrimeval — the oracle works when asked comparatively.** Predicting the gap directly with the
order swap, the oracle *beats* xmm at the macro (+0.57 vs +0.50) and on deepseek (+0.58 vs +0.51),
gemini (+0.64 vs +0.44), gpt-nano (+0.50 vs +0.33) and qwen (+0.54 vs +0.49); it ties llama-3.3
(+0.57) and trails only maverick (+0.61 vs +0.65). Position bias mattered most for llama-3.3
(+0.42 single-order → +0.57 swapped) and maverick (+0.51 → +0.61). So the model *can* predict its
own discrimination direction and relative magnitude — the earlier "loses everywhere" reading was
an artifact of the wrong elicitation grain, not a property of self-prediction on this eval.

## Result 2 — self-specificity: real on the direct-gap eval, modest on capability

Identity transfer scores model A's predictions against model B's targets; diag ≫ off-diag means
the predictions are about *A itself*, not a generic prior. The earlier abstract methods showed
diag ≈ off-diag (no self-specificity). The oracle is different, most clearly on discrimeval:

| eval | transfer diag (self) | transfer off-diag (other models) | partial r given xmm (range) |
|---|---|---|---|
| sycophancy_pushback | +0.31 | +0.27 | −0.24 … +0.46 (mixed) |
| capability_mmlu | +0.32 | +0.16 | −0.05 … +0.53 (mixed) |
| discrimeval (direct gap, swapped) | **+0.57** | +0.43 | **+0.30 … +0.46 (positive 6/6)** |

On **discrimeval** the direct-gap oracle is both self-specific (diag +0.57 > off-diag +0.43) and
carries **unique signal beyond the pool prior on every model** (partial r +0.30..+0.46) — the
model's stated bias is genuinely its own, and adds to what xmm knows. On **capability** the self
score is ~2× the cross-model score (+0.32 vs +0.16) — the concrete item drags the model's own
per-item knowledge state into the estimate (plausibly it partly solves the item and experiences
the difficulty) — but the partial r is mixed, so it is not reliably additive. On **sycophancy**
self-specificity is weak (diag ≈ off-diag) and partial r swings both ways.

## Synthesis

The elicitation **grain**, not just the information, governs whether verbal self-prediction
works — and it interacts with what the eval measures:

- **Absolute per-condition rates (sycophancy, capability):** even with the verbatim item and
  explicit reasoning, the oracle loses to the cross-model difficulty prior and stays far below
  the noise ceiling (0.7–0.9). The verbal channel for *"how often, in absolute terms, will I do
  X here"* saturates low; the headroom is reachable only behaviorally (cf. the own-behavior-proxy
  and excluded-probe results above). Miscalibration of the *level* is a second, independent
  failure.
- **A direct signed comparison (discrimeval gap):** the model predicts its own behavior *better*
  than the pool prior (+0.57 vs +0.50) and adds unique, self-specific signal on top. Asking *"how
  differently will I treat B vs A"* is a question the model can answer; asking it to produce two
  absolute rates and subtract is not (differencing destroys the gap in noise). This is the
  constructive half of the finding and the clearest lever for method design.

Directions this opens:
- **Elicit comparatively / at the scored grain, and cancel position bias.** Two re-framings of the
  *same* information — predicting the gap directly (+0.32 → +0.51) and swapping applicant order
  (+0.51 → +0.57) — are the largest single moves in this investigation. Where a target is a
  contrast, ask for the contrast, in both orders.
- **xmm + oracle ensemble (`oracle_xmm`, implemented — see the ensemble section below):**
  variance-matched averaging beats *both* components wherever both carry signal (e.g. sycophancy
  qwen +0.77, capability llama-3.3 +0.75) but a fixed equal weight dilutes xmm where the oracle is
  near-dead, so the macro only ties xmm. The follow-up is reliability-gated weighting.
- **Grounded comparative prompts for sycophancy** (show two subjects' items, ask which you flip
  on more — a pairwise/gap framing rather than an absolute rate) are worth trying, since the
  discrimeval result says the gap framing is what unlocks self-specific signal.
- **The oracle is the honest ceiling row for the leaderboard**: "verbal self-prediction, maximal
  information", read against xmm and the noise ceiling per eval.

---

# The oracle_xmm ensemble (2026-07-07)

`oracle_xmm` (`behavior_prediction/methods/trained.py::OracleXmm`) tests the stacking direction
above: it reads the on-disk `cross_model_mean` and `informed_oracle` predictions, z-scores each
across the scored units (equal variance → comparable importance, as requested), and averages
them into a ranking score. Stateless (the standardization uses the prediction vectors, not
targets), keyed at the eval's scored grain (per-condition rate elsewhere; on discrimeval the
prior's per-cell rates are differenced to the contrast grain first, matching the oracle).

**Macro leaderboard (dev): a wash overall, because the oracle's value is eval-dependent.**

| eval | cross_model_mean | informed_oracle | **oracle_xmm** |
|---|---|---|---|
| sycophancy_pushback | +0.58 | +0.31 | +0.56 |
| capability_mmlu | +0.69 | +0.32 | +0.65 |
| discrimeval | +0.50 | +0.57 | +0.57 |
| macro (these 3) | +0.59 | +0.40 | **+0.59** |

Equal-weight averaging can't downweight a weak partner, so the macro just ties xmm. **But the
per-model detail is bimodal and is the real result:** the ensemble *exceeds both components*
wherever both individually carry signal, and only loses where one component is near-dead.

- **Synergy (ensemble > both components):** sycophancy qwen **+0.77** (xmm +0.71, oracle +0.67),
  sycophancy gpt-nano **+0.72** (+0.64, +0.47), sycophancy deepseek **+0.70** (+0.66, +0.35),
  capability llama-3.3 **+0.75** (+0.64, +0.41), discrimeval maverick **+0.67** (+0.65, +0.61),
  discrimeval llama-3.3 **+0.60** (+0.57, +0.57). Two partly-independent predictors beat either.
- **Dilution (ensemble < best):** wherever the oracle is near-zero the equal weight drags xmm
  down — capability deepseek +0.50 (xmm +0.70, oracle −0.06), sycophancy gemini +0.30 (+0.49,
  +0.00). This is the equal-weight scheme's structural limit, not a signal problem.

Mean(ensemble − best single component): discrimeval −0.01 (wins 3/6), sycophancy −0.02 (3/6),
capability −0.04 (1/6). So a *fixed* variance-matched mean is the wrong aggregator: it should be
**weighted by each component's reliability**, which would keep the synergy wins while dropping the
near-dead oracle on the rate evals. That is exactly what the trained sibling does.

## The trained ensemble — `oracle_xmm_learned` (2026-07-07)

`oracle_xmm_learned` (trained; `methods/trained.py::OracleXmmLearned`) learns a **convex** weight
`w·z_xmm + (1−w)·z_oracle` per CV fold, where `w` is xmm's share of clipped fit-fold reliability
`r⁺_xmm/(r⁺_xmm+r⁺_oracle)`, shrunk toward 0.5 by a swept `shrink` (0.5 or 1.0). No model calls —
it reads the two component files and is CV'd out-of-fold; only the weight uses fit-fold labels.

Two design points were load-bearing (both discovered empirically):
- **Convex / sum-to-one weights.** A first attempt used the raw fit-fold correlations as
  coefficients; because their magnitude varies fold to fold, the pooled out-of-fold vector became
  a patchwork of differently-scaled blocks and the correlation *collapsed* (discrimeval llama-3.3
  +0.60 → +0.29). Normalizing to a convex combination puts every fold on one scale and fixed it.
- **Standardize globally, weight locally.** z-scoring is label-free, so it is done over the full
  prediction set (leak-free) to keep folds comparable; only the weight is fit per fold.

Result — **it is the top method on all three evals the oracle covers, beating both the equal-weight
ensemble and either component:**

| eval | cross_model_mean | informed_oracle | oracle_xmm (equal) | **oracle_xmm_learned** |
|---|---|---|---|---|
| sycophancy_pushback | +0.58 | +0.31 | +0.56 | **+0.59** |
| capability_mmlu | +0.69 | +0.32 | +0.65 | **+0.70** |
| discrimeval | +0.50 | +0.57 | +0.57 | **+0.57** |
| macro (these 3) | +0.59 | +0.40 | +0.59 | **+0.62** |

The learning earns its keep exactly where the equal weight failed: **capability deepseek +0.50 →
+0.70** (the near-dead oracle, r=−0.06, is weighted to ~0 instead of diluting xmm's +0.70), and it
never blows up where the components are comparable (discrimeval llama-3.3 back to +0.60). The
tuner picks `shrink=1.0` on sycophancy/capability (where one component is often much stronger, so
full reliability weighting pays) and `shrink=0.5` on discrimeval (where the two are close, so more
shrinkage toward equal is safer). This is the reliability-gated aggregator the equal-weight
baseline motivated, and it delivers: **+0.62 macro, the best single number in the study**, plus
the top per-cell results (sycophancy qwen +0.75, capability llama-3.3 +0.77, capability deepseek
+0.70).

---

# Selection pool widened to 6 models (2026-07-09)

`methods.yaml: selection_pool` aggregated over 4 models (llama-3.3-70b, llama-4-maverick,
deepseek-v4-flash-low, qwen3.7-plus-low) while `bp-evaluate` scored over 6. The two evaluated-but-
not-pooled models — gemini-3.1-flash-lite-low and gpt-5.4-nano-low — have **full prediction coverage
on all four evals**, so excluding them from tuning was an accident of when they were added, not a
design choice. Widening the pool to those six makes `bp-tune`'s selection criterion and
`bp-evaluate`'s reported score aggregate over the same models. (gpt-5.5-{low,off} remain out: still
partial coverage.)

Effect: `informed_oracle` on PB goes +0.21 → +0.351 as the tuned score, matching what evaluate
reports. Three methods reselect a different best setting:

| eval | method | was | now |
|---|---|---|---|
| capability_mmlu | `self_report` | `self_report` | `self_report-honest` |
| discrimeval | `oracle_xmm_learned` | `oracle_xmm_learned` | `oracle_xmm_learned-full` |
| propensitybench | `list_experiment` | `list_experiment-n6` | `list_experiment` |

Everything else keeps its setting; the cached `agg_r` values shift by ≲0.1 as expected when two
models join the macro-mean.

**Two artifacts don't survive a naive regeneration** (see also `bp-tune --methods X`, which rewrites
`tuning.json` with only X):

- `behavioral_sampling` predictions are gitignored (`results/**/predictions/behavioral_sampling*.json`
  — bulky, regenerable), so any `bp-tune` on a clean clone nulls its cached score. It has a single
  setting, so selection is unaffected and restoring the committed entry is safe; the `agg_r` carried
  in `tuning.json` for it is still the **old 4-model** number (3-model on discrimeval). Regenerate
  its predictions to refresh it.
- `bp-evaluate` therefore also **drops `behavioral_sampling` entirely** from
  `paper/generated/results.tex`. Regenerating the paper tables requires a machine that has those
  predictions on disk; a clean clone silently loses the `\resultBehavioralsampling*` macros. The
  pending update there is `\resultInformedoracleN` 18 → 24 (PB's six cells), with the mean moving
  +0.40 → +0.39.
