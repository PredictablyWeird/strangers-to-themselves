# Self-prediction finetuning corpus v1 — draft spec (2026-07-09)

> **Design note.** Records a design decision, not results; it predates the final runs.

Goal: a finetuning corpus whose task matches *our* evaluation (predict your own behavioral **rate**
over a family of situations), built entirely from Anthropic's model-written evals
([anthropics/evals](https://github.com/anthropics/evals)) so that none of our four scored evals is
touched. No train split, no per-fold finetune, one finetune per model.

Background and the probes that motivate every choice here: `docs/rate-self-prediction-finetuning.md`.

## The design in one paragraph

A **condition** is a family of *distinct* items sharing a describable label (Probe 2: our evals'
rate variance is across-item, not within-item — 98% of DiscrimEval items are individually
deterministic). Conditions come from crossing three axes over the suite: **category** (what behavior
is being probed) × **topic cluster** (what the items are about) × **polarity** (whether the
behavior-matching answer is the affirmative one). The target is the model's own measured rate on
that family, plus the polarity **gap** as a contrast target.

## Why polarity is the load-bearing axis

Every category ships perfectly balanced: 500 items where the matching answer is `Yes` / `(A)` and
500 where it is `No` / `(B)`. Measured on 150 categories (40 items each, from Probe 4's data):

| quantity | Llama | Qwen | r(Llama, Qwen) across categories |
|---|---|---|---|
| raw rate | sd 0.253 | sd 0.278 | **+0.836** |
| polarity gap (POS − NEG) | mean −0.150, sd 0.290 | mean −0.040, sd 0.276 | **−0.066** |

The raw rate is mostly shared across models — it encodes how agreeable a behavior is.
The **gap is not**: it is that model's own acquiescence/position bias on that family, and the two
models' gaps are uncorrelated. Mean |gap difference| is 0.283.

Reliability check, because a near-zero correlation can also mean "all noise": at 20 items per half,
SE(gap) = 0.158, and the observed gap sd is 0.290, so the *true* gap sd is
√(0.290² − 0.158²) ≈ 0.24 and reliability ≈ 0.70. Disattenuating, the true cross-model correlation
is ≈ −0.09. The gap is real signal, and it is model-specific.

This mirrors `docs/method-diagnostics.md`: on DiscrimEval the oracle only beats `cross_model_mean`
when asked for the demographic **gap** directly rather than differencing two absolute rates.

## Inventory

| | |
|---|---|
| categories | 151 (16 `advanced-ai-risk/human_generated_evals` + 135 `persona`) |
| items per category | median 1000, min 300 |
| **total items** | **141,320** |
| answer format | 16 categories `(A)`/`(B)`, 135 `Yes`/`No` |
| polarity balance | exactly 50/50 within every category |

## Counts: prompts needed, and samples produced

**Elicitation = exactly one greedy call per item, per model.** Items are individually
near-deterministic (Probe 2), so no resampling: a condition's rate is the fraction of its items
answered with `answer_matching_behavior`. Both training objectives are derived from this single
pass — the item-level targets *are* the calls, and the condition-level rates are aggregates of them.

| stage | count (per model) |
|---|---|
| elicitation calls, train split (80% of each category) | **113,056** |
| elicitation calls, held-out split (20%) | 28,264 |
| **total elicitation calls** | **141,320** (one pass over the suite) |

Cost: prompts run ~80 tokens with an 8-token completion, so ≈10M tokens ≈ **$9** on Together's
serverless Llama-70B, a few hours wall-clock at 24 workers. Must be repeated per model, since
self-prediction requires the model's *own* behavior.

Conditions and the SFT pairs they yield, from that same pass:

| | formula | count |
|---|---|---|
| train conditions | 151 categories × 10 topic clusters × 2 polarities | **3,020** |
| items per condition | 113,056 / 3,020 | ≈ 37 (rate SE ≈ 0.082) |
| **item-level pairs** (matches `informed_oracle`) | one per train item | **113,056** |
| **condition-rate pairs** (matches `self_report`) | one per condition | **3,020** |
| **contrast pairs** (matches DiscrimEval's oracle grain) | one per category × cluster | **1,510** |
| **total available SFT pairs** | | **117,586** |

Binder trained on 30k. The recommended v1 mix keeps that scale and upweights the objectives that
match our headline methods:

| objective | pairs | note |
|---|---|---|
| item-level | 25,000 | subsampled, stratified over category × polarity |
| condition-rate | 3,020 | all |
| contrast | 1,510 | all |
| Binder property pairs (retain the general skill) | 5,000 | from the existing `sft_train.jsonl`, **excluding `mmlu_non_cot`** |
| **total** | **≈ 34,530** | |

## The three prompt templates

Built with the repo's own builders wherever possible, so train format == test format.

1. **item-level** — `spec.oracle_items` shape: the verbatim item plus a protocol description, asking
   the model to predict its own answer. Target: the letter/word it actually produced.
2. **condition-rate** — `elicitation.self_report_prompt` shape: "Out of 100 questions like
   *&lt;cluster description&gt;* where the &lt;behavior&gt; answer is the affirmative one, in how many
   would you choose it?" Target: the measured rate as an integer 0–100.
3. **contrast** — "…how much more often when the matching answer is affirmative than when it is
   negative?" Target: the signed gap, in points.

The cluster description is the only authored text: cluster each category's items (embed, k=10) and
have an LLM name each cluster. 1,510 labels total, generated once.

## Splits and held-out structure

Nothing about `splits.json` changes — all four scored evals stay fully held out. Within this corpus,
report three nested generalization rings:

- **new items, seen condition** — the 20% held-out split (28,264 items)
- **new condition, seen category** — hold out 2 of the 10 topic clusters per category
- **new category** — hold out 30 of the 151 categories entirely

The third ring is the claim worth making, and it is what predicts transfer to our evals.

## v0 headroom result (RUN 2026-07-09, `scripts/selfpred_headroom.py`)

Verdict: **build v1, and make the gap the primary target.** Base models predict the shared,
content-driven part of a condition worse than a cross-model prior does, and their access to their
*own* bias is weak (Llama) or partial (Qwen). Correlations over 150 categories, greedy self-report,
one call per (category, polarity) half; measured rates reuse the Probe 4 answers (no new elicitation).

**Gap (the model-specific quantity), predicted vs each model's actual:**

| predictor | vs Llama actual | vs Qwen actual |
|---|---|---|
| Llama self-report | **+0.273** (self) | +0.227 |
| Qwen self-report | −0.084 | **+0.615** (self) |
| cross-model prior (other model's actual) | −0.066 | −0.066 |

**Raw rate (the shared quantity):**

| predictor | vs Llama actual | vs Qwen actual |
|---|---|---|
| Llama self-report | +0.706 (self) | +0.605 |
| Qwen self-report | +0.628 | +0.737 (self) |
| cross-model prior (other model's actual) | **+0.854** | **+0.854** |

Four things fall out, and the built-in cross-prediction control makes them interpretable:

1. **On the raw rate, self-report loses to the cross-model prior** (0.71/0.74 vs 0.85). Verbal
   self-prediction adds nothing model-specific there — it is estimating how agreeable a behavior is,
   which the pool already knows. Same conclusion as `docs/method-diagnostics.md` on the absolute-rate
   evals.
2. **On the gap, self-report massively beats the prior** (+0.273 / +0.615 vs −0.066). The gap is
   where verbal self-prediction carries genuinely self-specific signal — mirroring DiscrimEval, where
   the oracle only beats `cross_model_mean` in the contrast grain.
3. **Qwen's gap prediction is strongly self-specific; Llama's is not.** Qwen scores +0.615 against
   its own gap and −0.084 against Qwen→Llama; Llama scores +0.273 against its own and +0.227 against
   Qwen's — barely a self-specific margin. Llama is largely predicting *a* model's bias, not its own.
4. **Calibration is badly off even where correlation is decent.** Llama's predicted gap has mean
   −0.627 against a measured −0.150 (MAE 0.539) and sd 0.481 against a measured 0.289: it thinks it
   says "Yes" far less often than it does, and it over-disperses. Qwen: MAE 0.288.

Against the pre-registered thresholds (ceiling for the gap is r ≈ 0.84, since reliability at 20
items/half is ≈ 0.70), Llama at +0.273 sits squarely in the "real headroom" band and Qwen at +0.615
just past it. Both have large calibration errors that a rate-valued finetune directly targets.

**Item-level is confirmed degenerate**, as predicted: asked to recall the answer it gave, Llama
scores 89.6% and Qwen 93.5% (n≈300). Behavior *is* the answer on this suite, so the item-level
objective teaches almost nothing. **Drop it from the v1 mix** and reallocate to condition-rate and
contrast pairs — which also removes the corpus's main cost driver.

## Probe 5 — does the topic-cluster axis carry signal? (No.)

The cluster axis was supposed to multiply conditions 10×. Tested on the existing screen data (no new
calls) by permuting item→cluster assignment *within each polarity half* of a category, 400
permutations, and comparing the observed between-cluster variance of the rate and of the gap against
that null:

| model | target | observed | permuted null | perm p | verdict |
|---|---|---|---|---|---|
| Llama | rate | 7.07 | 6.77 | 0.115 | not distinguishable from noise |
| Llama | gap | 10.99 | 9.89 | 0.095 | not distinguishable from noise |
| Qwen | rate | 7.58 | 6.89 | **0.005** | real, but small |
| Qwen | gap | 8.92 | 9.12 | 0.574 | not distinguishable from noise |

Effect sizes, as a true between-cluster sd: rate 0.046 (Llama) / 0.068 (Qwen) against a
between-*category* sd of 0.253 / 0.278; gap 0.093 (Llama) / 0.000 (Qwen) against a between-category
gap sd of 0.29. Even taking Llama's gap estimate at face value, 0.093 sits *below* the 0.100 standard
error a 50-item condition gives it — the labels would be mostly noise.

Caveat on power: the screen samples only ~4 items per cluster, so "not distinguishable" is not "zero".
But the effect-size ceiling is what decides it, and that is small regardless.

A first, wrong analysis of this gave z = +361 and had to be thrown out: with 2 items per half the
inverse-variance weights blow up when a cell is unanimous (var → 0, clamped, weight → 1000). The
permutation test needs no variance model and is the right tool.

**Conclusion: drop the topic-cluster axis.** All ten clusters in a category probe one behavior; the
category *is* the topic. Conditions are **category × polarity = 302**, yielding 302 rate pairs and
151 gap pairs. That is far too small a corpus, so the multiplier has to come from somewhere real:

- **More items per condition** (use all 1000, i.e. 500/half → gap SE 0.032 instead of 0.100). Free
  precision, no new conditions.
- **A framing axis** — system persona, evaluation-awareness, stakes. Plausibly shifts acquiescence,
  is describable, and gives a genuinely different gap per framing. Costs one elicitation pass per
  framing level (~141k calls each, ~$11).
- **More categories**: `advanced-ai-risk/lm_generated_evals` adds 17, plus `sycophancy` and
  `winogenerated`.
- **Cross-category topic families**: cluster all 141,320 items *globally*, so a family mixes
  behaviors and its rate genuinely varies. Testable on the existing screen data before spending.

The framing axis is the only one that clearly multiplies *and* changes the gap; I would test it on
one category before committing to a full pass.

## Revised v1 mix (item-level dropped)

| objective | pairs | note |
|---|---|---|
| condition-rate | 3,020 | all |
| contrast (gap) | 1,510 | all — **the primary target** |
| Binder property pairs | 5,000 | retain general skill; exclude `mmlu_non_cot` |
| **total** | **≈ 9,530** | |

To reach Binder's 30k the conditions must be multiplied, not the items: more topic clusters (k=20 →
6,040 rate + 3,020 contrast pairs) and/or a framing axis. Elicitation is unchanged at one greedy
call per item, but only the items actually needed to estimate each condition's rate — at 37
items/condition and 9,060 conditions that is ~335k calls per model, so k=20 plus a 3-level framing
axis is the point where cost starts to bite (~$25/model, still cheap).

## Original v0 plan (~6,300 calls per model, no finetuning)

By design, the corpus is only worth building if the base model does **not** already
predict its own answers. We already have measured behavior for 6,040 items per model from Probe 4,
so the check costs one self-prediction pass over the same items:

| | calls per model |
|---|---|
| item-level self-prediction on the Probe 4 items | 6,040 |
| condition-rate self-report, 151 categories × 2 polarities | 302 |

Report: item-level self-prediction accuracy against a majority-class baseline; condition-rate
correlation against measured rates; and the same for the polarity gap. **If baseline item accuracy
is already near ceiling, or condition-rate r is already high, there is no headroom and v1 should not
be built.** If the gap is predicted much worse than the rate — which Probe 4 predicts, since the gap
is the model-specific part — that quantifies exactly what the finetune has to teach.

## Controls (unchanged, from the parent doc)

Cross-prediction (labels from a different model), shuffled labels, self-fulfilment (finetuning on
"you do X 37% of the time" can move behavior toward 37%), and constant-output collapse (stratify to a
flat rate histogram; watch held-out `sd(pred)`).

One caveat to state in any writeup: because r(Llama, Qwen) on raw rates is +0.836, a model that
learns only the rate is largely reproducing a pool prior. The **gap** is the part only the model
itself can know, which is why it gets its own objective here rather than being left implicit.
