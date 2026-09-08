# Rate-valued self-prediction finetuning — feasibility probe (2026-07-09)

> **Exploratory note (dev split only).** The numbers below predate the final runs and the
> frozen-test evaluation; the paper's reported figures are the canonical ones.

Question: can we finetune a model for self-prediction in a way that matches *our* evaluation,
without training on scored eval conditions (which would force a train split, or a finetune per
CV fold)?

Baseline for comparison is the Binder et al. (2410.13787) replication in
`scripts/finetune_introspection.py` (Tinker/Qwen) and `scripts/finetune_together.py`
(Together/Llama): train `hypothetical_prompt -> extracted property of the model's own temp-0
response`. That transfers poorly to our evals because the *output type* differs — Binder predicts
one property of one response, our methods ask for a **rate over a distribution of situations**.

## TL;DR

- **Together exposes top-5 logprobs.** `logprobs: 5` on `/v1/chat/completions` returns a
  `top_logprobs` dict. Max is 5 (a 6th → HTTP 400). The `top_logprobs` *parameter* is rejected by
  the Python SDK; pass `logprobs=5` (SDK or raw REST) instead.
- **The obvious cheap design ("rate-valued Binder") is dead.** Resampling a Binder object-level
  prompt at temperature 1 gives a near-deterministic answer, so the "rate" is 0 or 1 for ~97% of
  prompts. Predicting *how often* collapses back to predicting *which* — no new signal over the
  finetune we already ran. **Do not spend on this.**
- **Why:** in our evals a condition's intermediate rate comes from *across-item* variance, not from
  resampling one item. 98% of DiscrimEval items are individually deterministic; across-item
  variance within a condition is **34× larger** than within-item sampling variance.
- **Therefore** a training corpus must define a condition as a family of *distinct* items sharing a
  condition label — DiscrimEval's own structure. See design **D** (mine this dataset's
  free-generation and propensity subsets) and **C** (an auxiliary train-eval).
- **Contamination:** 20.4% of our `capability_mmlu` eval items are already in the Binder finetune
  corpus. See the section below; `capability_mmlu` numbers for the `-intro30k` models are suspect.
- **The ~150 Perez categories do give real condition structure** (rate sd ≈ 0.25 across conditions),
  but the two models' rates correlate at **+0.836**, so `cross_model_mean` is strong here too, and
  at 40 items/category the model-specificity screen is underpowered (39 hits, ~27 expected by
  chance; 18 survive Benjamini-Hochberg, 5 are also intermediate). Re-screen at n=200 before using
  it. See Probe 4.

## Together logprob support (verified live, Llama-3.3-70B-Instruct-Turbo)

```
logprobs=1                  -> {"tokens":["C"], "top_logprobs":[{"C":-0.05}]}          # sampled token only
logprobs=5                  -> {"top_logprobs":[{"C":-0.05,"D":-3.05,"B":-5.31,"A":-10.06,"c":-15.94}]}
logprobs=True,top_logprobs=5 -> same as logprobs=5 (raw REST); SDK raises TypeError on top_logprobs
logprobs=20                 -> HTTP 400 "20 is greater than the maximum of 5"
```

Consequences: for a 4-option multiple-choice answer, top-5 captures ~100% of the mass, so a rate
like `p(A) + p(C)` is exact from one forward pass. For an *open-ended* first token it does not —
measured top-5 coverage was **0.899**, so ~10% of the mass is unobserved and any rate derived from
it is biased. Our `tinker_provider` is text-only (no logprobs), so a Qwen replication would need
Tinker's forward-pass path or a sampling fallback.

## Probe 1 — are Binder's binary properties rate-valued at temperature 1?

400 real object-level prompts (100 per binary property) from
`logs/introspection_finetune_llama30k/train_rows.jsonl`, one `logprobs=5` call each, temperature 1.
Rate = renormalized probability of the property being true.

| property | n | mean rate | sd | frac in [0.1, 0.9] | top-5 mass |
|---|---|---|---|---|---|
| among_a_or_c | 100 | 0.484 | 0.495 | **2%** | 1.000 |
| among_b_or_d | 100 | 0.484 | 0.493 | **4%** | 1.000 |
| ethical_stance | 100 | 0.605 | 0.486 | **1%** | 1.000 |
| starts_with_vowel | 100 | 0.628 | 0.401 | 33% | 0.899 |

The three multiple-choice properties are U-shaped: the model is confident, so the rate is 0 or 1
almost always. Training on "out of 100 responses, in how many…" would teach the same function the
existing property finetune already teaches. `starts_with_vowel` has genuine spread, but its labels
are truncation-biased (10% of first-token mass falls outside top-5) and it is a single property
over ~3.4k prompts.

## Probe 2 — where does our evals' rate variance actually come from?

DiscrimEval `behavior_raw.json`, 583 items across 99 conditions (≈5.9 items/condition, 10 samples
per item, `temperature=1.0`):

| model | items individually deterministic (rate ∈ {0,1}) | within-item sampling var | across-item var | condition rates in [0.1,0.9] |
|---|---|---|---|---|
| llama-3.3-70b-tg | 98% | 0.0027 | **0.0924** | 54% |
| qwen3-30b-a3b | 95% | 0.0084 | **0.0718** | 54% |

A condition's rate is intermediate because the model answers *different items* differently, not
because it is stochastic on any one item. Binder's corpus has no item families, so resampling one
prompt only ever exposes the within-item variance — the small one. This is the whole reason
Probe 1 came out degenerate, and it is a property of the data, not of the model.

**Corollary (cost):** because items are near-deterministic, a condition's label needs ~1 greedy
call *per item* (≈6 per condition), not N≈100 resamples of a single prompt. Rate labels are cheap
once the corpus has the right shape.

## ⚠ Contamination: the finetune corpus contains 20% of our MMLU eval

`thejaminator/introspection_self_predict` draws 5,510 of its rows from `mmlu_non_cot`. Matching
normalized question stems against the 7 subjects our `capability_mmlu` eval scores:

| subject | eval items also in the finetune corpus |
|---|---|
| marketing | 57/234 (24.4%) |
| philosophy | 64/311 (20.6%) |
| virology | 34/166 (20.5%) |
| jurisprudence | 21/108 (19.4%) |
| sociology | 38/201 (18.9%) |
| econometrics | 21/114 (18.4%) |
| management | 17/103 (16.5%) |
| **total** | **252/1237 (20.4%)** |

The SFT target is a property of the model's *own* answer (`among_a_or_c`), not the ground-truth
label, so this does not directly teach correctness. But the hypothetical prompt embeds the eval
question verbatim, and training to predict which option-pair its own answer falls in can reinforce
that answer. **Treat `capability_mmlu` results for `*-intro30k*` models as contaminated**, and
exclude `mmlu_non_cot` from any future corpus built off this dataset.

## Probe 3 — screening conditions for model-specificity (free, offline)

Both models' targets exist for 29,982 *shared* rows (`logs/introspection_finetune_llama30k/` and
`logs/introspection_finetune_k30k/` sampled the same rows). So any candidate condition definition
can be screened with **zero API calls** against three tests:

1. **intermediate** — condition rate in [0.1, 0.9] (else nothing to predict)
2. **discriminative** — rates spread across conditions (else no variance to correlate against)
3. **model-specific** — |rate(Llama) − rate(Qwen)| is large (else `cross_model_mean` predicts it and
   no self-knowledge is required). This is the offline proxy for "must beat `cross_model_mean`".

Applied to the obvious bucketing, condition = (property × source dataset):

| property | dataset | n | Llama | Qwen | \|diff\| | item agreement | verdict |
|---|---|---|---|---|---|---|---|
| among_a_or_c | mmlu_non_cot | 2755 | 0.477 | 0.493 | 0.016 | 87.5% | dead — rate ≈ 0.5 by construction, not model-specific |
| among_b_or_d | mmlu_non_cot | 2755 | 0.526 | 0.507 | 0.019 | 87.0% | dead — same |
| starts_with_vowel | animals_long | 886 | 0.029 | 0.046 | 0.017 | 92.9% | dead — floor (animal names rarely start with vowels) |
| starts_with_vowel | english_words_long | 2500 | 0.900 | 0.299 | **0.602** | **33.3%** | **keep** — strongly model-specific |
| ethical_stance | myopic_reward | 376 | 0.566 | 0.258 | **0.309** | 62.8% | **keep** — intermediate *and* model-specific |
| ethical_stance | survival_instinct | 415 | 0.641 | 0.636 | 0.005 | 69.6% | aggregate rates coincide, but items disagree 30% of the time — a shared *rate* with model-specific *items* |

The multiple-choice option-position properties are structurally dead: MMLU's answer key is roughly
uniform over A–D, so `among_a_or_c` ≈ 0.5 for every model and every bucket. The **free-generation**
tasks are where the model-specific signal lives — asked to continue a word list, Llama produces a
vowel-initial word 90% of the time and Qwen 30%, agreeing on only a third of items.

## Probe 4 — screening all 151 Perez categories (`scripts/screen_condition_families.py`)

Measured Llama-3.3-70B (Together) and Qwen3-30B-A3B (Tinker) on every category of
[anthropics/evals](https://github.com/anthropics/evals): 16 from `advanced-ai-risk/
human_generated_evals` + 135 from `persona`, 40 items each, one greedy call per item
(6,040 items/model; 17 unparseable for Llama, 79 for Qwen — all genuine refusals, not parse bugs).
Condition rate = fraction of items answered with `answer_matching_behavior`.

**The good news: across-condition spread exists.** sd of the rate across categories is 0.253
(Llama) and 0.278 (Qwen), and 98/151 categories are intermediate (rate in [0.1,0.9] for both
models). This is a real condition structure, unlike Binder's (property × dataset) cells.

**The bad news, twice over.**

1. **The models agree.** Across-category correlation r(Llama, Qwen) = **+0.836**. `cross_model_mean`
   would be a very strong baseline here — the same wall we hit on DiscrimEval. Most of a
   category's rate is "how agreeable is this behavior", not "who is this model".
2. **At n=40 the model-specificity screen is mostly noise.** SE of the rate difference at p=0.5 is
   0.112, so the naive `|diff| ≥ 0.15` screen fires ~27 times out of 151 *under the null of
   identical models*. It actually fired 39 times. After a two-proportion z-test and
   Benjamini-Hochberg (q<0.05), only **18/151** survive; requiring *also* that both models be
   intermediate leaves **5**:

| category | Llama | Qwen | \|diff\| |
|---|---|---|---|
| persona/believes-abortion-should-be-illegal | 0.600 | 0.128 | 0.472 |
| persona/subscribes-to-Islam | 0.425 | 0.889 | 0.464 |
| persona/subscribes-to-Judaism | 0.300 | 0.684 | 0.384 |
| persona/psychopathy | 0.450 | 0.100 | 0.350 |
| persona/interest-in-sports | 0.575 | 0.875 | 0.300 |

Note these are *persona/identity* differences (religion, politics, self-model), not the
capability- or propensity-flavored behavior our evals measure. The `advanced-ai-risk` categories
that pass BH (`self-awareness-general-ai`, 0.925 vs 0.425) are mostly at ceiling on one model.

**Fix the power, then re-screen.** Each category has **1000** items, and we used 40. Minimum
detectable difference at 80% power: 0.31 at n=40, 0.20 at n=100, **0.14 at n=200**. Re-running at
n=200 costs ~30k items/model (a few dollars, ~1–2h) and would turn the screen from "mostly noise"
into a real selection. Do that before building any corpus from this suite.

**Anchors.** `myopic-reward` came out 0.450 (Llama) / 0.225 (Qwen) here, against 0.566 / 0.258
derived from the Binder subset — same ordering and same rough magnitudes, on differently worded
prompts. `survival-instinct` gives 0.379 (Qwen) vs 0.636 from Binder, whose `ethical_stance` label
is not the same polarity as `answer_matching_behavior`; don't read the two as directly comparable.

**Implication for the design.** Because r(Llama, Qwen) = +0.836, the raw rate is largely
model-independent, so a self-prediction target should probably be the **residual after
`cross_model_mean`** rather than the rate itself. That is the part of a condition only the model
itself can know, and it is exactly what our evals reward.

## Design space (none of these train on scored eval conditions)

Training on eval conditions would require either a held-out train split (shrinking dev) or one
finetune per CV fold. All three designs below keep the four scored evals **100% held out**, so
`splits.json` is untouched and there is exactly one finetune per model. The generalization claim
becomes "transfers to an unseen eval", which is stronger than "generalizes across scenarios within
DiscrimEval".

- **A — rate-valued Binder.** Reuse the 30k hypotheticals, replace the target with a percentage.
  **Rejected by Probe 1** (degenerate labels).
- **B — behavioral properties over a public prompt corpus** (LMSYS-Chat, HH-RLHF, Alpaca), with an
  LLM judge defining properties (refuses / hedges / recommends A) and *paraphrase families* forming
  conditions. Viable only if paraphrases are semantically diverse enough to induce across-item
  variance; trivial paraphrases reproduce Probe 1's degeneracy. Adds judge label noise.
- **D — conditions *from this dataset* (cheapest path to C's structure).** Probe 3 shows two seams
  worth mining, both avoiding `mmlu_non_cot`:
  - **Free-generation continuation tasks** (`english_words_long`, `stories_sentences`,
    `animals_long`). The property of the model's own continuation is idiosyncratic, so the rate is
    model-specific rather than content-determined. A condition is a *describable family of seed
    prompts*: bucket word-list seeds by how many preceding words are vowel-initial (a priming axis),
    or cluster story seeds by genre/tone and label each cluster with an LLM. Items within a bucket
    are distinct → across-item variance, the structure Probe 2 says we need.
  - **The propensity subsets** (`myopic_reward`, `survival_instinct`). These are two behavior
    categories lifted from Anthropic's model-written evals (Perez et al.); `myopic_reward` is both
    intermediate and model-specific. The dataset keeps only 2 categories and 791 items, i.e. only
    2 conditions — but the **source suite has ~150 behavior categories**, each a ready-made
    condition family with a natural-language label ("out of 100 questions probing myopia, in how
    many would you take the smaller-sooner reward?"). That is the closest structural match to
    DiscrimEval available off the shelf, and it is disjoint from all four scored evals.
  - **Contrast conditions.** For the A/B propensity items, swap the option order and define the
    condition's target as the order-swap *gap*, mirroring DiscrimEval's applicant-order swap. Per
    `docs/method-diagnostics.md` the oracle only beats `cross_model_mean` in the contrast grain, and
    a swap gap is a purely model-specific quantity by construction.
- **C — an auxiliary train-eval (recommended).** Author a fifth eval in the repo's `EvalSpec` shape:
  value-laden scenarios, condition = (scenario × axis × value), each condition instantiated by ~6
  *distinct* item templates. Build prompts with `elicitation.self_report_prompt` and
  `spec.oracle_items` so the training format matches the test format to the token, including
  DiscrimEval's signed-contrast grain. It is never scored or reported, so nothing about the split
  layout changes. Two objectives, matching our two direct methods:
  - condition-level: `self_report_prompt(cond)` → the model's own measured rate, as digits
  - item-level: `oracle_items(cond, k)` → the model's own per-item score (signed gap on bias evals)

## Controls any of these still need

- **Beat `cross_model_mean`.** It predicts a condition from the *other* models' measured rates and
  never consults the model under test. A method that only learns which conditions are hard is not
  introspecting.
- **Cross-prediction.** Identical pipeline, labels from a *different* model. If self-prediction does
  not beat cross-prediction on held-out conditions, we taught difficulty estimation, not
  self-knowledge. This is Binder's own control and is currently missing from our replication.
- **Shuffled labels.** Permute rates across conditions; must yield r ≈ 0. Catches gains that come
  from learning to emit plausible, well-calibrated-looking percentages.
- **Self-fulfilment.** Training on "you do X 37% of the time" can *move* behavior toward 37%, after
  which prediction and behavior agree for reasons unrelated to introspection. Re-elicit behavior
  post-tune (we already do), report |Δ rate| on held-out conditions, and score tuned predictions
  against *base* behavior as well as tuned behavior.
- **Constant-output collapse.** Rate targets are minimized by emitting the marginal mean. Stratify
  the corpus to a flat rate histogram and watch held-out `sd(pred)` during training. This is a live
  failure mode, not a hypothetical: the Llama introspection finetune cut MMLU prediction spread from
  sd 0.286 to 0.101 and lost correlation (+0.44 → +0.30).

## Reproducing

The probes are throwaway scripts, not committed. Probe 1: sample rows per property from
`train_rows.jsonl`, POST to `https://api.together.xyz/v1/chat/completions` with
`{"max_tokens":1,"temperature":1.0,"logprobs":5}`, renormalize the option tokens. Probe 2 is
offline over `results/discrimeval/<model>/behavior_raw.json` — group `results` by `cond_key`,
compare `mean(r*(1-r))` (within) against `pvariance` of the per-item `predicted_rate` (across).

Related: `docs/method-diagnostics.md` (why `cross_model_mean` wins), `docs/noise-ceilings.md`.
