# DiscrimEval — design notes

DiscrimEval (Anthropic, [arXiv:2312.03689](https://arxiv.org/abs/2312.03689); HF
`Anthropic/discrim-eval`) asks a model to make a high-stakes *advantageous* yes/no decision about a
person (a loan, a job, parole, a medical procedure, …). **Discrimination** = the favorable ("yes")
rate depending on the person's demographics. This eval asks the project's question of that:
**can a model predict its own demographic bias?**

This file documents the design choices in `behavior_prediction/evals/discrimeval.py`. The original
adapter measured a rate per `(template, axis, value)` and differenced afterward; that grain was
mis-specified (per-template sampled rates are too noisy to difference, and holding out templates is
not a real generalization once templates act as samples). The current design fixes that.

## Grain: topic category × axis comparison

The 70 decision templates carry **no category labels** in the dataset (only `decision_question_id`,
`age` ∈ 20–100, `gender` ∈ 3, `race` ∈ 5). We **impose nine topic categories** and treat the
templates within a category as **samples**.

- **Condition (a measured cell)** = `(category, cell)` where a cell is the shared *baseline* person
  or one *comparison value* on one axis. Rate = favorable ("yes") fraction, measured by
  **sampling** yes/no (no logprobs → works for reasoning models), pooled over the category's
  templates × `--samples`.
- **Contrast (the headline quantity)** = `(category, axis, value)` bias =
  `rate(comparison cell) − rate(baseline cell)`. Each compared value is its **own** contrast —
  white-vs-Black *and* white-vs-Asian, not a single per-axis "race" number.
- **Scenario** (the `EvalSpec` grouping / split unit) = the **category**.

We never score predictions of an *individual cell rate* (e.g. "the age=80 approval rate"); the only
scored quantity is the **bias** (the gap). Every introspective method predicts that gap **directly**
via a comparative prompt; predicting per-cell rates and differencing them is left for a separate
experiment.

### Demographic comparison values

One baseline person — **40 / male / white** (shared by every contrast) — and, per axis, the
non-baseline values compared against it (`COMPARISONS` in `discrimeval.py`):

| axis | baseline | compared values |
|---|---|---|
| race | white | **Black, Asian, Hispanic, Native American** |
| gender | male | **female, non-binary** |
| age | 40 | **20, 60, 80, 100** |

### Condition & contrast key scheme

Distinct namespaces so cell-rate keys never collide with contrast keys:

- baseline cell: `"{cat}/baseline"`
- comparison cell (a rate): `"{cat}/{axis}/{value}"` — e.g. `consumer_service/race/Black`
- contrast (a bias gap): `"{cat}/{axis}/{value}-vs-{baseline}"` — e.g. `consumer_service/race/Black-vs-white`

Per category: 1 baseline + 10 comparison cells = **11 cells**; **10 contrasts** (4 race + 2 gender +
4 age). Across 9 categories: **99 measured rates → 90 bias contrasts**.

## The nine categories

53 of 70 templates are used; each category has ≥5. Source of truth is `CATEGORY` in
`discrimeval.py`; the `qid`s below mirror it (glosses abbreviated).

| Category (slug) | n | Template qids |
|---|---|---|
| Personal lending & credit (`personal_lending`) | 5 | 9, 29, 55, 65, 70 |
| Business financing (`business_financing`) | 5 | 12, 34, 87, 94, 89 |
| Consumer service & transactions (`consumer_service`) | 9 | 3, 27, 36, 37, 56, 60, 62, 90, 93 |
| Immigration & border (`immigration`) | 5 | 4, 8, 31, 61, 74 |
| Access & security control (`access_security`) | 6 | 5, 41, 52, 59, 64, 82 |
| Academic & technical review (`academic_review`) | 5 | 18, 42, 47, 83, 84 |
| Hiring & admissions (`hiring_admissions`) | 7 | 16, 19, 71, 78, 15, 33, 75 |
| Government licensing, permits & benefits (`govt_licensing`) | 6 | 11, 30, 80, 95, 73, 7 |
| Recognition & creative judging (`recognition`) | 5 | 26, 40, 48, 67, 92 |

**Discarded (17):** health (`0, 23, 81` — too few to estimate a stable rate); sign-flipped
`54` (block telemarketer) and `77` (suspend account), where "yes" is *un*favorable and so breaks
DiscrimEval's yes=favorable premise; no-institution dating `1, 66`; platform/content `13, 28`;
housing/tenancy `24, 76`; and misc. discretionary one-offs `10, 14, 21, 44, 72, 43`.

Two deliberate near-clean merges: `89` (a county property-deed grant) sits with **business
financing**, and **hiring & admissions** unifies jobs with school admissions/scholarships as
"merit selection of a person".

## Split: hold out whole categories

The split unit is the **category** (`split_unit(cond) = cond["scenario"]`); the universe is the
nine categories (`split_units()`).

- **Forced test (3):** `consumer_service`, `immigration`, `recognition` — diverse spheres
  (commerce / state-gatekeeping / creative judging). `forced_split` pins these to `test`.
- **Dev pool (6):** the rest, paired into **three fold-groups** via `fold_group` so 3-fold CV does
  leave-one-group-out = **4 train / 2 eval** per fold:
  - `f_money_expert` = personal_lending + academic_review
  - `f_biz_people` = business_financing + hiring_admissions
  - `f_sec_state` = access_security + govt_licensing

This tests a use-case-relevant generalization: *predict your bias in a decision domain you didn't
tune on*. Build/inspect the manifest with `bp-splits --eval discrimeval [--build]`
(`results/discrimeval/splits.json`).

## Prediction: every method predicts the bias directly

`scoring_semantics = "bias_contrast"`; `score_contrasts` is **dual-mode** per `(category, axis,
value)`: `actual` always differences the measured cell rates, and `predicted` is the direct
contrast-keyed gap when present, otherwise (for measurement/trained methods that can only produce
per-cell rates) the differenced cell-rate predictions.

Every **introspective** method predicts the gap **directly** via a comparative prompt: it asks the
model for the signed approval-rate gap between one comparison applicant and the baseline *within a
category's decisions*. Asked in both A/B orders (the swapped order negated and pooled) to cancel
position bias; `--runs` must be even. Predictions are keyed by contrast key. Routed through
`elicitation.elicit_comparative_bias` whenever `scoring_semantics == "bias_contrast"`.

- **`self_report`.** Plain approval framing; the `perspective` (first/third person) and
  `honesty_nudge` knobs are swept. At its default setting this is the former standalone `direct_bias`
  method (now merged in).
- **`value`** — **age axis only.** A different framing that does no pairwise comparison (so it isn't
  just a copy of self_report): two opposing value-principles on a -100..+100 scale (0 = neutral),
  the model's lean read as the predicted gap. The `aggregate` knob switches the scope wording;
  the `value_detail`/`context` knobs don't apply to this value-scale prompt and are pruned (so the
  tuned ids are just `value` / `value-aggregate`). The − pole favours the safe,
  conventional, category-specific choice (`_CATEGORY_ACTION`: lending / security clearance / ...);
  the + pole *resists an assumption* about an age-correlated trait (`_AGE_DOUBT`: short track record
  → experience past its best → reduced stamina → failing cognition) — framing the trait as a bias to
  resist, never stating an age or asserting it is true. **Race & gender are not predicted by the
  value method** (self_report covers them): they have no neutral value-relevant correlate, so the
  only thing that would distinguish their comparison groups is a stereotype, which we won't encode.
  `value` has no swept args (single value-scale prompt → the tuned id is just `value`).

Predicting per-cell rates and *differencing* them (the old `self_report` derived path) is left for a
separate experiment; `score_contrasts` retains the derived path only for measurement/trained methods
(e.g. `behavioral_sampling`).

`selection_cv` pools the chosen method's *contrast pairs* per held fold (contrasts are disjoint by
category), so it is keying-agnostic across direct and derived methods.

**Reporting.** Both the per-eval report and the cross-eval overview score self-prediction on the
**bias** (per-contrast gap), on the **dev** split, via `metrics.corr_spec`/`mae_spec` — the same
path the leaderboard uses — so the correlation on the DiscrimEval report page equals the one in the
overview. The report does **not** show per-cell-rate prediction accuracy (only the bias table).

## Variants: explicit (default) + implicit

`DiscrimEvalSpec(config=...)` is registered twice: **`discrimeval`** (explicit, named demographics)
and **`discrimeval_implicit`** (name-based). Only *measurement* differs (the templates used); the
**prediction prompts stay explicit-demographic** for both, since the bias being predicted is the
demographic gap. Each eval name gets its own `results/<name>/…` tree and split manifest.

## Running it

```bash
bp-targets  --eval discrimeval --model <m> [--samples 10] [--categories ...]   # measure cells
bp-splits   --eval discrimeval --build                                          # build the manifest
bp-benchmark --evals discrimeval --models <m>                                   # predict (dev) + score
bp-report   --eval discrimeval                                                  # HTML incl. bias table
```

`run_behavior` sends each `(cell × template)` prompt `--samples` times and aggregates per cell in
`produce_targets`. The report's bias table (actual Δ vs. each method's directly-predicted Δ, with
MAE) is computed by `report_extra_html` from `targets.json` + the introspective methods' contrast-
keyed prediction files (`self_report` / `value` / `value-aggregate`).

### Subsampling for cheap test runs

`bp-targets --eval discrimeval` exposes three independent knobs (compose freely):

- `--categories a,b` — measure only some categories (the baseline cost driver).
- `--axes race,gender` — measure only some comparison axes (subset of `race,gender,age`); the
  baseline cell is always measured, and an unmeasured axis just yields no contrast.
- `--max-templates N` — cap templates per category to the first N (sorted qids).
- `--samples N` — requests per template (the rate's sample size).

These flow through to **prediction** automatically: the benchmark derives the conditions from the
measured targets, and the comparative-bias prediction infers the categories *and* axes present, so it
only predicts contrasts it can score. Example smoke run:

```bash
bp-targets --eval discrimeval --model <m> --categories immigration,recognition \
           --axes race --max-templates 2 --samples 2
```
