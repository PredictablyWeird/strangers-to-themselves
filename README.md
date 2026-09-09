# Behavior Prediction

Code and results for **"Strangers to Themselves: What Language Models Say About Themselves Is
Generic"** (`paper/`).

Can a model predict its own behavior? We measure how often a model takes a target action
under each condition of a behavioral evaluation, elicit a *prediction* of those rates through
several channels, and score the prediction against the measurement. The headline: what a
model says about itself tracks behavior about as well when the self is removed from the
question — and considerably worse than simply averaging *other* models' measured rates.

The design has three axes. **Information**: abstract description → the model's own measured
history → the verbatim items. **Subject**: you / a named other model / a generic assistant.
**Form**: rate, forced choice, sampled behavior.

## Setup

Python 3.12+. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'     # deps + the bp-* commands + pytest
.venv/bin/python -m pytest     # 247 passed, 6 skipped (the skips need the clones below)
```

or the equivalent with `python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'`.

Needs a `.env` with `OPENROUTER_API_KEY` — every subject model, grader and analyst is routed
through OpenRouter (the finetuning scripts additionally want `TOGETHER_API_KEY` /
`TINKER_API_KEY`). Three evaluations read their items from an external clone, each defaulting
to a sibling of this repository and overridable by environment variable:
PropensityBench needs [`scaleapi/propensity-evaluation`](https://github.com/scaleapi/propensity-evaluation)
(`--pb-repo`/`--pb-python`), `reward_hacking` needs
[`keing1/reward-hack-generalization`](https://github.com/keing1/reward-hack-generalization)
(`REWARD_HACK_REPO`), and the τ² evals need a `tau2-bench` clone (`TAU2_REPO`) — without that
last one the τ² item-informed methods silently produce nothing.

## The benchmark

Split units are partitioned per evaluation into a **dev pool** and a frozen **test** set
(`results/<eval>/splits.json`), so that fitting method parameters and selecting among methods
never touch the reported score. The primary metric is Pearson *r* across conditions —
whether a method tracks *which* conditions are riskier — macro-averaged over (model × eval)
cells with bootstrap CIs. Constant predictions score 0.

```bash
# 1. Measure behavior, then extract ground-truth rates
bp-sweep   --eval sycophancy_pushback --model llama-3.3-70b
bp-targets --eval sycophancy_pushback --model llama-3.3-70b

# 2. Generate dev predictions for every setting of every method
bp-benchmark --models llama-3.3-70b --phase generate

# 3. Tune: pick the best setting per (method, eval) on dev, aggregated over the selection
#    pool. Calls no models.
bp-tune --evals sycophancy_pushback

# 4. Evaluate the tuned settings -> results/reports/ + paper/generated/
bp-evaluate --evals sycophancy_pushback --split dev
```

`bp-elicit` runs a single method; `bp-compare` diffs prediction files; `bp-splits` inspects or
rebuilds a split manifest. Every command takes `--help`.

**The test split was elicited and scored exactly once**, against a pre-registration written
first (`docs/test-prespecification.md`): claims, directions, margins and per-claim fallback
text, all pinned before any test number existed. Test is otherwise never touched —
`bp-evaluate` guards it behind `--allow-test`.

## Evaluations

Eight evaluations are scored (`ACTIVE_EVALS`), plus one reserved as held-out:

| eval | behavior measured | split unit |
|---|---|---|
| `sycophancy_pushback` | changing a correct answer under pushback | MMLU subject |
| `capability_mmlu` | correctness | MMLU subject |
| `reward_hacking` | picking an over-valued wrong answer on a graded exam | MMLU subject |
| `discrimeval` | demographic gap in favorable decisions (bias contrast) | topic category |
| `tau2_policy` | forbidden write call in an agentic customer-service sim | condition |
| `tau2_transfer` | the same, on transfer tasks | condition |
| `propensitybench` | invoking a forbidden tool under escalating pressure | domain/workspace |
| `mask_subdomain_pressure` | lying under pressure, against its own elicited belief | subdomain |
| `agentic_misalignment` *(held out)* | blackmail / leaking / murder | all forced to test |

The three MMLU-subject evals share one partition by construction (same subjects, same seed),
and PropensityBench forces every cyber-security workspace into test, so the split also carries
a domain-generalization signal.

`agentic_misalignment` is never tuned or fit on: its settings were frozen by donor-tuning
across the other eight before any prediction existed, and it is reported separately as an
external-validity check. Also registered but not in the scored suite:
`propensitybench_benign`, `discrimeval_implicit`, `mask`, `mask_subdomain`.

Adding an evaluation is one `EvalSpec` in `evals/` (conditions, `frame(cond)`,
`produce_targets`, `split_unit`, scoring semantics) plus registration.

## Methods

Fifteen methods run by default, spanning the three axes (`methods/`, prompts single-sourced in
`elicitation.py`):

- **Asking, abstract** — `self_report` (how often would *you*…), `generic_report` (…would
  *an assistant*), `value` (which of two of your values wins), `pairwise` (forced choice
  between two conditions, both orders).
- **Asking, item-informed** — `informed_oracle` (shown the verbatim items),
  `generic_oracle` and `oracle_pairwise` (the same exhibits, subject or form swapped).
- **History-informed** — `few_shot` (the model's own measured rates as prior conversation
  turns, then the `k+1`-th question), `few_shot_other` (byte-identical turns, answered by a
  different model), `llm_prediction` (the same labels as one text block, read by a fixed
  analyst).
- **Watching instead of asking** — `behavioral_sampling` (generate scenarios, run the model,
  grade the responses), `informed_sampling` (same, with the real items shown to the generator).
- **Pool-derived controls** — `cross_model_mean` (other models' measured rates on the same
  condition), `report_mean` and `oracle_report_mean` (other models' *predictions*). These
  never consult the model under test, so they are the "no self-knowledge needed" nulls every
  introspective method has to beat.

A method's settings are declared in `methods.yaml`, not in code; each setting is its own
`method_id` (`self_report`, `self_report-honest`, …). `bp-tune` picks one shared best setting
per (method, eval) over the selection pool. Adding a setting is a YAML edit; adding a method
is one `MethodAdapter`.

Fourteen further methods are registered but off by default — the phrasing ablation arms, the
pilots, `list_experiment`, the `oracle_xmm` ensembles, and a trainable plumbing baseline.

## Models

`models.yaml` maps short names to provider strings plus a reasoning mode; unknown `--model`
values pass through verbatim. The paper's pool is 12 models over six labs, half small and
half frontier, with reasoning on and off where applicable. Every result file records the full
model identity (`{model, model_shortcut, reasoning}`) and directories are keyed by a
reasoning-aware slug, so reasoning variants never share a directory and predictions whose
identity disagrees with their targets are skipped at score time.

One standard grader project-wide (`common.DEFAULT_GRADER`), with a different cheap grader
substituted whenever the subject model is the grader's own family, so no model judges itself.

## Layout

```
behavior_prediction/   # the installable package
  common.py            # model registry + identity, parsing, elicitation runners, paths
  elicitation.py       # prediction-method prompt templates (single-sourced)
  splits.py metrics.py selection.py   # dev/test manifests, scoring, CV folds
  tuning.py evaluate.py results_io.py # bp-tune / bp-evaluate / shared loaders
  evals/ methods/ cli/ # the three adapter seams + console entry points
  models.yaml methods.yaml            # model registry; method grids and pools
scripts/               # analyses and figures behind the paper (see scripts/README.md)
docs/                  # design notes, pilots, the frozen-test pre-registration
paper/                 # LaTeX source; numbers in paper/generated/ are script-generated
results/<eval>/<model-slug>/{targets.json,predictions/*.json}   # canonical record
tests/                 # pytest suite
```

## Reproducing the paper

`results/` holds the committed targets and predictions, so the scoring and figure pipeline
runs offline with no model calls:

```bash
scripts/run_minimal_pipeline.sh      # tune -> evaluate -> macros -> plots (dev, scoped small)
```

The full set of analyses behind the paper's tables and figures is listed in
`scripts/README.md`; the `\input` comments at the top of `paper/main.tex` name the script
that generates each block of numbers.

## Full artifacts

The prediction files under `results/` are the canonical scoring record, but the largest ones
(item-informed sampling and oracle methods, roughly 350 files above 1 MB) ship here with the
verbatim model transcripts (`raw`) and generated scenario texts (`scenarios`) removed; each such
file carries a `release_note` block saying so. Nothing the scoring pipeline reads is affected:

```bash
bp-evaluate --split test --allow-test \
  --evals sycophancy_pushback,discrimeval,capability_mmlu,reward_hacking,tau2_policy,tau2_transfer,propensitybench,mask_subdomain_pressure,agentic_misalignment
```

reproduces `paper/generated/results.tex` and the tables next to it from this tree.

The complete record — unstripped prediction files, per-call reasoning traces, sampled
transcripts, and the raw Inspect logs of every behavioral measurement — is archived on Zenodo:
**DOI: 10.5281/zenodo.22674300**. Unpack it over this checkout to restore the full tree. `scripts/make_release.py`
is the script that produced this slimmed tree from the full one.
