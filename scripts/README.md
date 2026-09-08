# scripts/

Everything here runs from the repository root and expects the package installed
(`pip install -e .`). Two kinds of thing live here:

- **top level** — the analyses, figures and exports behind the paper, plus the tooling they
  depend on. Most read committed results off disk and call no models.
- **`runs/`** — operational drivers: the shell lanes that actually measured behavior,
  elicited predictions, and trained/served the finetunes. They are kept as the record of how
  the results were produced. They are not a supported interface: several are one-offs pinned
  to a particular model, endpoint or point in the run order, and they will bill real money.

## Paper analyses and figures

`paper/main.tex` names the generator of each block of numbers in the `\input` comments at the
top of the file; these are those generators. All of them read
`results/reports/evaluation.json` or the committed prediction files and write into
`paper/generated/`.

| script | produces |
|---|---|
| `run_minimal_pipeline.sh` | tune → evaluate → macros → plots, offline, on a small scope |
| `significance_tests.py` | significance / equivalence macros and the claim tiers (`stats.tex`) |
| `predictability_decomposition.py` | the self-specific share per evaluation |
| `frontier_selfspec.py` | scale and reasoning analysis — does capability buy self-knowledge? |
| `oracle_identity_transfer.py`, `identity_transfer_figure.py` | identity transfer (diagonal vs off-diagonal) |
| `hero_figure.py` | `fig:overview` |
| `capability_scaling_figure.py` | `fig_scaling` |
| `selfgeneric_ablation.py` | the self-vs-generic phrasing ablation (all pre-specified analyses) |
| `selfgeneric_info_gain.py`, `selfgeneric_samples.py`, `selfgeneric_dump.py` | its paired and sample-level views |
| `generic_prior_analysis.py` | is the self-image more than "the generic assistant"? |
| `other_report_matrix.py`, `fewshot_analyst_matrix.py`, `analyst_zero_info.py` | the subject-axis and analyst-gain matrices |
| `selfserving_bias.py` | self-serving level shift |
| `tuning_sensitivity.py` | the hindsight-tuning bound |
| `learned_flip_sweep.py` | learned sign-flipping applied to every method (appendix) |
| `spearman_robustness.py` | rank-correlation robustness table |
| `export_grain_macros.py`, `export_selfpred_paper.py`, `export_finetune_full.py` | LaTeX table/macro exports |
| `pilot_protocol_report_score.py` | the protocol-tier table |

## Measurement quality

| script | purpose |
|---|---|
| `noise_ceiling.py` | split-half / bootstrap reliability — the ceiling on any correlation |
| `noise_ceiling_analytic.py` | independent analytic cross-check of the same quantity |
| `check_raw_targets_sync.py` | guard: every `behavior_raw.json` agrees with its committed `targets.json` |
| `pb_topup.py` | equalize PropensityBench dev coverage across models before comparing them |
| `fix_reportmean_test_donors.py` | repair of donor-averaged predictions computed against partial donor sets |
| `method_diagnostics.py`, `cot_flip_sign_diagnostics.py`, `method_stacking.py` | offline diagnostics |

## Finetuning

Corpus construction, training and serving for the self-prediction finetunes. Training and
endpoint scripts spend money and hold GPUs; endpoints bill per minute and every driver in
`runs/` tears its endpoint down on exit.

`build_selfpred_corpus.py`, `finetune_together.py`, `finetune_fireworks.py`,
`finetune_introspection.py` (Tinker), `deploy_v2.py`, `dmi_deploy.py`, `dmi_teardown.py`,
`set_model_string.py`, `selfpred_report.py`, `selfpred_headroom.py`,
`render_corpus_viewer.py`.

## Pilots and probes

Exploratory work that informed the design: `pilot_*.py` (alternative targets, implicit
DiscrimEval, generator audits, exemplar counts, transfer framings), `prompt_iteration*.py`
(prompt rounds for the self-report ask), `rhg_*_probe.py` and `deceptionbench_probe.py`
(saturation/validity checks on candidate evaluations), `proxy_probe.py`,
`screen_condition_families.py`. Their numbers are dev-split and predate the final runs — see
the notes in `docs/`.

## runs/

Ordered roughly as they were used:

- **Measurement and elicitation** — `run_endpoint_model.sh`, `run_frontier_model.sh`,
  `run_all_endpoints.sh`, `run_default_complete.sh`, `run_frontier_complete.sh`,
  `run_results_*.sh`, `fill_frontier_gaps.sh`, `run_*_fill*.sh`.
- **Sampling regeneration** — `run_sampling_regen_{small,frontier}.sh` and the
  `run_sampling_reasoningfix_*.sh` re-runs that repinned the generator's reasoning config,
  plus `run_protocol_*.sh` for the protocol-tier arms.
- **PropensityBench** — `run_pb_topup.sh`, `run_pb_predictions_refresh.sh`,
  `pb_complete_finetunes.sh`, `pb_qwen72_pair.sh`.
- **Finetunes** — `selfpred_*.sh` (corpus → train → serve → measure, per family),
  `intro30k_*.sh` (the Binder-style property-prediction finetunes),
  `run_finetunes_*.sh`, `refinetune_*.sh`, `gap_eval_*.sh`, `measure_*.sh`,
  `deepseek_base_coverage.sh`, `diagnose_v1.sh`.
- **Final passes** — `run_stage7_dev_eval.sh` (the canonical dev pass: re-tune, evaluate and
  the whole downstream report/figure/tex chain, entirely offline), then the frozen-test
  lanes: `run_test_targets.sh`, `run_frozen_test_elicitation.sh`, `run_frozen_test_am.sh`.

Several need external clones on `TAU2_REPO` (tau2-bench) or `EVALS_REPO`; both default to a
sibling of this repository. Elicitation is globally rate-limited at roughly 50 item-calls per
minute regardless of concurrency, so these lanes are long — never wrap one in `timeout`, which
kills it before it writes its prediction file.
