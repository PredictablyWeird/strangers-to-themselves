# Frozen-test pre-specification

Status: **FROZEN 2026-08-16** — the frozen-test run was ordered with all pins filled from
the final dev artifacts (pin commits `4e0271d`..`96027db`; ceiling rule = target reliability
≥ 0.5, commit `e21a729`; C8(a) stands at the committed +0.12). From here: no method,
setting, prompt, pool, grader, or threshold change; test is evaluated once. Test elicitation
runs via the top-up seam (`eb6a124`) — dev predictions stay byte-identical.

## 1. Process rules

1. **One evaluation.** Test split elicited (`--include-test --tuned-only`) and evaluated
   exactly once (`bp-evaluate`), immediately before submission.
2. **Order.** stage-7 dev evaluation final → fill §5 pins, freeze → measure remaining test
   targets (τ² test tasks; PB test cells small-four cohort; MASK done) → test elicitation →
   one test evaluation → swap macros at the marked TODOs.
3. **Failures.** Ceiling-filter drops (method-blind), dev-documented refusal NAs, and
   provider errors are recorded, not re-rolled. Re-runs only for verifiable infrastructure
   failure (empty/corrupt file), never for a disliked number.
4. **Reporting.** Main tables test-only; the dev leaderboard and every dev-only analysis
   (sweeps, multi-setting tables) in the appendix; one appendix paragraph states dev↔test
   deltas.
5. **Failed claims.** A directional expectation that fails on test is reported as measured,
   its prose scoped to dev-split evidence, and named in one Limitations sentence. No silent
   rewording. (Per-claim fallbacks exist only where §4 states one.)

## 2. Statistics and scope

- Machinery: `scripts/significance_tests.py` — cellwise-paired bootstrap CIs, sign-flip
  permutation, TOST; macro over (model × eval) cells, default pool, tuned-frozen settings.
- **95% intervals only, everywhere.** TOST runs through the 95% CI (α=0.025 per side);
  "advantage at most X" bounds are the 95% CI's upper end. No 90% interval anywhere.
- **Claims are average-level.** Thresholds bind the macro (or the stated paired statistic).
  Per-eval or per-model exceptions are expected, reported, and are not refutation.
- **Method set for test:** the canonical tuned suite. `list_experiment` and `oracle_xmm*`
  are excluded — appendix, dev split only. No learned-sign decode on test.
- **Frontier coverage:** all 8 evals (τ²-transfer + MASK-pressure included).
- **Sampling generator:** gpt-5.5-off (reasoning disabled) for all sampling variants.
  RE-PINNED 2026-08-16: the §1.0c regeneration is complete and verified (96/96 files, fresh
  digests, audited generator). Final dev anchors: behavioral_sampling +0.02 [−0.04,+0.09],
  informed_sampling +0.25 [+0.19,+0.30], informed_oracle +0.26 [+0.20,+0.31],
  protocol_sampling +0.04 (appendix).
- **Default pin rule:** unless stated otherwise, `[PIN]` = final dev value + 0.10 slack.

## 3. The claim ladder

For "X is no better than Y" claims (C3a, C6a): Δ = X − Y with 95% CI [lo, hi]; margin
m = 0.20 × informed_oracle final-dev macro (PINNED 2026-08-16: 0.20 × 0.255 = **0.05**) — derived non-inferiority
style from the strongest asking-channel method; conservative vs. both the clinical
(40–50% of reference) and psychology (r = 0.1 absolute) conventions; frozen at pin time,
never recomputed on test. **The primary claim is estimation:** the headline reports the
CI's upper end ("any advantage is at most hi"); the margin only cuts the ladder. The
paper uses the sentence of the highest tier the test CI attains
(`significance_tests.py::claim_tier`, stamped into `stats.tex`):

| tier | condition (95% CI) | licensed sentence |
|---|---|---|
| **E** | CI ⊂ (−m, +m) | "statistically equivalent (TOST, ±m)" |
| **D** | hi < +m, lo ≤ −m | "no more accurate; any advantage ≤ hi" (CI < 0: control side strictly better) |
| **N** | straddles 0, hi ≥ m | "no significant difference"; cannot bound below m — named in Limitations |
| **S** | lo > 0, hi < m | "small but significant advantage ≤ hi"; headline softened |
| **F** | lo > 0, hi ≥ m | claim fails; for C3 = stop-and-rethink |

Design-resolution note (also stated in the paper's appendix): with 6 models per tier the
CI half-width is ≈ 0.04, the floor on resolvable margins. Equivalence at m is not
expected to survive every split; tier D carries the headline either way. Dev: C6 attains
E, C3(a) attains D (self-advantage capped at +0.00).

## 4. Pre-specified claims

**C1 — Abstract questions are very weak (average-level).** Self-report macro ≤ **+0.17**
(PINNED 2026-08-16: dev +0.07 + 0.10 slack);
every abstract-form macro (value, pairwise) ≤ **+0.26** (dev max = pairwise +0.16 + slack).
Wording is "barely / very weak
band", not "zero"; pairwise and generic report are expected to top the band (~+0.15 dev).
*If an abstract form's macro exceeds its band:* re-run the §4.4 genericness controls on
that form — the thesis survives if its gain is matched by its generic/committee controls.

**C2 — Information helps; predictions stay moderate.** few_shot − self_report and
informed_oracle − self_report paired CIs exclude 0; informed_oracle macro < 0.5, median
r/ceiling < 0.6.

**C3 — The signal is not about the self (the thesis).** (a) Δ = informed_oracle −
generic_oracle on the ladder; "removing the self costs nothing" requires tier D+; tier S
folds into C4's residue story; **tier F = stop-and-rethink: the title/abstract thesis is
revised** — the one claim never softened by rewording. (b) oracle_report_mean ≥
informed_oracle − 0.05. (c) report_mean − self_report ≥ 0, CI excluding large negative.

**C4 — Small self-specific residue, concentrated.** Identity-transfer advantage
(diagonal − off-diagonal, item-informed tier) in [0, 0.15], concentrated on
**PropensityBench (+0.21 dev) and Capability-MMLU (+0.17 dev)** (PINNED 2026-08-16;
pooled self-advantage +0.08). Zero/negative → "no detectable residue", conclusion drops
to one pocket. Large → C3 stop-and-rethink.

**C5 — Understatement bias.** Mean signed self-report bias ≤ 0 on norm-violating evals,
most negative on PropensityBench.

**C6 — Scale does not buy self-knowledge (directional).** (a) Δ = frontier − small
self-report tier difference on the ladder; abstract's "no more accurate than smaller
models'" requires tier D+ ("statistically equivalent" allowed at E). (b) frontier
identity-transfer advantage ≤ small-tier + 0.05.

**C7 — The DiscrimEval within-item pocket.** informed_oracle partial correlation given
cross_model_mean > 0 on DiscrimEval (CI excluding 0, or positive for ≥ **7/12** models —
PINNED 2026-08-16; dev: +0.21 [+0.08, +0.32], positive 9/12).

**C8 — Watching is not a shortcut, but shows the information axis.** (a)
behavioral_sampling macro ≤ **+0.12** (PINNED 2026-08-16: dev +0.02 + 0.10 slack); (b)
informed_sampling − behavioral_sampling paired CI excludes 0 (dev gap +0.22); (c)
informed_sampling macro ≤ informed_oracle macro + 0.05 (dev: +0.247 vs +0.255 — met with
room). *If (c) fails:* "not a shortcut" rests on cost + distance to ceilings.

**C9 — (target-side, descriptive) Self-specific share orders the evaluations.**
Capability/reward-hacking low, agentic evals high; PINNED 2026-08-16 (dev shares, all 8
evals): Capability 0.07 < RewardHacking 0.16 < MASK-pressure 0.36 < Sycophancy 0.45 <
DiscrimEval 0.52 < τ²-transfer 0.57 < PropensityBench 0.76 < τ²-policy 0.79. The claim
binds the ENDS (capability/reward-hacking bottom two; τ²-policy/PB top two), not every
adjacent swap.

**C10 — Held-out generalization: agentic_misalignment.** All-test, donor-tuned
(`donor_modal_best`, committed before any AM prediction), reported separately from the
8-eval suite; scored only after this doc freezes, `--evals agentic_misalignment`
explicitly, never writes `evaluation.*`/`paper/generated`. Direction-only expectations:
(a) abstract self-report in the weak band, denial + negative bias; (b) item-informed >
abstract; (c) generic-subject/others' controls match self-framed; (d) cross_model_mean
carries most predictable signal (caveat: 24 conditions, n=20, SE ≈ 0.11). AM
contradictions *scope* claims ("holds on the tuned suite, not held-out"), never revise
the 8-eval results. Paper footnote: June 2026 pilot ran self-report scripts on AM —
selection never saw it, but it is not virgin.

## 5. Pin sheet (FILLED 2026-08-16 from the post-§1.0c final dev artifacts)

| pin | rule | final value |
|---|---|---|
| ladder margin m | 0.20 × informed_oracle macro (0.255) | **0.05** |
| C1 self-report ceiling | dev +0.07 + 0.10 | **+0.17** |
| C1 abstract-band ceiling | dev pairwise +0.16 + 0.10 | **+0.26** |
| C4 concentration list | eval list from dev | **PropensityBench (+0.21), Capability (+0.17)** |
| C7 model fraction | proposed ≥ 7/12 | **≥ 7/12** (dev 9/12, partial +0.21 [+0.08,+0.32]) |
| C8(a) blind-watching ceiling | dev +0.02 + 0.10 | **+0.12** |
| C8(b,c) anchors | §1.0c re-run | **informed +0.247, gap +0.22, oracle +0.255** |
| C9 ordering | dev decomposition, 8 evals | **0.07 / 0.16 / 0.36 / 0.45 / 0.52 / 0.57 / 0.76 / 0.79** (ends bind) |
| C6(b) small-tier reference | dev identity-transfer | **small-tier transfer +0.05 → frontier ≤ +0.10** (dev frontier-off −0.06) |

No open decisions. (Margin anchor + fraction blessed 2026-08-15; §1.0c anchors re-pinned
and the sampling-budget ablation re-confirmed under the fixed generator 2026-08-16.)
