# Self-prediction as a fixed point — Llama-3.3-70B, rounds 1 and 2 (2026-07-09)

> **Exploratory note (dev split only).** The numbers below predate the final runs and the
> frozen-test evaluation; the paper's reported figures are the canonical ones.

Setup, corpus and controls: `docs/selfpred-corpus-v1.md`. Reports:
`results/reports/selfpred_finetune_llama.md` (round 1) and `…_v2.md` (round 2), both gitignored.

Each round is one step of the Picard iteration

    G(M) = finetune(base, behavior(M))

i.e. measure a model's own behavior, then train a fresh LoRA **from base** to predict it. The fixed
point of `G` is a model that predicts itself. Training from base every round (rather than continuing
the previous LoRA) keeps the recipe identical, so versions are directly comparable.

The quantity being predicted is the **polarity gap** `yes_pos + yes_neg - 1` — the model's own
acquiescence bias on a family of questions. It is the model-specific part: across models the raw
`rate` correlates +0.836 but the gap correlates −0.066, and a cross-model prior scores −0.14…−0.22
on it here. All metrics are per **category**, n=40 held out entirely from training.

## Result: the iteration contracts, and self-knowledge follows

| round | predicts its **own** behavior | predicts its **training target** | \|Δgap\| vs previous | signed |
|---|---|---|---|---|
| base | +0.444 | — | — | — |
| v1 | +0.633 | +0.841 | 0.275 | +0.263 |
| v2 | **+0.915** | +0.951 | **0.066** | +0.062 |

Noise ceiling on the gap is 0.98–0.99 (labels use 200 items/half, SE 0.050).

- **Contraction factor** `|Δgap|(v1→v2) / |Δgap|(base→v1)` = **0.24**. The map is a contraction, so
  the iteration converges rather than oscillating.
- **The self-knowledge deficit closes.** The quantity that matters is how much worse a model
  predicts *itself* than it predicts the labels it was trained on: v1 was **+0.208** worse, v2 only
  **+0.036** worse. v2 is at its noise ceiling on both.
- Mean gap moves −0.120 → +0.143 → +0.205; a geometric extrapolation puts the fixed point near
  **+0.225**. Note the model does not converge to its *original* bias — it converges to a *new*,
  self-consistent one.

## Why round 1 alone was misleading

Round 1 looked like a triumph (gap r +0.444 → +0.841) but the drift control showed the finetune had
moved the model's own acquiescence by +0.263, nearly a full between-category sd, while leaving *what*
it does intact (|Δrate| 0.053, r(base,tuned rate) = +0.967). Behavior moved **away** from the
model's own predictions, so this was never self-fulfilment; it was the opposite — the LoRA installed
accurate knowledge of the *pre-finetune* model while changing the model. Scored against its own
behavior, v1 managed only +0.633.

Round 2 fixes exactly that: trained on v1's behavior, v2 predicts v1 at +0.951 **and itself at
+0.915**, because v2's behavior barely moved from v1's.

**The lesson generalises beyond this experiment**: any self-prediction finetune changes the model it
was trained to describe, so a single round's "self-prediction accuracy" overstates self-knowledge.
The honest number is always `predictions vs the tuned model's own re-elicited behavior`. Binder et
al.'s setup has the same exposure.

## ⚠ Known defect in v2's training mix (not yet fixed)

Self-prediction means predicting **your own** behavior, so every response in the training mix must
come from the model being finetuned — and must be **re-collected each round**, because round *n+1*
is trained on round *n*'s behavior.

v2 violates this for half its data. `build_selfpred_corpus._binder_pairs` hardcoded
`logs/introspection_finetune_llama30k` (base-Llama's object-level responses), so:

| pairs | source of the labels | correct for v2? |
|---|---|---|
| 4,348 rate pairs | **v1's** measured behavior | ✅ |
| 5,000 Binder property pairs | **base-Llama's** responses | ❌ should be v1's |

So ~53% of v2's training mix taught it to predict a model it is not. The same applies, less
severely, to v1: its Binder pairs *were* base-Llama's, and v1 is trained on base-Llama's behavior, so
v1's mix is internally consistent. **Only v2 is affected.**

What this does and does not invalidate:

- **The fixed-point result stands.** v2's +0.915 against its own re-elicited behavior, the 0.24
  contraction factor, and the drift measurements are all scored against behavior measured *after*
  training. They do not depend on the training mix being ideal.
- If anything the convergence result is **conservative**: a correct round 2 trains on a fully
  self-consistent mix and should converge at least as well.
- The absolute numbers for v2 (gap r vs its training target, +0.951) mix two label sources and should
  not be quoted as "trained purely on v1's behavior".

Fixed in the tooling, not yet in the artifacts: `--stage binder` collects the finetuning model's own
responses per round, `_binder_pairs` takes `--binder-dir` and hard-fails when that model's responses
are absent rather than silently reaching for Llama's. **Re-running v2 on the corrected mix is
outstanding.**

## Caveats

- The fixed point is a property of `G`, not of the model's "true" self-knowledge. v2 is
  self-consistent partly because the LoRA moved its behavior less, not only because it introspects
  better. A shuffled-label control (train on permuted targets; must give r≈0) has **not** been run.
- Only the gap converges this cleanly. `rate` was already near ceiling (+0.853 base) and a
  cross-model prior predicts it about as well (+0.81), so it carries little self-specific signal.
- 40 held-out categories: a gap r of +0.915 has a 95% CI of roughly [+0.84, +0.95].

## Operational notes (both cost real time)

- **Empty responses.** A dedicated endpoint under load drops bursts of requests; the client returns
  `""` after retries, and a resumable cache then records them as *done*. That silently poisoned 3,575
  items (9 whole categories) of the v1 pass before it was caught. `build_selfpred_corpus._elicit_model`
  now aborts a chunk if >2% of calls come back empty, writing nothing. Keep workers at ~12–16;
  24–32 triggers it.
- **Idle timeout.** `deploy` used `inactive_timeout=20`, which STOPPED the endpoint during a pause
  between passes; every subsequent request then hung against a dead endpoint. Now 120 minutes — long
  enough never to fire mid-run, short enough to cap the damage if a crash skips `teardown`.
- Together has no serverless LoRA for Llama-70B: a dedicated 4×H100 endpoint (~$0.36/min) must be
  deployed and torn down per use. `teardown` verifies deletion with
  `endpoints.list(type="dedicated", mine=True)` — an unfiltered list also returns Together's public
  endpoints, which we neither own nor pay for.

## Models kept (for later evaluation on the four scored evals)

Stable finetuned-model ids (endpoint names rotate on every deploy):

    v1  phblandfort_ac3d/Llama-3.3-70B-Instruct-Reference-selfpred-llama-v1-03d958a8
    v2  phblandfort_ac3d/Llama-3.3-70B-Instruct-Reference-selfpred-llama-v2-e57d5a06

Both need `finetune_together.py deploy --out logs/selfpred_llama[_v2]` before use, and
`teardown` after. See the commented entries in `behavior_prediction/models.yaml`.
