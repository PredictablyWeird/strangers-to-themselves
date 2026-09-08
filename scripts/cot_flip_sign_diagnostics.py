"""cot_flip sign diagnostics: is the fold-learned sign stable, and is the out-of-fold gain
recoverable from noise? (Roadmap review-pass-2: "cot_flip's learned sign = fishing?")

Reads the stored ``predictions/cot_flip.json`` files only — each out-of-fold entry carries the
``fit`` info of the fold that predicted it (sign, train_r, train_pred_mean/sd, n_train) plus the
raw pre-flip estimate ``raw_predicted_rate`` — so both diagnostics are computed without any new
elicitation:

  1. SIGN STABILITY: group a cell's entries by their ``fit`` tuple (= CV fold) and report the
     per-fold signs. A cell is fold-stable when all folds learn the same sign; the per-model
     majority sign across evals shows the model-specific-polarity claim directly.
  2. SIGN-PERMUTATION CONTROL: rebuild each fold's out-of-fold predictions from the stored raw
     estimates (z-scored with the stored fold mean/sd), but with the per-fold sign drawn at
     random (+/-1); pool and correlate against the dev targets. Repeating this REPS times gives
     the null distribution of the pooled dev r under arbitrary per-fold signs; the one-sided
     permutation p is the fraction of draws >= the actual learned-sign r. With F folds there are
     only 2^F distinct sign assignments, so for small F we enumerate exactly instead of sampling.

Default-pool models only; dev split; evals where cot_flip applies (no DiscrimEval — bias evals
have no cot_flip). Cells with a constant prediction vector or <3 scored conditions are skipped
(matching the evaluator's treatment of undefined correlations).

Run:  .venv/bin/python scripts/cot_flip_sign_diagnostics.py [> results/reports/cot_flip_sign_diagnostics.txt]
"""
from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from behavior_prediction import common, metrics, splits
from behavior_prediction.evals import get_spec

EVALS = ["sycophancy_pushback", "capability_mmlu", "reward_hacking", "tau2_policy",
         "propensitybench"]
REPS = 5000          # sampled draws when 2^F is large
SCALE = 0.1          # CotFlip.SCALE (fixed, correlation-invariant)
rng = random.Random(1234)

POOL = list(yaml.safe_load(open(Path(__file__).resolve().parent.parent
                                / "behavior_prediction/models.yaml"))["pools"]["default"])


def pooled_r(fold_entries: dict[tuple, list[tuple[float, float, float, float]]],
             signs: dict[tuple, float]) -> float | None:
    """Pooled dev correlation with per-fold signs applied to the stored z-scores.
    ``fold_entries[fold] = [(actual, raw, mean, sd), ...]``."""
    pairs = []
    for fold, rows in fold_entries.items():
        s = signs[fold]
        for actual, raw, m, sd in rows:
            z = (raw - m) / sd if sd > 1e-9 else 0.0
            pairs.append((actual, 0.5 + SCALE * s * z))
    return metrics.pearson(pairs)


def main() -> None:
    print(f"{'eval':<20}{'model':<26}{'folds':>6}{'signs':>10}{'stable':>7}"
          f"{'r_learned':>10}{'r_flip':>8}{'perm p':>8}{'null':>16}")
    sign_by_cell: dict[str, dict[str, float]] = defaultdict(dict)
    stable_cells = flip_cells = 0
    perm_ps: list[float] = []
    # Per cell: (r_learned, fold_entries, ordered folds) for the pooled macro permutation below.
    cells: list[tuple[float, dict, list]] = []
    for ev in EVALS:
        spec = get_spec(ev)
        man = splits.load_manifest(ev)
        for slug in POOL:
            ppath = Path(f"results/{ev}/{slug}/predictions/cot_flip.json")
            tpath = Path(f"results/{ev}/{slug}/targets.json")
            if not ppath.exists() or not tpath.exists():
                continue
            preds = common.load_json(ppath)["predictions"]
            targets = common.load_json(tpath)["targets"]
            dev = set(splits.split_keys(spec, targets, man, "dev"))
            # fold key = the fit tuple stored on each out-of-fold entry
            fold_entries: dict[tuple, list] = defaultdict(list)
            learned_signs: dict[tuple, float] = {}
            for key, e in preds.items():
                fit = e.get("fit")
                t = targets.get(key)
                if (key not in dev or not fit or t is None or t.get("rate") is None
                        or e.get("raw_predicted_rate") is None):
                    continue
                fold = (fit["sign"], fit["train_r"], fit["train_pred_mean"],
                        fit["train_pred_sd"], fit["n_train"])
                fold_entries[fold].append((t["rate"], e["raw_predicted_rate"],
                                           fit["train_pred_mean"], fit["train_pred_sd"]))
                learned_signs[fold] = fit["sign"]
            n_scored = sum(len(v) for v in fold_entries.values())
            if not fold_entries or n_scored < 3:
                continue
            signs = [learned_signs[f] for f in fold_entries]
            n_pos, n_neg = signs.count(1.0), signs.count(-1.0)
            stable = (n_pos == 0 or n_neg == 0)
            r_learned = pooled_r(fold_entries, learned_signs)
            r_flipped = pooled_r(fold_entries, {f: -s for f, s in learned_signs.items()})
            if r_learned is None:
                continue
            # train_r can be None (constant fit-split predictions), so sort with a None-safe key
            folds = sorted(fold_entries,
                           key=lambda f: tuple(-1e9 if v is None else v for v in f))
            # Null distribution over per-fold random signs (exact when 2^F is small).
            if 2 ** len(folds) <= 4096:
                draws = [dict(zip(folds, combo))
                         for combo in itertools.product((1.0, -1.0), repeat=len(folds))]
            else:
                draws = [{f: rng.choice((1.0, -1.0)) for f in folds} for _ in range(REPS)]
            null = [r for d in draws if (r := pooled_r(fold_entries, d)) is not None]
            p = sum(1 for r in null if r >= r_learned) / len(null) if null else float("nan")
            perm_ps.append(p)
            cells.append((r_learned, fold_entries, folds))
            null_mean = sum(null) / len(null)
            null_hi = sorted(null)[int(0.95 * (len(null) - 1))]
            stable_cells += stable
            flip_cells += not stable
            sign_by_cell[slug][ev] = (n_neg >= n_pos and -1.0 or 1.0) if not stable else signs[0]
            print(f"{ev:<20}{slug:<26}{len(folds):>6}{f'{n_pos}+/{n_neg}-':>10}"
                  f"{('yes' if stable else 'NO'):>7}{r_learned:>+10.3f}{r_flipped:>+8.3f}"
                  f"{p:>8.3f}{f'{null_mean:+.2f}/{null_hi:+.2f}':>16}")
        print()

    print(f"Fold-stability: {stable_cells}/{stable_cells + flip_cells} cells have the same "
          f"learned sign in every CV fold.")
    print("\nMajority learned sign per (model, eval):")
    evs = [e for e in EVALS]
    print(f"{'model':<26}" + "".join(f"{e[:12]:>14}" for e in evs))
    for slug in POOL:
        row = "".join(f"{('+' if sign_by_cell[slug].get(e) == 1.0 else '-' if sign_by_cell[slug].get(e) == -1.0 else '.'):>14}"
                      for e in evs)
        print(f"{slug:<26}{row}")
    sig = sum(1 for p in perm_ps if p <= 0.05)
    print(f"\nPermutation control: {sig}/{len(perm_ps)} cells with p<=0.05 "
          f"(one-sided, learned-sign r vs random per-fold signs). NOTE: with 5 folds the "
          f"per-cell floor is p=1/32=0.031 (3 folds: 1/8) — per-cell p is floor-limited, "
          f"which is what the pooled test below is for.")

    # Pooled macro permutation: draw random per-fold signs in EVERY cell simultaneously and
    # compare the macro mean r over cells against the actual learned-sign macro. Not
    # floor-limited (the null has 2^(total folds) states).
    actual_macro = sum(r for r, _, _ in cells) / len(cells)
    null_macro: list[float] = []
    for _ in range(REPS):
        rs = []
        for _, fe, folds in cells:
            d = {f: rng.choice((1.0, -1.0)) for f in folds}
            r = pooled_r(fe, d)
            if r is not None:
                rs.append(r)
        if rs:
            null_macro.append(sum(rs) / len(rs))
    pm = sum(1 for m in null_macro if m >= actual_macro) / len(null_macro)
    null_mean = sum(null_macro) / len(null_macro)
    null_hi = sorted(null_macro)[int(0.99 * (len(null_macro) - 1))]
    print(f"Pooled macro permutation over all {len(cells)} cells: actual macro r "
          f"{actual_macro:+.3f} vs null mean {null_mean:+.3f} (99th pct {null_hi:+.3f}); "
          f"one-sided p = {pm:.4f} ({REPS} draws).")


if __name__ == "__main__":
    main()
