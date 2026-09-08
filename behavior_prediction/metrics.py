#!/usr/bin/env python3
"""Eval-agnostic accuracy metrics for prediction-vs-actual analysis.

A prediction method maps each condition to a predicted rate; the targets give the actual rate.
We score a method two ways:

- **MAE** — mean absolute error in rate (lower is better). Sensitive to a constant offset but
  blind to whether the prediction *tracks* which conditions are higher/lower.
- **Correlation** — Pearson r between predicted and actual rates across conditions (higher is
  better). Undefined (``None``) when either side has no variance — in particular when a method
  outputs the *same number for every condition*, which several models do. A low/undefined
  correlation next to a small MAE means the method isn't really predicting, just guessing a flat
  rate near the mean.

To calibrate MAE, compare against the **constant baselines** in :data:`BASELINES`: a predictor
that always answers 0%, 50%, or 100%. A method whose MAE doesn't beat the best baseline carries
no condition-level signal. Baselines are constant, so their correlation is always undefined.
"""

from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Any, Callable

# Constant-predictor baselines: label -> the rate (fraction) it always outputs.
BASELINES: dict[str, float] = {"always 0%": 0.0, "always 50%": 0.5, "always 100%": 1.0}


def pairs(targets: dict[str, Any], preds: dict[str, float | None],
          keys: set[str] | None = None) -> list[tuple[float, float]]:
    """(actual, predicted) over conditions where both are present.

    When ``keys`` is given, only those condition keys are scored — the seam used to restrict
    scoring to a train/dev/test split (see ``splits.py``). ``keys=None`` scores everything, so
    all existing callers are unchanged."""
    out: list[tuple[float, float]] = []
    for k, t in targets.items():
        if keys is not None and k not in keys:
            continue
        a, p = t.get("rate"), preds.get(k)
        if a is not None and p is not None:
            out.append((a, p))
    return out


def mae(targets: dict[str, Any], preds: dict[str, float | None],
        keys: set[str] | None = None) -> float | None:
    """Mean absolute error (fraction) between predicted and actual rate; None if no overlap."""
    ps = pairs(targets, preds, keys)
    return mean(abs(p - a) for a, p in ps) if ps else None


def correlation(targets: dict[str, Any], preds: dict[str, float | None],
                keys: set[str] | None = None) -> float | None:
    """Pearson correlation between predicted and actual rate across conditions.

    Returns None when it is undefined: fewer than 2 overlapping conditions, or zero variance on
    either side (e.g. the method predicts a constant rate)."""
    return pearson(pairs(targets, preds, keys))


def pearson(ps: list[tuple[float, float]]) -> float | None:
    """Pearson r over a list of (actual, predicted) pairs; None if <2 pairs or zero variance.

    Factored out so the bias-contrast path (which builds its own pairs via
    ``contrast_pairs``) scores identically to the absolute-rate path."""
    if len(ps) < 2:
        return None
    xs = [a for a, _ in ps]
    ys = [p for _, p in ps]
    mx, my = mean(xs), mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sqrt(sxx * syy)


def baseline_mae(targets: dict[str, Any], const: float,
                 keys: set[str] | None = None) -> float | None:
    """MAE of a predictor that always outputs ``const``; None if no actual rates."""
    actuals = [t["rate"] for k, t in targets.items()
               if t.get("rate") is not None and (keys is None or k in keys)]
    return mean(abs(const - a) for a in actuals) if actuals else None


def contrast_pairs(score_contrasts: dict[str, tuple[float, float]],
                   keys: set[str] | None = None) -> list[tuple[float, float]]:
    """Turn an eval's ``score_contrasts`` output ({contrast_key: (actual, predicted)}) into the
    same (actual, predicted) list ``pearson``/``mae`` consume, optionally restricted to ``keys``.
    A contrast belongs to its split unit, so ``keys`` is the set of in-split contrast keys."""
    return [v for k, v in score_contrasts.items()
            if v[0] is not None and v[1] is not None and (keys is None or k in keys)]


# --- spec-aware scoring (used by the report builders so they match the leaderboard) ------------
# For a "bias_contrast" eval the unit scored is the signed group-vs-baseline gap, NOT the
# per-condition rate; everywhere we report a correlation/MAE we must therefore route through the
# eval's ``score_contrasts``. These helpers pick the right path from ``spec.scoring_semantics``.

def scored_pairs(spec, targets: dict[str, Any], preds: dict[str, float | None],
                 keys: set[str] | None = None) -> list[tuple[float, float]]:
    """(actual, predicted) pairs as ``spec`` is scored: per-contrast gaps for a ``bias_contrast``
    eval, else per-condition rates. ``keys`` (in-split condition keys) restricts the scope."""
    if getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast":
        return contrast_pairs(spec.score_contrasts(targets, preds, keys=keys))
    return pairs(targets, preds, keys)


def corr_spec(spec, targets: dict[str, Any], preds: dict[str, float | None],
              keys: set[str] | None = None) -> float | None:
    """Pearson r the way this eval is scored (contrast gaps for bias evals); ``None`` if undefined."""
    return pearson(scored_pairs(spec, targets, preds, keys))


def mae_spec(spec, targets: dict[str, Any], preds: dict[str, float | None],
             keys: set[str] | None = None) -> float | None:
    """MAE the way this eval is scored (per-contrast gap error for bias evals); ``None`` if empty."""
    ps = scored_pairs(spec, targets, preds, keys)
    return mean(abs(p - a) for a, p in ps) if ps else None


def baseline_mae_spec(spec, targets: dict[str, Any], const: float,
                      keys: set[str] | None = None) -> float | None:
    """MAE of the trivial baseline. For a ``bias_contrast`` eval that is *predict-zero-bias*
    (``const`` ignored — every contrast predicted as a 0-point gap); else the constant-rate
    predictor that always answers ``const``."""
    if getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast":
        actuals = [a for a, _ in scored_pairs(spec, targets,
                                              {k: t.get("rate") for k, t in targets.items()}, keys)]
        return mean(abs(a) for a in actuals) if actuals else None
    return baseline_mae(targets, const, keys)


def bootstrap_ci(ps: list[tuple[float, float]],
                 stat: Callable[[list[tuple[float, float]]], float | None],
                 *, n: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float] | None:
    """Percentile bootstrap CI for ``stat`` over (actual, predicted) pairs.

    Resamples pairs with replacement ``n`` times; returns the (lower, upper) percentile bounds,
    or None if there are <2 pairs or every resample is degenerate. Undefined ``stat`` values
    (e.g. constant-prediction correlation) are coerced to 0.0, matching the leaderboard's
    treatment of flat predictors."""
    import random

    if len(ps) < 2:
        return None
    rng = random.Random(seed)
    m = len(ps)
    vals: list[float] = []
    for _ in range(n):
        sample = [ps[rng.randrange(m)] for _ in range(m)]
        s = stat(sample)
        vals.append(0.0 if s is None else s)
    if not vals:
        return None
    vals.sort()
    lo = vals[int(alpha / 2 * n)]
    hi = vals[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi
