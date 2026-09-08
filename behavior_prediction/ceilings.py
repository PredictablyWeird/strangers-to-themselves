#!/usr/bin/env python3
"""Per-(eval, model) noise ceiling: the largest correlation *any* predictor could reach against
the measured dev targets, given the targets' own sampling noise.

Two uses downstream:

- **Filter weak cells.** Where the ground truth carries no reliably-measurable spread, no method
  can be validated and a cell's correlation is noise. :func:`is_reliable` drops such cells so they
  do not pollute the leaderboard (both the primary mean-r and the secondary normalized metric).
- **Ceiling-normalized score.** :func:`normalize` expresses a cell's correlation as the *fraction
  of reachable signal captured*, ``r / ceiling`` — a +0.40 against a 0.94 ceiling and a +0.40
  against a 0.50 ceiling are very different achievements, and raw r hides that.

Estimator (committed-data only). For each condition we know its measured ``rate`` and item count
``n`` from ``targets.json``; ``behavior_raw.json`` (the per-item outcomes needed for an
assumption-free split-half) is gitignored and often absent, so we use a **parametric bootstrap**:
draw two independent Binomial replicate rate-vectors ``Binom(n, rate)/n`` per rep and take their
Pearson r across conditions as the reliability of one replicate. For a ``bias_contrast`` eval the
replicates are routed through ``spec.score_contrasts`` first, so discrimeval is scored at the gap
grain, matching the leaderboard. ``ceiling = sqrt(rel)`` (a perfect predictor of the *true* rates
correlates ``sqrt(rel)`` with one noisy measurement). A percentile CI over reps gives the lower
bound the reliability filter keys on.

This matches the assumption-free split-half ceiling (over ``behavior_raw`` per-item outcomes, as
in ``scripts/noise_ceiling.py``) to within ~0.05 on every pool cell where both are computable, and
needs no raw file. The parametric form treats items within a condition as i.i.d. Bernoulli, so it
slightly *over*-estimates reliability where items are heterogeneous (mainly capability_mmlu); the
gap is small and the estimate stays conservative for the filter's purpose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean
from typing import Any

from behavior_prediction import metrics

CEILING_REPS = 300
CEILING_SEED = 1234
MIN_CONDITIONS = 3  # too few conditions -> correlation is undefined / uninformative


@dataclass(frozen=True)
class Ceiling:
    """Parametric noise-ceiling estimate for one (eval, model) cell on a fixed split."""
    rel: float             # mean reliability over reps (Pearson r of two Binomial replicates)
    rel_lo: float          # 2.5th percentile of the per-rep reliability
    rel_hi: float          # 97.5th percentile
    ceiling: float | None  # sqrt(rel) if rel > 0 else None (undefined ceiling)
    ceiling_lo: float      # sqrt(max(rel_lo, 0)); the lower bound the reliability filter uses
    n_conditions: int      # conditions that entered the estimate
    method: str = "parametric"


def estimate(spec, targets: dict[str, Any], keys: set[str] | None, *,
             reps: int = CEILING_REPS, seed: int = CEILING_SEED) -> Ceiling | None:
    """Estimate the noise ceiling of ``targets`` over ``keys`` (in-split condition keys), scored the
    way ``spec`` is scored. Returns ``None`` when fewer than :data:`MIN_CONDITIONS` conditions have
    a usable ``(rate, n)`` — the ceiling is then undefined and the cell should be treated as weak."""
    conds = {k: t for k, t in targets.items()
             if (keys is None or k in keys) and t.get("rate") is not None and t.get("n")}
    if len(conds) < MIN_CONDITIONS:
        return None
    bias = getattr(spec, "scoring_semantics", "absolute_rate") == "bias_contrast"
    rng = random.Random(seed)

    def replicate() -> dict[str, float]:
        return {k: sum(1 for _ in range(t["n"]) if rng.random() < t["rate"]) / t["n"]
                for k, t in conds.items()}

    rs: list[float] = []
    for _ in range(reps):
        a, b = replicate(), replicate()
        if bias:
            ta = {k: {**targets[k], "rate": a[k]} for k in a}
            r = metrics.pearson(metrics.contrast_pairs(spec.score_contrasts(ta, b, keys=keys)))
        else:
            r = metrics.pearson([(a[k], b[k]) for k in conds])
        if r is not None:
            rs.append(r)
    if len(rs) < 2:
        return None
    rs.sort()
    rel = mean(rs)
    lo = rs[int(0.025 * len(rs))]
    hi = rs[min(len(rs) - 1, int(0.975 * len(rs)))]
    return Ceiling(
        rel=rel, rel_lo=lo, rel_hi=hi,
        ceiling=(rel ** 0.5 if rel > 0 else None),
        ceiling_lo=(lo ** 0.5 if lo > 0 else 0.0),
        n_conditions=len(conds),
    )


#: A cell is scored only when at least half of its target variance is reliable signal:
#: reliability >= 0.5, i.e. ceiling >= sqrt(0.5) ~= 0.707. The 0.5 anchor is the classical
#: test-theory boundary where signal stops outweighing measurement error — a grounded
#: threshold rather than a free constant (2026-08-16). The point estimate (not the CI
#: lower bound) is compared, so small-condition cells with clean targets are kept; the
#: earlier lower-CI-bound rule silently acted as a condition-count filter.
MIN_RELIABILITY = 0.5


def is_reliable(c: Ceiling | None, min_ceiling: float = MIN_RELIABILITY ** 0.5) -> bool:
    """Is this cell's ground truth reliable enough to validate a predictor? True when the ceiling
    POINT estimate reaches ``min_ceiling`` (default sqrt(0.5): at least half the target variance
    is reliable signal — see :data:`MIN_RELIABILITY`). This is a property of the *target*, not of
    any method, so applying it as a filter is decided before looking at any method's score."""
    return c is not None and c.ceiling is not None and c.ceiling >= min_ceiling


def normalize(r: float | None, c: Ceiling | None) -> float | None:
    """Ceiling-normalized correlation ``r / ceiling`` — the fraction of reachable signal captured.
    ``None`` when either input is missing or the ceiling is undefined. Values can exceed 1.0 when a
    method's cell r overshoots the noisy ceiling estimate; callers may clip if a bounded score is
    wanted, but we keep it raw so the overshoot stays visible."""
    if r is None or c is None or not c.ceiling:
        return None
    return r / c.ceiling


def as_record(c: Ceiling | None) -> dict[str, Any] | None:
    """JSON-serializable view for the exported results doc; ``None`` passes through."""
    if c is None:
        return None
    return {"ceiling": c.ceiling, "ceiling_lo": c.ceiling_lo, "rel": c.rel,
            "n_conditions": c.n_conditions, "method": c.method}
