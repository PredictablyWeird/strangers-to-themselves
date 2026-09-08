"""Noise-ceiling estimate, the reliability filter, and ceiling-normalization.

Fully offline: synthetic ``(rate, n)`` targets scored through a real spec's semantics.
"""

from __future__ import annotations

from behavior_prediction import ceilings
from behavior_prediction.evals import get_spec

SPEC = get_spec("sycophancy_pushback")  # absolute_rate semantics; no network for the spec


def _targets(rates: dict[str, float], n: int) -> dict[str, dict]:
    return {k: {"rate": r, "n": n, "scenario": k, "condition": {"scenario": k}}
            for k, r in rates.items()}


def test_spread_and_large_n_is_reliable() -> None:
    """A real spread of rates measured on many items -> high ceiling, passes the filter."""
    rates = {f"c{i}": v for i, v in enumerate([0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 0.5])}
    est = ceilings.estimate(SPEC, _targets(rates, n=300), keys=None)
    assert est is not None
    assert est.ceiling is not None and est.ceiling > 0.9
    assert est.ceiling_lo > 0.0
    assert ceilings.is_reliable(est)


def test_no_spread_is_unreliable() -> None:
    """Every condition at the same rate -> no rank signal in the ground truth -> filtered out."""
    est = ceilings.estimate(SPEC, _targets({f"c{i}": 0.5 for i in range(8)}, n=50), keys=None)
    # Either undefined (constant replicates) or a ceiling whose CI floor sits at 0 — not reliable.
    assert not ceilings.is_reliable(est)


def test_small_n_widens_ci_toward_unreliable() -> None:
    """Same spread but few items per condition -> the CI lower bound collapses toward 0."""
    rates = {f"c{i}": v for i, v in enumerate([0.45, 0.5, 0.55, 0.5, 0.48, 0.52])}
    tiny = ceilings.estimate(SPEC, _targets(rates, n=4), keys=None)
    assert not ceilings.is_reliable(tiny)


def test_too_few_conditions_returns_none() -> None:
    assert ceilings.estimate(SPEC, _targets({"a": 0.2, "b": 0.8}, n=100), keys=None) is None


def test_normalize_is_r_over_ceiling() -> None:
    rates = {f"c{i}": v for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.8, 0.4])}
    est = ceilings.estimate(SPEC, _targets(rates, n=300), keys=None)
    assert ceilings.normalize(0.4, est) == 0.4 / est.ceiling
    # Missing inputs pass through as None rather than raising.
    assert ceilings.normalize(None, est) is None
    assert ceilings.normalize(0.4, None) is None


def test_min_ceiling_threshold_tightens_filter() -> None:
    """Raising the threshold rejects a cell whose ceiling is real but modest. Since 2026-08-15
    the criterion is the ceiling POINT estimate (a high-ceiling cell with a wide CI is kept)."""
    rates = {f"c{i}": v for i, v in enumerate([0.35, 0.45, 0.5, 0.55, 0.65, 0.5, 0.48, 0.6])}
    est = ceilings.estimate(SPEC, _targets(rates, n=40), keys=None)
    assert ceilings.is_reliable(est, min_ceiling=0.0)
    assert not ceilings.is_reliable(est, min_ceiling=0.99)


def test_estimate_is_reproducible() -> None:
    rates = {f"c{i}": v for i, v in enumerate([0.1, 0.3, 0.5, 0.7, 0.9, 0.2])}
    t = _targets(rates, n=100)
    assert ceilings.estimate(SPEC, t, keys=None).rel == ceilings.estimate(SPEC, t, keys=None).rel
