"""Broken-eval gating and default-method selection."""

from __future__ import annotations

import pytest

from behavior_prediction.evals import REGISTRY as EVALS, get_spec, require_runnable
from behavior_prediction.methods import DEFAULT_METHODS, REGISTRY as METHODS


def test_broken_flags():
    assert get_spec("discrimeval").broken is False
    assert get_spec("propensitybench").broken is False
    assert get_spec("propensitybench_benign").broken is False
    assert get_spec("agentic_misalignment").broken is False
    # every broken eval explains why
    for s in EVALS.values():
        if s.broken:
            assert s.broken_reason


def test_require_runnable_gate(monkeypatch):
    broken = get_spec("propensitybench")
    ok = get_spec("agentic_misalignment")
    monkeypatch.setattr(broken, "broken", True)     # simulate a broken eval to exercise the gate
    with pytest.raises(SystemExit):
        require_runnable([broken], force=False)
    require_runnable([broken], force=True)          # forced -> allowed
    require_runnable([ok], force=False)             # non-broken -> allowed
    require_runnable([ok, broken], force=True)      # mixed, forced -> allowed
    with pytest.raises(SystemExit):
        require_runnable([ok, broken], force=False)  # any broken member gates


def test_train_scenario_mean_not_default_but_registered():
    assert "train_scenario_mean" in METHODS          # still runnable by name / tests
    assert "train_scenario_mean" not in DEFAULT_METHODS
    assert "self_report" in DEFAULT_METHODS and "llm_prediction" in DEFAULT_METHODS
