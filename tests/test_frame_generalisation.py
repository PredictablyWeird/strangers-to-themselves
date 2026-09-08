"""Frame slot rename + new EvalSpec flags are wired through every existing eval."""
from __future__ import annotations

import pytest

from behavior_prediction.evals import REGISTRY
from behavior_prediction.evals.base import EvalSpec, Frame


def test_frame_uses_generic_action_slots():
    f = Frame(setting="s", target_action="do X", action_frame="X-ish")
    assert f.target_action == "do X"
    assert f.action_frame == "X-ish"
    # value slots are now optional with empty defaults
    assert f.driver_value == "" and f.brake_value == ""
    assert not hasattr(f, "harmful_action")
    assert not hasattr(f, "harm_frame")


def test_evalspec_flag_defaults():
    assert EvalSpec.has_value_framing is True
    assert EvalSpec.test_only is False


@pytest.mark.parametrize("name", ["agentic_misalignment", "propensitybench", "discrimeval"])
def test_existing_specs_build_a_populated_frame(name):
    spec = REGISTRY[name]
    cond = spec.make_condition(spec.scenarios[0])
    f = spec.frame(cond)
    assert f.setting and f.target_action and f.action_frame
    # the three harm evals keep value framing
    assert spec.has_value_framing is True
    # agentic_misalignment is the held-out test eval (2026-08-15); the others have real splits
    assert spec.test_only is (name == "agentic_misalignment")
