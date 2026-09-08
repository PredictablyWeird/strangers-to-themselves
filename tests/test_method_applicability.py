"""Value methods skip evals without value framing; trained methods skip test-only evals."""
from __future__ import annotations

from behavior_prediction.evals.base import EvalSpec
from behavior_prediction.methods import REGISTRY


class _NoValueSpec(EvalSpec):
    name = "novalue"
    has_value_framing = False


class _TestOnlySpec(EvalSpec):
    name = "testonly"
    test_only = True


class _PlainSpec(EvalSpec):
    name = "plain"


def test_value_methods_skip_no_value_eval():
    assert REGISTRY["value"].applies_to(_NoValueSpec()) is False
    assert REGISTRY["value"].applies_to(_PlainSpec()) is True


def test_self_report_still_applies_everywhere():
    assert REGISTRY["self_report"].applies_to(_NoValueSpec()) is True
    assert REGISTRY["self_report"].applies_to(_TestOnlySpec()) is True


def test_trained_method_skips_test_only_eval():
    assert REGISTRY["train_scenario_mean"].applies_to(_TestOnlySpec()) is False
    assert REGISTRY["train_scenario_mean"].applies_to(_PlainSpec()) is True


def test_behavioral_sampling_applies_regardless_of_value_framing():
    # 2026-08-09: the value-framing gate was dropped — sampling builds its generation prompt
    # from the Frame's setting/situation/action fields only, so value-less evals (MMLU) apply.
    assert REGISTRY["behavioral_sampling"].applies_to(_NoValueSpec()) is True
    assert REGISTRY["behavioral_sampling"].applies_to(_PlainSpec()) is True
