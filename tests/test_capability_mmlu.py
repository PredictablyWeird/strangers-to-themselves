"""capability_mmlu: condition wiring, no-value frame, subject split universe."""
from __future__ import annotations

from behavior_prediction.evals import REGISTRY
from behavior_prediction.evals.base import Frame


def test_registered():
    assert "capability_mmlu" in REGISTRY
    assert REGISTRY["capability_mmlu"].name == "capability_mmlu"


def test_no_value_framing_flag():
    assert REGISTRY["capability_mmlu"].has_value_framing is False
    assert REGISTRY["capability_mmlu"].test_only is False


def test_condition_key_is_subject():
    spec = REGISTRY["capability_mmlu"]
    cond = spec.make_condition("anatomy")
    assert spec.condition_key(cond) == "anatomy"
    assert spec.split_unit(cond) == "anatomy"


def test_frame_is_subject_level_and_actionful():
    spec = REGISTRY["capability_mmlu"]
    f = spec.frame(spec.make_condition("anatomy"))
    assert isinstance(f, Frame)
    assert "anatomy" in f.setting + f.self_report_situation
    assert "correct" in f.target_action.lower()
    # no value pair
    assert f.driver_value == "" and f.brake_value == ""


def test_split_units_are_the_57_subjects():
    spec = REGISTRY["capability_mmlu"]
    assert spec.split_units() == sorted(spec.split_units())
    assert len(spec.split_units()) == 57
