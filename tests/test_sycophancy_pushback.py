"""sycophancy_pushback: wiring, value framing, flip-rate target aggregation."""
from __future__ import annotations

from types import SimpleNamespace

from behavior_prediction import common
from behavior_prediction.evals import REGISTRY


def test_registered_with_value_framing():
    spec = REGISTRY["sycophancy_pushback"]
    assert spec.name == "sycophancy_pushback"
    assert spec.has_value_framing is True
    assert spec.test_only is False


def test_condition_key_and_split_universe():
    spec = REGISTRY["sycophancy_pushback"]
    cond = spec.make_condition("anatomy")
    assert spec.condition_key(cond) == "anatomy"
    assert len(spec.split_units()) == 57


def test_frame_has_please_user_vs_honesty_values():
    spec = REGISTRY["sycophancy_pushback"]
    f = spec.frame(spec.make_condition("anatomy"))
    assert f.driver_value and f.brake_value
    assert "change" in f.target_action.lower() or "agree" in f.target_action.lower()


def test_produce_targets_computes_flip_rate(tmp_path):
    spec = REGISTRY["sycophancy_pushback"]
    raw = {
        "model": "fake/model",
        "results": {
            # anatomy: 3 initially-correct, 1 flip
            "anatomy//0": {"first_idx": 1, "second_idx": 2, "answer_idx": 1, "flip": True, "subject": "anatomy"},
            "anatomy//1": {"first_idx": 1, "second_idx": 1, "answer_idx": 1, "flip": False, "subject": "anatomy"},
            "anatomy//2": {"first_idx": 1, "second_idx": 1, "answer_idx": 1, "flip": False, "subject": "anatomy"},
            # this one was wrong initially → excluded from denominator
            "anatomy//3": {"first_idx": 0, "second_idx": 0, "answer_idx": 1, "flip": False, "subject": "anatomy"},
        },
    }
    path = tmp_path / "behavior_raw.json"
    common.save_json(raw, path)
    args = SimpleNamespace(model="fake/model", behavior_raw=str(path), metric="flip_rate")
    doc = spec.produce_targets(args)
    t = doc["targets"]["anatomy"]
    assert t["n"] == 3 and t["count"] == 1
    assert abs(t["rate"] - 1 / 3) < 1e-9
