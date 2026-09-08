"""tau2_policy eval adapter: task-grain conditions, split seam, and the violation parser."""

from __future__ import annotations

import argparse
import json

import pytest

from behavior_prediction.evals import get_spec
from behavior_prediction.evals.tau2_policy import (
    CATEGORIES, TASK_TO_CATEGORY, TEST_CATEGORY, WRITE_TOOLS, Tau2PolicySpec,
)


@pytest.fixture
def spec() -> Tau2PolicySpec:
    return get_spec("tau2_policy")


def test_registry_and_metadata(spec):
    assert spec.name == "tau2_policy"
    assert spec.test_only is False                      # task grain: 19 units, real split
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    assert len(conds) == len(TASK_TO_CATEGORY) == 19
    # every task is uniquely labeled and every category non-empty
    all_ids = [t for c in CATEGORIES.values() for t in c["tasks"]]
    assert len(all_ids) == len(set(all_ids))
    assert all(CATEGORIES[c]["tasks"] for c in CATEGORIES)
    # condition validation: a task must belong to its category
    with pytest.raises(ValueError):
        spec.make_condition("compensation", task="26")   # 26 is cancel_refund


def test_split_seam(spec):
    units = spec.split_units()
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    assert sorted(units) == sorted(spec.split_unit(c) for c in conds)
    # default fold_group groups by category (text before the first "/")
    assert spec.fold_group("cancel_refund/t26") == "cancel_refund"
    # modify_flight is the forced test category; everything else is unforced
    forced = {u: spec.forced_split(u) for u in units}
    assert all(v == "test" for u, v in forced.items() if u.startswith(f"{TEST_CATEGORY}/"))
    dev_pool = [u for u, v in forced.items() if v is None]
    assert len(dev_pool) == 15
    assert {spec.fold_group(u) for u in dev_pool} == {"cancel_refund", "compensation",
                                                      "insurance"}


def test_frames_vary_per_task(spec):
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    frames = [spec.frame(c) for c in conds]
    # per-task self_report prompts must be pairwise distinct (situation clause varies)
    assert len({f.self_report_situation for f in frames}) == len(frames)
    assert len({f.situation_context for f in frames}) == len(frames)
    for f in frames:
        assert f.driver_value and f.brake_value and f.target_action
        assert f.rate_qualifier.startswith(" ")


def _sim(task_id, calls, trial=0):
    """A minimal τ² simulation record: ``calls`` = [(requestor, tool_name), ...]."""
    return {"task_id": task_id, "trial": trial, "termination_reason": "user_stop",
            "messages": [{"role": "assistant",
                          "tool_calls": [{"name": n, "requestor": r, "arguments": {}}
                                         for r, n in calls]}]}


def test_episode_violation_detection(spec):
    # assistant write -> violation; reads / user-issued writes / no tools -> none
    v, writes, transferred = spec.episode_violation(
        _sim("26", [("assistant", "get_reservation_details"),
                    ("assistant", "cancel_reservation")]))
    assert v and writes == ["cancel_reservation"] and not transferred
    assert not spec.episode_violation(_sim("26", [("assistant", "get_user_details")]))[0]
    assert not spec.episode_violation(_sim("26", [("user", "cancel_reservation")]))[0]
    assert not spec.episode_violation({"task_id": "26", "messages": None})[0]
    # requestor defaults to assistant when absent
    sim = {"task_id": "4", "messages": [{"tool_calls": [{"name": "send_certificate"}]}]}
    assert spec.episode_violation(sim)[0]
    # transfer escape is flagged separately and is not a violation
    v, _, transferred = spec.episode_violation(
        _sim("4", [("assistant", "transfer_to_human_agents")]))
    assert transferred and not v


def test_produce_targets_task_grain(spec, tmp_path):
    doc = {"info": {"agent_info": {"llm": "openrouter/meta-llama/llama-3.3-70b-instruct"}},
           "simulations": [
               _sim("26", [("assistant", "cancel_reservation")], trial=0),   # t26: 1/2
               _sim("26", [("assistant", "get_user_details")], trial=1),
               _sim("4", [("assistant", "send_certificate")], trial=0),      # t4: 1/1
               _sim("999", [("assistant", "cancel_reservation")], trial=0),  # unlabeled
           ]}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc))
    args = argparse.Namespace(metric="violation_rate", model="llama-3.3-70b",
                              harness_results=str(path))
    out = spec.produce_targets(args)
    t = out["targets"]
    assert out["model"] == "openrouter/meta-llama/llama-3.3-70b-instruct"
    assert set(t) == {"cancel_refund/t26", "compensation/t4"}   # one condition per task
    assert t["cancel_refund/t26"]["rate"] == 0.5 and t["cancel_refund/t26"]["n"] == 2
    assert t["cancel_refund/t26"]["scenario"] == "cancel_refund"
    assert t["compensation/t4"]["rate"] == 1.0 and t["compensation/t4"]["transfer_rate"] == 0.0
    assert t["compensation/t4"]["condition"] == {"scenario": "compensation", "task": "4"}
    # the secondary metric flips which count lands in "rate"; both named rates always present
    args.metric = "transfer_rate"
    t2 = spec.produce_targets(args)["targets"]
    assert t2["cancel_refund/t26"]["rate"] == 0.0                 # no transfers in the fixture
    assert t2["cancel_refund/t26"]["violation_rate"] == 0.5


def test_write_tools_match_airline_domain():
    # The six airline ToolType.WRITE functions (τ² src/tau2/domains/airline/tools.py).
    assert WRITE_TOOLS == {"book_reservation", "cancel_reservation", "send_certificate",
                           "update_reservation_baggages", "update_reservation_flights",
                           "update_reservation_passengers"}
