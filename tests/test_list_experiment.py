"""The list_experiment method: randomised neutral subsets, shared control, count differencing.
All model calls are mocked, so these stay offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_prediction import common, elicitation
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY

_TARGETS = "results/sycophancy_pushback/llama-3.3-70b/targets.json"


@pytest.fixture
def syco():
    if not Path(_TARGETS).exists():
        pytest.skip("committed sycophancy targets not present")
    spec = get_spec("sycophancy_pushback")
    targets = common.load_json(_TARGETS)["targets"]
    return spec, targets


def test_applies_to_value_framed_absolute_rate_evals_only():
    m = REGISTRY["list_experiment"]
    assert m.applies_to(get_spec("sycophancy_pushback"))
    assert m.applies_to(get_spec("propensitybench"))
    assert not m.applies_to(get_spec("discrimeval"))      # bias_contrast: no list form
    assert not m.applies_to(get_spec("capability_mmlu"))  # no value framing: not a willingness question


def test_list_draws_are_deterministic_and_vary_by_variant(syco):
    spec, targets = syco
    cond = next(iter(targets.values()))["condition"]
    a = elicitation._list_items(spec, cond, {}, treatment=True, variant=0)
    b = elicitation._list_items(spec, cond, {}, treatment=True, variant=0)
    c = elicitation._list_items(spec, cond, {}, treatment=True, variant=1)
    assert a == b                       # stable across processes/resume
    assert a != c                       # fresh draw per run
    assert len(a) == 5                  # list_size neutrals + the sensitive item
    sensitive = elicitation._list_sensitive_item(spec, cond)
    assert sensitive in a and sensitive in c
    # Control draws are condition-independent (shared across conditions).
    ctrl = elicitation._list_items(spec, None, {}, treatment=False, variant=3)
    assert len(ctrl) == 4 and sensitive not in ctrl
    assert all(i in elicitation.NEUTRAL_POOL for i in ctrl)


def test_predict_shares_one_control_and_differences_counts(syco, monkeypatch):
    spec, targets = syco
    conds = [t["condition"] for t in list(targets.values())[:3]]
    calls = []

    def fake_elicit(model_name, prompts, **kw):
        calls.append(prompts)
        # control keys answer 2, treatment keys answer 3 -> every condition's rate is 1.0
        return {k: {"predicted_rate": None, "n": 1,
                    "samples": [3.0 if "##t" in k else 2.0], "raw": [""], "parse_failures": 0}
                for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    out = elicitation.list_experiment_predict(spec, "mock/model", conds, runs=4)

    ctrl_prompts, treat_prompts = calls
    assert all(k.startswith("__control__##c") for k in ctrl_prompts)
    assert len(ctrl_prompts) == elicitation._control_runs(4)   # one shared control arm
    assert len(treat_prompts) == len(conds) * 4                # one key per (condition, run)
    for c in conds:
        e = out[spec.condition_key(c)]
        assert e["control_count"] == 2.0 and e["treatment_count"] == 3.0
        assert e["predicted_rate"] == 1.0 and e["n"] == 4
