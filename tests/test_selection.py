"""Selection-CV mechanics (B) and dev-only generation scope, over the committed discrimeval data."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_prediction import common, selection, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import RunConfig
from behavior_prediction.methods.base import MethodAdapter
from behavior_prediction.cli import benchmark

_TARGETS = "results/discrimeval/llama-3.3-70b/targets.json"


@pytest.fixture
def discrim():
    if not Path(_TARGETS).exists() or not Path("results/discrimeval/splits.json").exists():
        pytest.skip("committed discrimeval targets/splits not present")
    spec = get_spec("discrimeval")
    targets = common.load_json(_TARGETS)["targets"]
    manifest = splits.load_manifest("discrimeval")
    return spec, targets, manifest


def test_selection_cv_picks_the_oracle(discrim):
    spec, targets, manifest = discrim
    dev_keys = splits.split_keys(spec, targets, manifest, "dev")
    oracle = {k: t.get("rate") for k, t in targets.items()}   # = actual behavior
    flat = {k: 0.5 for k in targets}                          # constant -> carries no order signal
    methods = {"oracle": oracle, "flat": flat}

    A = selection.per_setting_dev(spec, targets, methods, dev_keys)
    assert abs(A["oracle"] - 1.0) < 1e-6 and A["flat"] == 0.0

    B = selection.selection_cv(spec, targets, methods, manifest, candidates=["oracle", "flat"])
    assert B is not None and B["n_folds"] >= 2
    assert set(B["picks"]) == {"oracle"}                      # selects the good setting every fold
    assert B["pooled_r"] <= A["oracle"] + 1e-9                # never beats the oracle whole-dev corr
    assert B["pooled_r"] > 0.99


def test_generation_scope_is_dev_only_unless_test_requested(discrim):
    spec, targets, _committed = discrim
    # Synthetic manifest so both splits are non-empty regardless of the committed data: assign the
    # first half of the split units to test, the rest to dev.
    units = sorted({spec.split_unit(t["condition"]) for t in targets.values()})
    assert len(units) >= 2
    cut = len(units) // 2
    manifest = {"seed": 1234,
                "assignments": {u: ("test" if i < cut else "dev") for i, u in enumerate(units)}}
    dev_keys = splits.split_keys(spec, targets, manifest, "dev")
    test_keys = splits.split_keys(spec, targets, manifest, "test")
    assert dev_keys and test_keys

    class Recorder(MethodAdapter):
        base_method = "rec"
        trained = False

        def predict(self, spec, model, conds, hp, cfg, fit_state=None):
            return {spec.condition_key(c): {"predicted_rate": 0.5, "condition": c} for c in conds}

    rec = Recorder()
    dev_only = benchmark._predict_method(rec, spec, None, targets, manifest, {},
                                         RunConfig(), include_test=False)
    with_test = benchmark._predict_method(rec, spec, None, targets, manifest, {},
                                          RunConfig(), include_test=True)
    assert set(dev_only) == set(dev_keys)
    assert set(with_test) == set(dev_keys) | set(test_keys)
    assert set(dev_only).isdisjoint(test_keys)               # test stays frozen by default
