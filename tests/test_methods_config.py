"""methods.yaml-driven grids and method ids: back-compat with the pre-refactor on-disk ids."""

from __future__ import annotations

from behavior_prediction import common
from behavior_prediction.methods import REGISTRY


def _ids(base: str) -> list[str]:
    m = REGISTRY[base]
    return [m.method_id(hp) for hp in m.hyperparameter_grid()]


def test_self_report_settings():
    assert _ids("self_report") == ["self_report", "self_report-honest"]


def test_merged_value_grid_ids():
    """The merged `value` method's sweep was retired 2026-07-24: the global grid is narrowed to
    the tuned winners plus the all-default `value` (OFAT always contains {}). Per-eval pinning
    (EvalSpec.method_grid) then reduces this to exactly one setting per eval — see
    test_eval_aware_grid_prunes_value_knobs_on_discrimeval_only."""
    ids = _ids("value")
    assert ids == ["value", "value-aggregate", "value-concrete"]


def test_sweep_strategy_cross_vs_ofat():
    """`methods.<name>.sweep` controls expansion: cross_product (default) is the cartesian product;
    one_at_a_time is the all-default setting plus each non-default value with others at default."""
    base = {"args": {"a": [0, 1], "b": ["x", "y", "z"]}}
    cross = common.method_arg_grid("m", {"methods": {"m": {**base, "sweep": "cross_product"}}})
    ofat = common.method_arg_grid("m", {"methods": {"m": {**base, "sweep": "one_at_a_time"}}})
    assert len(cross) == 2 * 3                      # full product
    assert ofat == [{}, {"a": 1}, {"b": "y"}, {"b": "z"}]   # 1 + (1) + (2)
    # default (no sweep key) == cross_product
    assert common.method_arg_grid("m", {"methods": {"m": base}}) == cross


def test_unknown_sweep_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        common.method_arg_grid("m", {"methods": {"m": {"args": {"a": [0, 1]}, "sweep": "bogus"}}})


def test_eval_aware_grid_prunes_value_knobs_on_discrimeval_only():
    """Per-eval value settings are PINNED (sweep retired 2026-07-24): `value-aggregate` on
    DiscrimEval (its value-scale prompt ignores `context`), `value-concrete` everywhere the
    concrete per-condition context renders distinct prompts (every other eval)."""
    from behavior_prediction.evals import get_spec
    disc, pb = get_spec("discrimeval"), get_spec("propensitybench")
    m = REGISTRY["value"]
    assert [m.method_id(h) for h in disc.method_grid(m)] == ["value-aggregate"]   # pinned
    assert [m.method_id(h) for h in pb.method_grid(m)] == ["value-concrete"]      # pinned
    for base in ("behavioral_sampling", "train_scenario_mean"):
        assert _ids(base) == [base]
    # llm_prediction sweeps only description_mode; calibration is always applied and the analyst
    # runs once at temperature 0 (a runs sweep was dropped — cost without consistent gain).
    assert _ids("llm_prediction") == ["llm_prediction", "llm_prediction-full"]
    # On DiscrimEval llm_prediction works at the contrast grain, where a contrast has one canonical
    # description — description_mode is pruned, leaving the single default setting.
    lp = REGISTRY["llm_prediction"]
    assert [lp.method_id(h) for h in disc.method_grid(lp)] == ["llm_prediction"]
    assert [lp.method_id(h) for h in pb.method_grid(lp)] == _ids("llm_prediction")


def test_grid_dicts_are_minimal():
    """Default values are omitted, so the default setting is {} and its id == base_method."""
    grid = REGISTRY["self_report"].hyperparameter_grid()
    assert {} in grid
    assert {"honesty_nudge": True} in grid


def test_method_id_default_equals_base():
    assert common.method_id("self_report", {}) == "self_report"
    # A default value passed explicitly still yields no suffix.
    assert common.method_id("value", {"aggregate": False, "context": "none"}) == "value"


def test_legacy_fallback_for_unknown_method():
    """A base method absent from methods.yaml falls back to the legacy suffix logic."""
    assert common.method_id("not_in_yaml", {"detail": "concrete"}) == "not_in_yaml-concrete"
    assert common.method_id("not_in_yaml", {"honesty_nudge": True}) == "not_in_yaml-honest"


def test_selection_pool_declared():
    pool = common.selection_pool()
    assert pool and all(isinstance(m, str) for m in pool)


def test_tuned_only_restricts_grid_to_best_setting():
    """--tuned-only collapses a method's full grid to just its tuned best setting; a method with no
    tuned best is skipped (None). This is what keeps the final test run to the chosen settings."""
    from behavior_prediction.cli.benchmark import _tuned_grid
    from behavior_prediction.evals import get_spec
    spec = get_spec("discrimeval")
    sr = REGISTRY["self_report"]
    best = {"self_report": "self_report-honest"}
    restricted = [sr.method_id(hp) for hp in _tuned_grid(sr, spec, best)]
    assert restricted == ["self_report-honest"]
    assert _tuned_grid(sr, spec, {}) is None          # no tuned best -> skip
    assert [sr.method_id(hp) for hp in _tuned_grid(sr, spec, None)] == [
        "self_report", "self_report-honest"]          # None -> full grid
