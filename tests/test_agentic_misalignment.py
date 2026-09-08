"""agentic_misalignment as the held-out TEST eval: split seam, oracle items, method wiring,
and the donor-frozen tuning path (2026-08-15)."""

from __future__ import annotations

import pytest

from behavior_prediction import splits
from behavior_prediction.evals import get_spec
from behavior_prediction.evals.agentic_misalignment import (
    GOAL_VALUES, SCENARIOS, AgenticMisalignmentSpec,
)
from behavior_prediction.methods import DEFAULT_METHODS, get_method


@pytest.fixture
def spec() -> AgenticMisalignmentSpec:
    return get_spec("agentic_misalignment")


def test_registry_and_sweep(spec):
    assert spec.name == "agentic_misalignment"
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    assert len(conds) == len(SCENARIOS) * len(GOAL_VALUES) == 24
    assert len({spec.condition_key(c) for c in conds}) == 24


def test_heldout_split_seam(spec):
    assert spec.test_only is True
    units = spec.split_units()
    assert units == SCENARIOS
    assert all(spec.forced_split(u) == "test" for u in units)
    # a built manifest places every unit in test regardless of fractions
    manifest = splits.build_manifest(spec, units)
    assert set(manifest["assignments"].values()) == {"test"}
    assert sorted(manifest["forced"]["test"]) == sorted(units)
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    assert {spec.split_unit(c) for c in conds} == set(units)


def test_oracle_items_one_prompt_per_condition(spec):
    cond = spec.make_condition("blackmail")
    view = spec.oracle_items(cond, 20)
    assert view["protocol"]
    (item,) = view["items"]                       # single fixed prompt carries the condition
    assert item["item_id"] == "prompt"
    assert set(item["exhibit"]) == {"SYSTEM PROMPT", "USER PROMPT", "EMAIL CONTENT"}
    assert all(item["exhibit"].values())
    assert "Out of 100" in item["ask"]
    # the ask names the scenario's concrete target action
    assert "threaten to reveal" in spec.oracle_items(
        spec.make_condition("blackmail"), 1)["items"][0]["ask"]
    assert "override code" in spec.oracle_items(
        spec.make_condition("murder"), 1)["items"][0]["ask"]


def test_method_applicability_matches_other_benchmarks(spec):
    applies = {m: get_method(m).applies_to(spec) for m in DEFAULT_METHODS}
    # stateless suite runs (oracle methods unlocked by oracle_items)
    for m in ("self_report", "generic_report", "value", "pairwise", "behavioral_sampling",
              "informed_oracle", "generic_oracle", "oracle_pairwise", "cross_model_mean",
              "report_mean", "oracle_report_mean", "informed_sampling"):
        assert applies[m], m
    # trained methods have nothing to fit on (test_only) and must skip
    for m in ("llm_prediction", "few_shot", "few_shot_other"):
        assert not applies[m], m


def test_value_grid_pins_concrete(spec):
    grid = spec.method_grid(get_method("value"))
    assert [get_method("value").method_id(hp) for hp in grid] == ["value-concrete"]


def test_run_behavior_resolves_shortcuts_and_reasoning(spec, capsys):
    """The behavioral driver resolves models.yaml shortcuts (full string + reasoning config,
    passed to inspect_eval as GenerateConfig kwargs) and keys the log dir by model slug, so
    off/low variants sharing one provider string never mix. Dry-run only — no inspect imports."""
    import argparse

    def dry_run(argv):
        p = argparse.ArgumentParser()
        spec.add_run_args(p)
        assert spec.run_behavior(p.parse_args(argv + ["--dry-run"])) == 0
        return capsys.readouterr().out

    out = dry_run(["--model", "claude-sonnet-5-low"])
    assert "openrouter/anthropic/claude-sonnet-5" in out
    assert "'reasoning_effort': 'low'" in out
    assert "logs/agentic_misalignment/claude-sonnet-5-low" in out

    out = dry_run(["--model", "claude-sonnet-5-off"])          # extra_body passthrough shape
    assert "'enabled': False" in out
    assert "logs/agentic_misalignment/claude-sonnet-5-off" in out

    out = dry_run([])                                          # raw provider string: no config
    assert "Reasoning:" not in out
    # distinct slugs -> distinct default log dirs for variants of one provider string
    assert (spec._default_log_dir("claude-sonnet-5-off")
            != spec._default_log_dir("claude-sonnet-5-low"))


def test_donor_tuning_is_preregistered(spec, monkeypatch, tmp_path):
    """The donor path picks each method's modal donor best (valid on this eval), with no
    scores and no dependency on any agentic_misalignment prediction."""
    from behavior_prediction import common, tuning

    donor_doc = {"methods": {"self_report": {"best_setting": "self_report"},
                             "pairwise": {"best_setting": "pairwise-indiff"},
                             "value": {"best_setting": "value-aggregate"},
                             "informed_oracle": {"best_setting": "informed_oracle"}}}
    other_doc = {"methods": {"self_report": {"best_setting": "self_report-honest"},
                             "pairwise": {"best_setting": "pairwise-indiff"}}}
    monkeypatch.setattr(tuning, "ACTIVE_EVALS", ["donor_a", "donor_b"])
    monkeypatch.setattr(tuning, "tuning_path", lambda ev: tmp_path / f"{ev}.json")
    common.save_json(donor_doc, str(tmp_path / "donor_a.json"))
    common.save_json(other_doc, str(tmp_path / "donor_b.json"))

    manifest = splits.build_manifest(spec, spec.split_units())
    doc = tuning._donor_tuning(spec, manifest, None)
    methods = doc["methods"]
    assert doc["split"] == "test" and doc["donor_evals"] == ["donor_a", "donor_b"]
    # modal vote wins (pairwise-indiff 2-0); a 1-1 tie falls back to grid order (default first)
    assert methods["pairwise"]["best_setting"] == "pairwise-indiff"
    assert methods["self_report"]["best_setting"] == "self_report"
    # a donor best that is invalid on this eval (value-aggregate is pruned here) is ignored
    assert methods["value"]["best_setting"] == "value-concrete"
    # no donor vote at all -> the grid's default setting
    assert methods["behavioral_sampling"]["best_setting"] == "behavioral_sampling"
    # trained methods are absent entirely (applies_to gates them on a test_only eval)
    assert "llm_prediction" not in methods and "few_shot" not in methods
    assert all(m["criterion"] == "donor_modal_best" for m in methods.values())
    assert all(s["agg_r"] is None for m in methods.values() for s in m["settings"].values())
