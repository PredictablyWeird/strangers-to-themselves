"""cross_model_mean: predict a condition's rate as the other models' mean measured rate.

Fully offline: fake targets docs are written under a temp BP_RESULTS_DIR and the method reads
them straight off disk (it queries no model).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY, RunConfig, valid_combo
from behavior_prediction.methods.trained import CrossModelMean
from behavior_prediction.models_adapter import ModelAdapter

EVAL = "sycophancy_pushback"  # condition_key == scenario; no network needed for the spec


def _write_targets(root: Path, slug: str, model: str, rates: dict[str, float | None],
                   reasoning: dict | None = None) -> None:
    targets = {k: {"rate": r, "scenario": k, "condition": {"scenario": k}}
               for k, r in rates.items()}
    doc = {"model": model, "model_shortcut": None, "reasoning": reasoning, "targets": targets}
    out = root / EVAL / slug / "targets.json"
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(doc))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("BP_RESULTS_DIR", str(tmp_path))
    # These tests exercise the LEGACY discovery mode (glob every stored targets doc): the fixed
    # donor_pool (methods.yaml, 2026-08-07) would read the real pool models, which don't exist
    # in this fake tree. The fixed-pool path is covered by test_fixed_donor_pool_is_honored.
    from behavior_prediction import common
    monkeypatch.setattr(common, "donor_pool", lambda: [])
    spec = get_spec(EVAL)
    target_model = ModelAdapter("prov/under-test")  # unknown shortcut -> passes through verbatim
    # The model under test itself, in two reasoning variants (same provider string) — both must
    # be excluded from its own prediction.
    _write_targets(tmp_path, "under-test", target_model.full, {"anatomy": 0.9, "virology": 0.9})
    _write_targets(tmp_path, "under-test-low", target_model.full, {"anatomy": 0.9},
                   reasoning={"effort": "low"})
    # Two other models; other-b additionally has a reasoning variant whose rates must be
    # averaged into ONE contribution (0.3), not double-counted.
    _write_targets(tmp_path, "other-a", "prov/other-a", {"anatomy": 0.2, "virology": 0.6})
    _write_targets(tmp_path, "other-b", "prov/other-b", {"anatomy": 0.4})
    _write_targets(tmp_path, "other-b-low", "prov/other-b", {"anatomy": 0.2},
                   reasoning={"effort": "low"})
    return spec, target_model


def _conds(*scenarios):
    return [{"scenario": s} for s in scenarios]


def test_registered_untrained_and_applies_everywhere():
    m = REGISTRY["cross_model_mean"]
    assert isinstance(m, CrossModelMean)
    # Never uses the target model's labels -> no CV needed, and no capability requirements.
    assert m.trained is False
    assert not m.capabilities.needs_activations and not m.capabilities.needs_logprobs
    assert valid_combo(m, ModelAdapter("llama-3.3-70b"), get_spec(EVAL))


def test_means_others_excluding_self_and_deduping_reasoning_variants(setup):
    spec, model = setup
    out = CrossModelMean().predict(spec, model, _conds("anatomy", "virology"), {}, RunConfig())
    # anatomy: other-a 0.2 + other-b mean(0.4, 0.2)=0.3 -> 0.25. Self (0.9, both variants) excluded.
    assert out["anatomy"]["predicted_rate"] == pytest.approx(0.25)
    assert out["anatomy"]["n"] == 2
    assert out["anatomy"]["contributors"] == ["prov/other-a", "prov/other-b"]
    # virology: only other-a measured it.
    assert out["virology"]["predicted_rate"] == pytest.approx(0.6)
    assert out["virology"]["n"] == 1


def test_unmeasured_condition_and_min_models_gate(setup):
    spec, model = setup
    out = CrossModelMean().predict(spec, model, _conds("astronomy"), {}, RunConfig())
    assert out["astronomy"]["predicted_rate"] is None and out["astronomy"]["n"] == 0
    strict = CrossModelMean(min_models=2).predict(spec, model, _conds("anatomy", "virology"),
                                                  {}, RunConfig())
    assert strict["anatomy"]["predicted_rate"] == pytest.approx(0.25)  # 2 contributors: kept
    assert strict["virology"]["predicted_rate"] is None               # 1 contributor: gated
    assert strict["virology"]["n"] == 1                               # count still reported


def test_fixed_donor_pool_is_honored(setup, monkeypatch):
    """With methods.yaml's donor_pool declared, ONLY the declared donors contribute — a stored
    doc outside the pool (other-b here) must not sneak in, and a declared-but-unmeasured donor
    is simply absent."""
    from behavior_prediction import common
    spec, model = setup
    # donor names resolve through model_slug ("prov/other-a" -> "prov_other-a"), so the doc
    # must live at the slug path — unlike the glob-mode docs above.
    root = Path(common.results_root())
    _write_targets(root, common.model_slug("prov/other-a2"), "prov/other-a",
                   {"anatomy": 0.2})
    monkeypatch.setattr(common, "donor_pool", lambda: ["prov/other-a2", "prov/missing-donor"])
    out = CrossModelMean().predict(spec, model, _conds("anatomy"), {}, RunConfig())
    assert out["anatomy"]["predicted_rate"] == pytest.approx(0.2)
    assert out["anatomy"]["contributors"] == ["prov/other-a"]
    assert out["anatomy"]["n"] == 1
