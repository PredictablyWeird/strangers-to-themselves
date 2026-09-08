"""Pairwise comparative elicitation: forced-choice comparisons -> per-condition win rate.

Offline: the single pooled LLM call (``common.elicit_rates``) is mocked; everything else —
pair construction, both-orders prompting, verdict parsing, win aggregation — runs for real.
"""

from __future__ import annotations

import pytest

from behavior_prediction import common, elicitation
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY, RunConfig
from behavior_prediction.methods.elicited import Pairwise

EVAL = "sycophancy_pushback"  # absolute_rate eval whose spec needs no network


def _conds(*scenarios):
    return [{"scenario": s} for s in scenarios]


class _Model:
    full = "prov/under-test"
    reasoning_cfg = None


def test_registered_with_grid_and_applicability():
    m = REGISTRY["pairwise"]
    assert isinstance(m, Pairwise) and m.trained is False
    assert m.capabilities.needs_all_conditions is True
    assert m.default_runs % 2 == 0                       # both orders need an even split
    ids = [m.method_id(hp) for hp in m.hyperparameter_grid()]
    assert ids == ["pairwise", "pairwise-indiff"]
    assert m.applies_to(get_spec(EVAL))
    assert not m.applies_to(get_spec("discrimeval"))     # bias_contrast evals are skipped


def test_parse_pair_choice():
    assert common.parse_pair_choice("A") == 1.0
    assert common.parse_pair_choice("**B**") == 0.0
    assert common.parse_pair_choice("EQUAL") == 0.5
    assert common.parse_pair_choice("equal") == 0.5
    # Last verdict wins: prose mentioning A must not shadow the final answer.
    assert common.parse_pair_choice("Situation A is tense, but honestly...\nB") == 0.0
    assert common.parse_pair_choice("Maybe A. No — they are EQUAL") == 0.5
    # Lower-case articles are prose, not verdicts.
    assert common.parse_pair_choice("in a situation like this, hard to say") is None
    assert common.parse_pair_choice("") is None
    assert common.parse_pair_choice(None) is None


def test_pairs_deterministic_and_cover_all_keys():
    keys = ["k1", "k2", "k3", "k4"]
    pairs = elicitation._pairwise_pairs(keys, rounds=3, seed=7)
    assert pairs == elicitation._pairwise_pairs(list(reversed(keys)), rounds=3, seed=7)  # order-free
    assert len(pairs) == 3 * 2                       # even count: every key paired every round
    from collections import Counter
    appearances = Counter(k for _, i, j in pairs for k in (i, j))
    assert appearances == {k: 3 for k in keys}
    # Odd count: exactly one key sits out per round.
    odd = elicitation._pairwise_pairs(keys + ["k5"], rounds=4, seed=7)
    assert len(odd) == 4 * 2
    assert elicitation._pairwise_pairs(keys, rounds=3, seed=8) != pairs  # seed moves the draw


def test_predict_aggregates_wins_over_both_orders(monkeypatch):
    spec = get_spec(EVAL)
    conds = _conds("anatomy", "astronomy", "virology")
    seen = {}

    def fake_elicit(model_name, prompts, **kw):
        seen["prompts"] = prompts
        seen["runs"] = kw.get("runs")
        assert kw.get("parse_fn") is common.parse_pair_choice
        out = {}
        for fkey in prompts:                         # "<ki>##vs##<kj>##r<r>##o<orient>"
            ki, kj = fkey.split("##vs##")[0], fkey.split("##vs##")[1].split("##")[0]
            first, second = (ki, kj) if fkey.endswith("o0") else (kj, ki)
            win = 1.0 if first < second else 0.0     # "model" prefers the alphabetically-first key
            out[fkey] = {"predicted_rate": win, "n": 1, "samples": [win], "raw": [""],
                         "reasoning": [None], "parse_failures": 0}
        return out

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    out = Pairwise(rounds=4).predict(spec, _Model(), conds, {}, RunConfig())

    assert seen["runs"] == 1                         # default_runs=2 -> one sample per order
    # Both orders of every matchup were asked.
    assert all(f.endswith(("o0", "o1")) for f in seen["prompts"])
    # A perfectly transitive preference anatomy > astronomy > virology must come out ranked:
    # anatomy beats every opponent it faces, virology loses to every one (astronomy's exact win
    # rate depends on which opponents the seeded matchings dealt it — 3 conds: one sits out/round).
    assert out["anatomy"]["predicted_rate"] == 1.0
    assert out["virology"]["predicted_rate"] == 0.0
    assert (out["anatomy"]["predicted_rate"] >= out["astronomy"]["predicted_rate"]
            >= out["virology"]["predicted_rate"])
    for e in out.values():
        assert e["n"] == 2 * e["comparisons"]        # two orders per matchup, one sample each


def test_indifference_hyperparameter_changes_prompt_and_scores_half(monkeypatch):
    spec = get_spec(EVAL)
    conds = _conds("anatomy", "astronomy")
    captured = {}

    def fake_elicit(model_name, prompts, **kw):
        captured["prompts"] = list(prompts.values())
        return {f: {"predicted_rate": 0.5, "n": 1, "samples": [0.5], "raw": ["EQUAL"],
                    "reasoning": [None], "parse_failures": 0} for f in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    m = Pairwise(rounds=2)
    out = m.predict(spec, _Model(), conds, {"indifference": True}, RunConfig())
    assert all('"EQUAL"' in p for p in captured["prompts"])
    # EQUAL is worth 0.5 to both sides, in either order.
    assert all(e["predicted_rate"] == 0.5 for e in out.values())

    m.predict(spec, _Model(), conds, {}, RunConfig())
    assert all('"EQUAL"' not in p for p in captured["prompts"])  # default: forced choice


def test_odd_runs_rejected_and_single_condition_yields_none(monkeypatch):
    spec = get_spec(EVAL)
    with pytest.raises(ValueError, match="even"):
        Pairwise().predict(spec, _Model(), _conds("anatomy", "virology"), {}, RunConfig(runs=3))
    monkeypatch.setattr(common, "elicit_rates",
                        lambda model_name, prompts, **kw: {})
    out = Pairwise().predict(spec, _Model(), _conds("anatomy"), {}, RunConfig())
    assert out["anatomy"]["predicted_rate"] is None  # nothing to compare against


def test_example_prompt_exhibit_renders_a_pair():
    spec = get_spec(EVAL)
    ex = elicitation.example_prompt_exhibit("pairwise", spec, {"scenario": "anatomy"},
                                            {"indifference": True})
    assert ex and "SITUATION A" in ex["prompt"] and "SITUATION B" in ex["prompt"]
    assert '"EQUAL"' in ex["prompt"]
