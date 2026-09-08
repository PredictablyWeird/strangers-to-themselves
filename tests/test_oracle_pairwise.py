"""Pairwise informed oracle: verbatim item exhibits, asked as a forced choice -> win rate.

Offline: the single pooled LLM call (``common.elicit_rates``) is mocked; the unit keying, item
selection, both-orders prompting, verdict parsing and win aggregation all run for real.
"""

from __future__ import annotations

import pytest

from behavior_prediction import common, elicitation
from behavior_prediction.evals import discrimeval as D
from behavior_prediction.evals import get_spec
from behavior_prediction.evals.base import EvalSpec
from behavior_prediction.methods import DEFAULT_METHODS, REGISTRY, RunConfig
from behavior_prediction.methods.elicited import OraclePairwise


class _OracleSpec(EvalSpec):
    """Minimal absolute-rate spec exposing oracle_items (needs no network)."""
    name = "oracle_toy"

    def condition_key(self, cond):
        return cond["scenario"]

    def oracle_items(self, cond, k):
        return {"protocol": "PROTO.", "items": [
            {"item_id": f"i{i}", "exhibit": {"Q": f"item {i} of {cond['scenario']}"},
             "ask": f"ask {i}?"} for i in range(k)]}


class _PlainSpec(EvalSpec):
    name = "plain_toy"

    def condition_key(self, cond):
        return cond["scenario"]


class _Model:
    full = "prov/under-test"
    reasoning_cfg = None


def _conds(*scenarios):
    return [{"scenario": s} for s in scenarios]


# --- 1. registration / applicability ------------------------------------------

def test_registered_with_grid_and_applicability():
    m = REGISTRY["oracle_pairwise"]
    assert isinstance(m, OraclePairwise) and m.trained is False
    assert "oracle_pairwise" in DEFAULT_METHODS
    assert m.capabilities.needs_all_conditions is True
    assert m.default_runs % 2 == 0                        # both orders need an even split
    assert [m.method_id(hp) for hp in m.hyperparameter_grid()] == ["oracle_pairwise"]
    # Same gate as informed_oracle: the eval must expose its measured items...
    assert m.applies_to(_OracleSpec()) is True
    assert m.applies_to(_PlainSpec()) is False
    assert m.applies_to(get_spec("agentic_misalignment")) is True   # oracle_items 2026-08-15
    # ...including the bias eval, where plain `pairwise` does not apply.
    assert m.applies_to(get_spec("discrimeval")) is True
    assert REGISTRY["pairwise"].applies_to(get_spec("discrimeval")) is False


def test_k_resolution():
    assert OraclePairwise()._k({}) == 20                  # methods.yaml default
    assert OraclePairwise(k_items=3)._k({}) == 3          # --method-config override


# --- 2. verdict tokens ---------------------------------------------------------

def test_parse_pair_choice_xy_tokens():
    xy = elicitation.ORACLE_PAIR_TOKENS
    assert common.parse_pair_choice("X", options=xy) == 1.0
    assert common.parse_pair_choice("**Y**", options=xy) == 0.0
    assert common.parse_pair_choice("EQUAL", options=xy) == 0.5
    # The whole point of X/Y: A/B tokens appear INSIDE the exhibits (DiscrimEval's two applicant
    # VERSIONs), and must not be read as verdicts.
    assert common.parse_pair_choice("VERSION B looks harsher, so my answer is Y",
                                    options=xy) == 0.0
    assert common.parse_pair_choice("Version A vs VERSION B", options=xy) is None
    # The A/B default is untouched for `pairwise`.
    assert common.parse_pair_choice("B") == 0.0


# --- 3. prompt rendering -------------------------------------------------------

def test_prompt_shows_both_items_verbatim_with_their_asks():
    x = {"item_id": "1", "exhibit": {"TURN 1": "VERBATIM-X"}, "ask": "ASK-X?"}
    y = {"item_id": "2", "exhibit": {"TURN 1": "VERBATIM-Y"}, "ask": "ASK-Y?"}
    p = elicitation.oracle_pairwise_prompt("PROTOCOL.", x, y)
    assert "PROTOCOL." in p
    assert "VERBATIM-X" in p and "VERBATIM-Y" in p       # both items shown, verbatim
    assert "QUESTION FOR ITEM X: ASK-X?" in p            # each side keeps its own ask
    assert "QUESTION FOR ITEM Y: ASK-Y?" in p
    assert "NOT to carry out" in p                       # predict, don't perform
    assert '"X" or "Y"' in p and "EQUAL" not in p
    assert 'EQUAL' in elicitation.oracle_pairwise_prompt("P.", x, y, indifference=True)


def test_example_prompt_exhibit(monkeypatch):
    from behavior_prediction.evals import mmlu_data
    q = mmlu_data.Question(subject="anatomy", stem="S?", choices=["a", "b", "c", "d"],
                           answer_idx=0)
    monkeypatch.setattr(mmlu_data, "load_questions", lambda subj, n: [q][:n])
    spec = get_spec("capability_mmlu")
    ex = elicitation.example_prompt_exhibit("oracle_pairwise", spec, {"scenario": "anatomy"}, {})
    assert ex is not None
    prompt = ex["prompt"]
    assert mmlu_data.mcq_prompt(q) in prompt              # real item on side X
    assert "another unit's measured item" in prompt       # placeholder opponent on side Y


# --- 4. aggregation ------------------------------------------------------------

def _rank_model(order):
    """Fake elicit_rates whose verdict is driven by a hidden per-unit ranking (higher wins)."""
    def fake_elicit(model_name, prompts, **kw):
        out = {}
        for fkey in prompts:                             # "<ki>##vs##<kj>##r<r>##o<orient>"
            ki, kj = fkey.split("##vs##")[0], fkey.split("##vs##")[1].split("##")[0]
            first, second = (ki, kj) if fkey.endswith("o0") else (kj, ki)
            win = 1.0 if order[first] > order[second] else 0.0
            out[fkey] = {"predicted_rate": win, "n": 1, "samples": [win], "raw": [""],
                         "reasoning": [None], "parse_failures": 0}
        return out
    return fake_elicit


def test_win_rate_recovers_the_ranking(monkeypatch):
    order = {"a": 3, "b": 2, "c": 1, "d": 0}
    seen = {}

    def fake(model_name, prompts, **kw):
        seen["prompts"] = prompts
        seen["runs"] = kw.get("runs")
        return _rank_model(order)(model_name, prompts, **kw)

    monkeypatch.setattr(common, "elicit_rates", fake)
    out = OraclePairwise(k_items=2, rounds=4).predict(
        _OracleSpec(), _Model(), _conds(*order), {}, RunConfig())

    assert seen["runs"] == 1                             # default_runs=2 -> one sample per order
    assert all(f.endswith(("o0", "o1")) for f in seen["prompts"])   # both orders of every matchup
    # A consistent ranker's win rates reproduce the ranking (position bias cancels across orders).
    assert out["a"]["predicted_rate"] == 1.0 and out["d"]["predicted_rate"] == 0.0
    assert out["a"]["predicted_rate"] > out["b"]["predicted_rate"] \
        > out["c"]["predicted_rate"] > out["d"]["predicted_rate"]
    assert out["a"]["comparisons"] == 4 and out["a"]["n"] == 8      # 4 rounds x 2 orders
    # Items cycle across rounds, so one item can't drive a unit's whole win rate.
    assert out["a"]["items_used"] == ["i0", "i1"]


def test_odd_runs_rejected():
    with pytest.raises(ValueError, match="even"):
        OraclePairwise().predict(_OracleSpec(), _Model(), _conds("a", "b"), {},
                                 RunConfig(runs=3))


# --- 5. bias_contrast eval (DiscrimEval) ---------------------------------------

def test_discrimeval_units_are_contrasts_and_canonical_items(monkeypatch):
    grp = dict(D.BASELINE_PERSON); grp["race"] = "Black"
    idx = {}
    for demo, tag in ((D.BASELINE_PERSON, "BASE"), (grp, "GRP")):
        for qid in D.templates_for_category("personal_lending"):
            idx[(qid, float(demo["age"]), demo["gender"], demo["race"])] = f"{tag} q{qid}"
    monkeypatch.setitem(D._EXAMPLE_INDEX, "explicit", idx)
    spec = get_spec("discrimeval")
    spec.config = "explicit"
    conds = [D.make_condition("personal_lending", axis="baseline"),
             D.make_condition("personal_lending", axis="race", value="Black")]

    units = elicitation.oracle_units(spec, conds, 4)
    assert set(units) == {"personal_lending/race/Black-vs-white"}   # contrast-keyed, baseline out
    unit = units["personal_lending/race/Black-vs-white"]
    # Only the canonical (sign +1) order of each template: both sides list the baseline applicant
    # first, so applicant-position bias is common to both sides of the comparison.
    assert all(it["sign"] == 1 for it in unit["items"])
    assert all(it["item_id"].endswith("-o0") for it in unit["items"])
    # Each item still shows BOTH demographic fills, and asks for the signed gap.
    ex = unit["items"][0]["exhibit"]
    assert len(ex) == 2 and "favor" in unit["items"][0]["ask"]


def test_discrimeval_predicts_contrast_keys(monkeypatch):
    idx = {}
    demos = [D.BASELINE_PERSON] + [dict(D.BASELINE_PERSON, race=r) for r in ("Black", "Asian")]
    for demo in demos:
        for qid in D.templates_for_category("personal_lending"):
            idx[(qid, float(demo["age"]), demo["gender"], demo["race"])] = f"q{qid}"
    monkeypatch.setitem(D._EXAMPLE_INDEX, "explicit", idx)
    spec = get_spec("discrimeval")
    spec.config = "explicit"
    conds = [D.make_condition("personal_lending", axis="baseline"),
             D.make_condition("personal_lending", axis="race", value="Black"),
             D.make_condition("personal_lending", axis="race", value="Asian")]
    order = {"personal_lending/race/Black-vs-white": 1,
             "personal_lending/race/Asian-vs-white": 0}
    monkeypatch.setattr(common, "elicit_rates", _rank_model(order))

    out = OraclePairwise(k_items=2, rounds=2).predict(spec, _Model(), conds, {}, RunConfig())
    assert set(out) == set(order)                        # contrast keys; baseline cell dropped
    assert out["personal_lending/race/Black-vs-white"]["predicted_rate"] == 1.0
    assert out["personal_lending/race/Asian-vs-white"]["predicted_rate"] == 0.0
    # score_contrasts consumes a contrast-keyed prediction directly (as for informed_oracle).
    targets = {
        "personal_lending/baseline": {"rate": 0.5, "condition": conds[0]},
        "personal_lending/race/Black": {"rate": 0.3, "condition": conds[1]},
        "personal_lending/race/Asian": {"rate": 0.6, "condition": conds[2]},
    }
    preds = {k: v["predicted_rate"] for k, v in out.items()}
    scored = spec.score_contrasts(targets, preds, keys=set(targets))
    assert scored["personal_lending/race/Black-vs-white"] == (pytest.approx(-0.2), 1.0)
