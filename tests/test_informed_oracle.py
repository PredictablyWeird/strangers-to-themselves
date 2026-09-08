"""Offline tests for the informed_oracle upper-bound method."""
from __future__ import annotations

import pytest

from behavior_prediction import common, elicitation
from behavior_prediction.evals import mmlu_data
from behavior_prediction.evals import discrimeval as D
from behavior_prediction.evals.base import EvalSpec
from behavior_prediction.evals.sycophancy import pushback
from behavior_prediction.methods import REGISTRY, DEFAULT_METHODS
from behavior_prediction.methods.base import RunConfig
from behavior_prediction.methods.elicited import InformedOracle


# --- fakes -------------------------------------------------------------------

class _OracleSpec(EvalSpec):
    """A minimal spec overriding oracle_items (to exercise applies_to True)."""
    name = "oracle_toy"

    def condition_key(self, cond):
        return cond["scenario"]

    def oracle_items(self, cond, k):
        return {"protocol": "PROTO.", "items": [
            {"item_id": str(i), "exhibit": {"Q": f"item {i} of {cond['scenario']}"},
             "ask": "ASK?"} for i in range(k)]}


class _PlainSpec(EvalSpec):
    name = "plain_toy"

    def condition_key(self, cond):
        return cond["scenario"]


class _FakeModel:
    def __init__(self, reasoning_cfg=None):
        self.full = "prov/fake"
        self.reasoning_cfg = reasoning_cfg or {}


# --- 1. applies_to gating (incl. PB exclusion) --------------------------------

def test_registered_in_defaults():
    assert "informed_oracle" in DEFAULT_METHODS
    assert isinstance(REGISTRY["informed_oracle"], InformedOracle)


@pytest.mark.parametrize("ev", ["sycophancy_pushback", "capability_mmlu",
                                "discrimeval", "discrimeval_implicit",
                                "propensitybench", "propensitybench_benign",
                                "agentic_misalignment"])   # oracle_items added 2026-08-15
def test_applies_to_supported_evals(ev):
    from behavior_prediction.evals import get_spec
    assert REGISTRY["informed_oracle"].applies_to(get_spec(ev)) is True


def test_applies_gate_is_hook_override():
    assert REGISTRY["informed_oracle"].applies_to(_PlainSpec()) is False
    assert REGISTRY["informed_oracle"].applies_to(_OracleSpec()) is True


# --- 2. verbatim prompt rendering ---------------------------------------------

def test_sycophancy_oracle_prompt_verbatim(monkeypatch):
    from behavior_prediction.evals import get_spec
    q = mmlu_data.Question(subject="anatomy", stem="Largest organ?",
                           choices=["Heart", "Skin", "Liver", "Brain"], answer_idx=1)
    monkeypatch.setattr(mmlu_data, "load_questions", lambda subj, n: [q][:n])
    spec = get_spec("sycophancy_pushback")
    view = spec.oracle_items({"scenario": "anatomy"}, 5)
    assert len(view["items"]) == 1
    it = view["items"][0]
    prompt = elicitation.informed_oracle_prompt(view["protocol"], it["exhibit"], it["ask"])
    assert mmlu_data.mcq_prompt(q) in prompt
    wrong = pushback.wrong_letter_idx(q.answer_idx)
    assert pushback.pushback_text(q, wrong) in prompt
    assert "your first answer to this question was B" in prompt  # correct letter named
    assert "no longer be B" in prompt
    assert "PREDICTION: <integer 0-100>" in prompt
    assert "NOT to carry out" in prompt


def test_capability_oracle_prompt_verbatim(monkeypatch):
    from behavior_prediction.evals import get_spec
    q = mmlu_data.Question(subject="anatomy", stem="Largest organ?",
                           choices=["Heart", "Skin", "Liver", "Brain"], answer_idx=1)
    monkeypatch.setattr(mmlu_data, "load_questions", lambda subj, n: [q, q][:n])
    spec = get_spec("capability_mmlu")
    view = spec.oracle_items({"scenario": "anatomy"}, 2)
    assert [i["item_id"] for i in view["items"]] == ["0", "1"]
    prompt = elicitation.informed_oracle_prompt(
        view["protocol"], view["items"][0]["exhibit"], view["items"][0]["ask"])
    assert mmlu_data.mcq_prompt(q) in prompt
    assert "answer it\ncorrectly" in prompt or "answer it correctly" in prompt
    assert "PREDICTION:" in prompt


def test_discrimeval_bias_oracle_shows_both_versions(monkeypatch):
    from behavior_prediction.evals import get_spec
    # Seed the template index for the baseline and the Black group at every qid.
    grp = dict(D.BASELINE_PERSON); grp["race"] = "Black"
    idx = {}
    for demo, tag in ((D.BASELINE_PERSON, "BASE"), (grp, "GRP")):
        for qid in D.templates_for_category("personal_lending"):
            idx[(qid, float(demo["age"]), demo["gender"], demo["race"])] = f"{tag} template q{qid}"
    monkeypatch.setitem(D._EXAMPLE_INDEX, "explicit", idx)
    # templates_for_cell enumerates all qids for a cell, in order
    assert [q for q, _ in D.templates_for_cell("personal_lending", grp, "explicit")] \
        == D.templates_for_category("personal_lending")

    spec = get_spec("discrimeval")
    spec.config = "explicit"
    n_t = len(D.templates_for_category("personal_lending"))
    # baseline cell -> no contrast, no items
    assert spec.oracle_items(D.make_condition("personal_lending", axis="baseline"), 20)["items"] == []
    # group cell -> TWO items per template (both applicant orders), each showing BOTH fills
    cond = D.make_condition("personal_lending", axis="race", value="Black")
    view = spec.oracle_items(cond, 20)
    assert len(view["items"]) == 2 * n_t
    o0, o1 = view["items"][0], view["items"][1]
    assert o0["item_id"].endswith("-o0") and o0["sign"] == 1
    assert o1["item_id"].endswith("-o1") and o1["sign"] == -1
    # o0: A = baseline (white man), B = group (Black man); o1 flips A/B
    la0, lb0 = list(o0["exhibit"])
    assert "white man" in la0 and "Black man" in lb0
    assert o0["exhibit"][la0].startswith("BASE") and o0["exhibit"][la0].endswith(D.YES_NO_SUFFIX)
    assert o0["exhibit"][lb0].startswith("GRP")
    la1, lb1 = list(o1["exhibit"])
    assert "Black man" in la1 and "white man" in lb1                        # swapped
    assert "differing only in the applicant" in view["protocol"]
    assert "favor" in o0["ask"]


def test_discrimeval_oracle_predicts_signed_contrast(monkeypatch):
    """The adapter keys bias predictions by contrast_key with a signed gap, and skips baseline."""
    from behavior_prediction.evals import get_spec
    grp = dict(D.BASELINE_PERSON); grp["race"] = "Black"
    idx = {}
    for demo in (D.BASELINE_PERSON, grp):
        for qid in D.templates_for_category("personal_lending"):
            idx[(qid, float(demo["age"]), demo["gender"], demo["race"])] = f"t{qid}"
    monkeypatch.setitem(D._EXAMPLE_INDEX, "explicit", idx)
    spec = get_spec("discrimeval")
    spec.config = "explicit"
    conds = [D.make_condition("personal_lending", axis="baseline"),
             D.make_condition("personal_lending", axis="race", value="Black")]

    seen = {}

    def _cap(model_name, prompts, **kw):
        seen["parse_fn"] = kw["parse_fn"]
        seen["keys"] = set(prompts)
        # Simulate a model whose true gap is −0.2 (favors baseline) PLUS a constant +0.1 position
        # bias toward whichever applicant is shown as VERSION B. It therefore answers:
        #   o0 (B=group):    −0.2 + 0.1 = −0.1
        #   o1 (B=baseline): +0.2 + 0.1 = +0.3   (its raw answer is baseline−group)
        # After sign correction (o1 ×−1) the two give −0.1 and −0.3; the mean, −0.2, recovers the
        # true gap and cancels the position bias.
        out = {}
        for kk in prompts:
            out[kk] = {"predicted_rate": -0.1 if kk.endswith("-o0") else 0.3,
                       "raw": ["x"], "reasoning": [""], "parse_failures": 0}
        return out

    monkeypatch.setattr(common, "elicit_rates", _cap)
    out = InformedOracle().predict(spec, _FakeModel(), conds, {"k_items": 20}, RunConfig())
    ckey = "personal_lending/race/Black-vs-white"
    assert set(out) == {ckey}                              # contrast-keyed, baseline skipped
    assert out[ckey]["predicted_rate"] == pytest.approx(-0.2)   # true gap recovered
    assert seen["parse_fn"] is common.parse_signed_prediction_tag
    # both applicant orders were asked, all under the contrast key
    assert all(k.startswith(ckey + "//") for k in seen["keys"])
    assert any(k.endswith("-o0") for k in seen["keys"]) and any(k.endswith("-o1") for k in seen["keys"])


# --- 3. aggregation math ------------------------------------------------------

def _canned(model_name, prompts, **kw):
    # one item per condition returns None (parse failure), the rest a fixed rate
    out = {}
    for i, key in enumerate(sorted(prompts)):
        rate = None if key.endswith("//0") else 0.4
        out[key] = {"predicted_rate": rate, "n": 0 if rate is None else 1,
                    "samples": [] if rate is None else [rate],
                    "raw": ["r"], "reasoning": [""],
                    "parse_failures": 1 if rate is None else 0}
    return out


def test_aggregation_mean_over_parsed(monkeypatch):
    monkeypatch.setattr(common, "elicit_rates", _canned)
    m = InformedOracle()
    conds = [{"scenario": "a"}, {"scenario": "b"}]
    cfg = RunConfig(concurrency=2)
    # k=3: items 0 (None),1,2 -> mean over the two parsed 0.4s
    out = m.predict(_OracleSpec(), _FakeModel(), conds, {"k_items": 3}, cfg)
    assert out["a"]["predicted_rate"] == pytest.approx(0.4)
    assert out["a"]["n"] == 2 and out["a"]["k_items"] == 3
    assert out["a"]["item_rates"] == {"0": None, "1": 0.4, "2": 0.4}
    assert out["a"]["parse_failures"] == 1


def test_all_items_fail_gives_none(monkeypatch):
    monkeypatch.setattr(common, "elicit_rates",
                        lambda model_name, prompts, **kw: {
                            k: {"predicted_rate": None, "raw": [], "reasoning": [],
                                "parse_failures": 1} for k in prompts})
    m = InformedOracle()
    out = m.predict(_OracleSpec(), _FakeModel(), [{"scenario": "a"}], {"k_items": 2},
                    RunConfig())
    assert out["a"]["predicted_rate"] is None and out["a"]["n"] == 0


def test_prompt_keys_are_condition_item(monkeypatch):
    seen = {}

    def _capture(model_name, prompts, **kw):
        seen.update(prompts)
        return {k: {"predicted_rate": 0.5, "raw": ["x"], "reasoning": [""],
                    "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", _capture)
    InformedOracle().predict(_OracleSpec(), _FakeModel(), [{"scenario": "a"}],
                             {"k_items": 2}, RunConfig())
    assert set(seen) == {"a//0", "a//1"}


# --- 4. temperature guard -----------------------------------------------------

def _capture_temp(monkeypatch):
    grabbed = {}

    def _cap(model_name, prompts, **kw):
        grabbed["temperature"] = kw["temperature"]
        return {k: {"predicted_rate": 0.5, "raw": ["x"], "reasoning": [""],
                    "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", _cap)
    return grabbed


def test_greedy_temperature_for_non_reasoning(monkeypatch):
    grabbed = _capture_temp(monkeypatch)
    InformedOracle().predict(_OracleSpec(), _FakeModel(reasoning_cfg={}),
                             [{"scenario": "a"}], {"k_items": 1},
                             RunConfig(temperature=1.0))
    assert grabbed["temperature"] == 0.0


def test_default_temperature_for_reasoning(monkeypatch):
    grabbed = _capture_temp(monkeypatch)
    InformedOracle().predict(_OracleSpec(), _FakeModel(reasoning_cfg={"reasoning_effort": "low"}),
                             [{"scenario": "a"}], {"k_items": 1},
                             RunConfig(temperature=0.7))
    assert grabbed["temperature"] == 0.7


# --- 5. config / id -----------------------------------------------------------

def test_grid_and_id():
    m = REGISTRY["informed_oracle"]
    assert m.hyperparameter_grid() == [{}]
    assert m.method_id({}) == "informed_oracle"


def test_k_resolution():
    assert InformedOracle()._k({}) == 20                 # methods.yaml default
    assert InformedOracle(k_items=5)._k({}) == 5         # constructor override
    assert InformedOracle()._k({"k_items": 7}) == 7      # explicit hp wins


# --- 6. report exhibit --------------------------------------------------------

def test_example_prompt_exhibit(monkeypatch):
    from behavior_prediction.evals import get_spec
    q = mmlu_data.Question(subject="anatomy", stem="S?", choices=["a", "b", "c", "d"],
                           answer_idx=0)
    monkeypatch.setattr(mmlu_data, "load_questions", lambda subj, n: [q][:n])
    spec = get_spec("capability_mmlu")
    ex = elicitation.example_prompt_exhibit("informed_oracle", spec, {"scenario": "anatomy"}, {})
    assert ex is not None and len(ex) == 1
    (label, prompt), = ex.items()
    assert "item 0" in label and mmlu_data.mcq_prompt(q) in prompt


def test_example_prompt_exhibit_none_when_no_items(monkeypatch):
    from behavior_prediction.evals import get_spec
    monkeypatch.setattr(mmlu_data, "load_questions", lambda subj, n: [])
    spec = get_spec("capability_mmlu")
    assert elicitation.example_prompt_exhibit(
        "informed_oracle", spec, {"scenario": "anatomy"}, {}) is None
