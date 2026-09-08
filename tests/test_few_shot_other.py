"""few_shot_other: few_shot's conversation, answered by a named other model. The defining property
is that the user/assistant turns are byte-identical to few_shot's — only a system turn and the
answering model differ — so the two methods form an ablation rather than a confound. The LLM call is
mocked, so these stay offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_prediction import common
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY, RunConfig
from behavior_prediction.methods.trained import FewShot, FewShotOther, LlmPrediction

_TARGETS = "results/propensitybench/llama-3.3-70b/targets.json"
_DE_TARGETS = "results/discrimeval/llama-3.3-70b/targets.json"


class _Model:
    def __init__(self, reasoning_cfg=None):
        self.full = "openrouter/meta-llama/llama-3.3-70b-instruct"
        self.reasoning_cfg = reasoning_cfg or {}


@pytest.fixture
def pb():
    if not Path(_TARGETS).exists():
        pytest.skip("committed propensitybench targets not present")
    return get_spec("propensitybench"), common.load_json(_TARGETS)["targets"]


@pytest.fixture
def de():
    if not Path(_DE_TARGETS).exists():
        pytest.skip("committed discrimeval targets not present")
    return get_spec("discrimeval"), common.load_json(_DE_TARGETS)["targets"]


def _capture(monkeypatch, rate=0.42):
    seen = {}

    def fake_elicit(model_name, prompts, **kw):
        seen["model"] = model_name
        seen["prompts"] = prompts
        seen.update({k: kw.get(k) for k in ("runs", "temperature", "parse_fn", "reasoning_config")})
        return {k: {"predicted_rate": rate, "n": 1, "samples": [rate], "raw": [""],
                    "reasoning": [None], "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    return seen


# --- model_display_name: routing prefix out, informative name in --------------------------------


def test_model_display_name_strips_routing_prefix_only():
    f = common.model_display_name
    assert f("openrouter/meta-llama/llama-3.3-70b-instruct") == "meta-llama/llama-3.3-70b-instruct"
    assert f("together/meta-llama/Llama-3.3-70B-Instruct-Turbo") == \
        "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    assert f("openrouter/anthropic/claude-sonnet-4") == "anthropic/claude-sonnet-4"
    assert f("bare-model-id") == "bare-model-id"          # nothing to strip


def test_registered_and_defaults_to_the_llm_prediction_analyst():
    m = REGISTRY["few_shot_other"]
    assert isinstance(m, FewShotOther) and m.trained is True
    assert m.predictor_model == LlmPrediction.DEFAULT_PREDICTOR   # same analyst as llm_prediction
    assert m.hyperparameter_grid() == [{}] and m.method_id({}) == "few_shot_other"


# --- the defining property: identical turns, only the system turn + answerer differ -------------


def test_turns_are_byte_identical_to_few_shot(pb, monkeypatch):
    spec, targets = pb
    conds = [t["condition"] for t in list(targets.values())[:3]]

    fs = FewShot(max_train_examples=3)
    fso = FewShotOther(max_train_examples=3)
    # Same fit: same subsample, same order, same labels.
    assert fso.fit(targets, None, spec, {}) == fs.fit(targets, None, spec, {})
    state = fs.fit(targets, None, spec, {})

    seen_self = _capture(monkeypatch)
    fs.predict(spec, _Model(), conds, {}, RunConfig(), state)
    self_prompts = dict(seen_self["prompts"])

    seen_other = _capture(monkeypatch)
    fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    other_prompts = dict(seen_other["prompts"])

    assert set(self_prompts) == set(other_prompts)
    for key, msgs in other_prompts.items():
        assert msgs[0]["role"] == "system"
        assert msgs[1:] == self_prompts[key]      # every user/assistant turn byte-identical


def test_system_turn_names_the_model_and_resolves_the_second_person(pb, monkeypatch):
    spec, targets = pb
    fso = FewShotOther(max_train_examples=2)
    state = fso.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:1]]
    seen = _capture(monkeypatch)

    fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    sys_turn = next(iter(seen["prompts"].values()))[0]["content"]

    name = "meta-llama/llama-3.3-70b-instruct"
    assert name in sys_turn
    assert "openrouter/" not in sys_turn          # routing prefix dropped
    # The three jobs of the system turn.
    assert '"you" refers to' in sys_turn          # resolves the second person in the eval's text
    assert "MEASURED" in sys_turn                 # assistant turns are ground truth, not guesses
    assert "predicting the behavior of another AI model" in sys_turn


def test_asks_the_analyst_not_the_model_under_test(pb, monkeypatch):
    spec, targets = pb
    fso = FewShotOther(max_train_examples=2)
    state = fso.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:1]]
    seen = _capture(monkeypatch)

    fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    assert seen["model"] == common.resolve_model(LlmPrediction.DEFAULT_PREDICTOR)[0]
    assert seen["model"] != _Model().full
    # The analyst is fixed and chosen by us, so it is greedy unconditionally.
    assert seen["temperature"] == LlmPrediction.ANALYST_TEMPERATURE


def test_reasoning_model_under_test_does_not_change_the_analyst_temperature(pb, monkeypatch):
    """few_shot's reasoning-endpoint carve-out keys off the model under test; for few_shot_other
    that model never answers, so it must not affect sampling."""
    spec, targets = pb
    fso = FewShotOther(max_train_examples=2)
    state = fso.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:1]]
    seen = _capture(monkeypatch)

    fso.predict(spec, _Model({"reasoning_effort": "low"}), conds, {}, RunConfig(temperature=1.0),
                state)
    assert seen["temperature"] == LlmPrediction.ANALYST_TEMPERATURE
    assert seen["reasoning_config"] == {}         # the ANALYST's config, not the subject's


def test_predictor_model_is_configurable(pb, monkeypatch):
    spec, targets = pb
    fso = FewShotOther(max_train_examples=2, predictor_model="openrouter/openai/gpt-5.4-nano")
    state = fso.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:1]]
    seen = _capture(monkeypatch)

    fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    assert seen["model"] == "openrouter/openai/gpt-5.4-nano"


def test_calibration_and_empty_fit_match_few_shot(pb, monkeypatch):
    spec, targets = pb
    fso = FewShotOther(max_train_examples=3)
    state = fso.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:2]]
    _capture(monkeypatch, rate=0.42)

    out = fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    expected = max(0.0, min(1.0, 0.42 - state["train_mean"] + FewShotOther.CALIBRATION_ANCHOR))
    for c in conds:
        e = out[spec.condition_key(c)]
        assert e["raw_rate"] == 0.42 and e["predicted_rate"] == pytest.approx(expected)

    empty = FewShotOther().predict(spec, _Model(), conds, {}, RunConfig(), {"examples": []})
    assert all(e["predicted_rate"] is None for e in empty.values())


# --- bias path: same contrast grain + both applicant orders, under one system turn ---------------


def test_bias_path_keeps_identical_turns_and_both_orders(de, monkeypatch):
    spec, targets = de
    conds = [t["condition"] for t in targets.values() if t["scenario"] == "personal_lending"]
    n_contrasts = sum(1 for c in conds if c["axis"] != "baseline")

    fs, fso = FewShot(max_train_examples=2), FewShotOther(max_train_examples=2)
    state = fs.fit(targets, None, spec, {})

    seen_self = _capture(monkeypatch, rate=-0.07)
    fs.predict(spec, _Model(), conds, {}, RunConfig(), state)
    self_prompts = dict(seen_self["prompts"])

    seen_other = _capture(monkeypatch, rate=-0.07)
    out = fso.predict(spec, _Model(), conds, {}, RunConfig(), state)
    other_prompts = dict(seen_other["prompts"])

    assert len(out) == n_contrasts
    assert len(other_prompts) == 2 * n_contrasts          # both applicant orders, as few_shot
    assert seen_other["parse_fn"] == spec.parse_bias_gap
    for key, msgs in other_prompts.items():
        assert msgs[0]["role"] == "system"
        assert msgs[1:] == self_prompts[key]              # identical turns on the bias path too
