"""The few_shot trained method: fit gathers labelled examples; predict shows them to the MODEL UNDER
TEST as a conversation whose assistant turns are the measured rates. The LLM call is mocked, so these
stay offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_prediction import common, elicitation
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY, RunConfig
from behavior_prediction.methods.trained import FewShot

_TARGETS = "results/propensitybench/llama-3.3-70b/targets.json"


class _Model:
    """Minimal stand-in for ModelAdapter (few_shot reads only .full / .reasoning_cfg)."""

    def __init__(self, reasoning_cfg=None):
        self.full = "openrouter/meta-llama/llama-3.3-70b-instruct"
        self.reasoning_cfg = reasoning_cfg or {}


@pytest.fixture
def pb():
    if not Path(_TARGETS).exists():
        pytest.skip("committed propensitybench targets not present")
    spec = get_spec("propensitybench")
    targets = common.load_json(_TARGETS)["targets"]
    return spec, targets


def _capture(monkeypatch, rate=0.42):
    """Patch elicit_rates; return the dict the call's arguments land in."""
    seen = {}

    def fake_elicit(model_name, prompts, **kw):
        seen["model"] = model_name
        seen["prompts"] = prompts
        seen.update({k: kw.get(k) for k in ("runs", "temperature", "parse_fn", "reasoning_config")})
        return {k: {"predicted_rate": rate, "n": 1, "samples": [rate], "raw": [""],
                    "reasoning": [None], "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    return seen


def test_registered_and_trained():
    m = REGISTRY["few_shot"]
    assert isinstance(m, FewShot) and m.trained is True
    assert not m.capabilities.needs_activations   # runs for any model, incl. API-only
    assert m.default_runs == 1                    # greedy: the k-turn context is the cost
    assert m.hyperparameter_grid() == [{}]        # k is configured, not swept
    assert m.method_id({}) == "few_shot"


def test_fit_caps_with_deterministic_random_subsample(pb):
    spec, targets = pb
    m = FewShot(max_train_examples=5, sample_seed=7)
    fs = m.fit(targets, None, spec, {})
    assert len(fs["examples"]) == 5                        # capped
    key, cond, rate = fs["examples"][0]
    assert isinstance(key, str) and isinstance(cond, dict) and 0.0 <= rate <= 1.0
    # Deterministic: same training set + seed -> identical examples AND identical order (so every
    # target this fit predicts sees the same conversation); a different seed generally differs.
    assert m.fit(targets, None, spec, {})["examples"] == fs["examples"]
    other = FewShot(max_train_examples=5, sample_seed=99).fit(targets, None, spec, {})
    assert other["examples"] != fs["examples"]
    assert fs["train_mean"] == pytest.approx(sum(r for _, _, r in fs["examples"]) / 5)


def test_fit_shuffles_presentation_order(pb):
    """Order is a seeded shuffle, not a sort — in a turn sequence, sorting by key would front-load
    whole scenarios right before a held-out-scenario target."""
    spec, targets = pb
    fs = FewShot(max_train_examples=None, sample_seed=1234).fit(targets, None, spec, {})
    keys = [k for k, _, _ in fs["examples"]]
    assert len(keys) > 5
    assert keys != sorted(keys)


def test_predict_builds_a_conversation_ending_in_the_self_report_prompt(pb, monkeypatch):
    spec, targets = pb
    m = FewShot(max_train_examples=3)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:3]]
    seen = _capture(monkeypatch)

    m.predict(spec, _Model(), conds, {}, RunConfig(), fs)

    # The model under test answers — NOT a third-party analyst (llm_prediction's key difference).
    assert seen["model"] == _Model().full
    assert seen["runs"] == 1 and seen["temperature"] == FewShot.GREEDY_TEMPERATURE
    assert seen["parse_fn"] is common.parse_percentage
    assert len(seen["prompts"]) == len(conds)

    for cond in conds:
        msgs = seen["prompts"][spec.condition_key(cond)]
        # The examples this target actually sees: all of them but itself (a target that happens to
        # be in the sampled fit set is filtered out of its own conversation).
        shown = [e for e in fs["examples"] if e[0] != spec.condition_key(cond)]
        # A conversation, not a string: strict user/assistant alternation ending on a user turn.
        assert isinstance(msgs, list)
        assert [msg["role"] for msg in msgs] == ["user", "assistant"] * len(shown) + ["user"]
        # The assistant turns are the MEASURED rates, as bare integers (self_report's answer format).
        assert [msg["content"] for msg in msgs[1::2]] == [str(round(r * 100)) for _, _, r in shown]
        # The controlled-comparison property: the final turn is EXACTLY what self_report sends.
        assert msgs[-1]["content"] == elicitation.self_report_prompt(spec, cond, None)
        # ...and each training turn is a well-formed self-report of its own case.
        for msg, (_k, c, _r) in zip(msgs[0::2], shown):
            assert msg["content"] == elicitation.self_report_prompt(spec, c, None)


def test_predict_calibrates_and_decorates_entries(pb, monkeypatch):
    spec, targets = pb
    m = FewShot(max_train_examples=3)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:2]]
    _capture(monkeypatch, rate=0.42)

    out = m.predict(spec, _Model(), conds, {}, RunConfig(), fs)

    expected = max(0.0, min(1.0, 0.42 - fs["train_mean"] + m.CALIBRATION_ANCHOR))
    for c in conds:
        e = out[spec.condition_key(c)]
        assert e["raw_rate"] == 0.42 and e["train_mean"] == fs["train_mean"]
        assert e["predicted_rate"] == pytest.approx(expected)
        assert e["scenario"] == c["scenario"] and e["condition"] == c
        assert e["n_train"] == len([x for x in fs["examples"] if x[0] != spec.condition_key(c)])


def test_reasoning_model_keeps_default_temperature(pb, monkeypatch):
    """A reasoning endpoint may reject temperature 0, so those models keep cfg.temperature."""
    spec, targets = pb
    m = FewShot(max_train_examples=2)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:1]]
    seen = _capture(monkeypatch)

    m.predict(spec, _Model({"reasoning_effort": "low"}), conds, {}, RunConfig(temperature=0.7), fs)
    assert seen["temperature"] == 0.7
    assert seen["reasoning_config"] == {"reasoning_effort": "low"}


def test_target_never_appears_in_its_own_conversation(pb, monkeypatch):
    """Belt-and-braces: even if a caller hands fit and predict overlapping keys (CV never does), a
    target's own measured rate must not enter its own prompt."""
    spec, targets = pb
    m = FewShot(max_train_examples=None)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:2]]
    seen = _capture(monkeypatch)

    m.predict(spec, _Model(), conds, {}, RunConfig(), fs)
    for cond in conds:
        key = spec.condition_key(cond)
        own = elicitation.self_report_prompt(spec, cond, None)
        # The target's self-report question appears exactly once: as the final (unanswered) turn.
        assert [msg["content"] for msg in seen["prompts"][key]].count(own) == 1


def test_predict_with_empty_fit_returns_none(pb):
    spec, targets = pb
    conds = [t["condition"] for t in list(targets.values())[:2]]
    out = FewShot().predict(spec, _Model(), conds, {}, RunConfig(), {"examples": []})
    assert all(e["predicted_rate"] is None for e in out.values())


# --- bias_contrast (DiscrimEval) grain: contrast-keyed turns, both applicant orders --------------

_DE_TARGETS = "results/discrimeval/llama-3.3-70b/targets.json"


@pytest.fixture
def de():
    if not Path(_DE_TARGETS).exists():
        pytest.skip("committed discrimeval targets not present")
    spec = get_spec("discrimeval")
    targets = common.load_json(_DE_TARGETS)["targets"]
    return spec, targets


def test_bias_fit_builds_signed_contrast_examples(de):
    spec, targets = de
    m = FewShot(max_train_examples=None)
    fs = m.fit(targets, None, spec, {})
    # One example per contrast — not per cell (no baseline entries), keyed so the conversation can
    # re-render each contrast's actual prompt.
    gaps = spec.score_contrasts(targets, {k: t.get("rate") for k, t in targets.items()})
    assert len(fs["examples"]) == len(gaps)
    for ck, payload, gap in fs["examples"]:
        assert ck == payload and "-vs-" in ck      # payload IS the contrast key on this path
        assert -1.0 <= gap <= 1.0
    assert any(g < 0 for _, _, g in fs["examples"])   # signed labels, not rates


def test_bias_predict_keys_by_contrast_and_asks_both_orders(de, monkeypatch):
    spec, targets = de
    m = FewShot(max_train_examples=2)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in targets.values() if t["scenario"] == "personal_lending"]
    n_contrasts = sum(1 for c in conds if c["axis"] != "baseline")
    seen = _capture(monkeypatch, rate=-0.07)

    out = m.predict(spec, _Model(), conds, {}, RunConfig(), fs)

    assert seen["parse_fn"] == spec.parse_bias_gap   # the eval's own signed-gap parser
    assert len(out) == n_contrasts
    # Two conversations per contrast (both applicant orders); baseline cells drop out.
    assert len(seen["prompts"]) == 2 * n_contrasts
    for key in out:
        assert "-vs-" in key                                   # contrast keys, not cell keys
        assert out[key]["scenario"] == "personal_lending"
        assert f"{key}##o0" in seen["prompts"] and f"{key}##o1" in seen["prompts"]


def test_bias_swapped_order_negates_training_gaps_and_the_reply(de, monkeypatch):
    spec, targets = de
    m = FewShot(max_train_examples=2)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in targets.values() if t["scenario"] == "personal_lending"]
    seen = _capture(monkeypatch, rate=-0.07)

    out = m.predict(spec, _Model(), conds, {}, RunConfig(), fs)

    key = next(iter(out))
    o0, o1 = seen["prompts"][f"{key}##o0"], seen["prompts"][f"{key}##o1"]
    # In the swapped order the applicants trade places, so each training answer is the negated gap.
    a0 = [int(msg["content"]) for msg in o0[1::2]]
    a1 = [int(msg["content"]) for msg in o1[1::2]]
    assert a1 == [-v for v in a0]
    assert o0[-1]["content"] != o1[-1]["content"]      # different applicant order in the target turn
    # Pooling negates the swapped order's reply, so a constant position bias cancels: here both
    # orders answer -0.07, which pools to 0 before calibration.
    assert out[key]["samples"] == [-0.07, 0.07]
    assert out[key]["raw_rate"] == pytest.approx(0.0)


def test_bias_calibrates_to_zero_and_allows_negative(de, monkeypatch):
    spec, targets = de
    m = FewShot(max_train_examples=2)
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in targets.values() if t["scenario"] == "personal_lending"]
    _capture(monkeypatch, rate=-0.07)

    out = m.predict(spec, _Model(), conds, {}, RunConfig(), fs)
    # Gaps re-centre on 0 (not CALIBRATION_ANCHOR); the pooled raw is 0.0 (see the test above).
    expected = max(-1.0, min(1.0, 0.0 - fs["train_mean"]))
    for e in out.values():
        assert e["train_mean"] == fs["train_mean"]
        assert e["predicted_rate"] == pytest.approx(expected)


# --- the elicit_rates seam: conversations vs plain strings ---------------------------------------


def test_as_model_input_passes_strings_through_and_converts_turns():
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    assert common._as_model_input("plain prompt") == "plain prompt"
    msgs = common._as_model_input([{"role": "user", "content": "q"},
                                   {"role": "assistant", "content": "35"}])
    assert [type(m) for m in msgs] == [ChatMessageUser, ChatMessageAssistant]
    assert [m.content for m in msgs] == ["q", "35"]


def test_as_model_input_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="unknown conversation role"):
        common._as_model_input([{"role": "bot", "content": "hi"}])
