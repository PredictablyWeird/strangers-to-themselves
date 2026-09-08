"""The llm_prediction trained method: fit gathers labelled examples; predict prompts an analyst
LLM. The single LLM call is mocked, so these stay offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavior_prediction import common, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import REGISTRY, RunConfig
from behavior_prediction.methods.trained import LlmPrediction

_TARGETS = "results/propensitybench/llama-3.3-70b/targets.json"


@pytest.fixture
def pb():
    if not Path(_TARGETS).exists():
        pytest.skip("committed propensitybench targets not present")
    spec = get_spec("propensitybench")
    targets = common.load_json(_TARGETS)["targets"]
    return spec, targets


def test_registered_and_trained():
    m = REGISTRY["llm_prediction"]
    assert isinstance(m, LlmPrediction) and m.trained is True
    assert not m.capabilities.needs_activations  # runs for any model, incl. API-only


def test_fit_caps_with_deterministic_random_subsample(pb):
    spec, targets = pb
    m = LlmPrediction(max_train_examples=5, sample_seed=7)
    fs = m.fit(targets, None, spec, {})
    assert len(fs["examples"]) == 5                       # capped
    assert fs["examples"] == sorted(fs["examples"])       # stable display order
    desc, rate = fs["examples"][0]
    assert isinstance(desc, str) and 0.0 <= rate <= 1.0
    # Deterministic: same training set + seed -> identical subsample (so every target this fit
    # predicts sees the same block); a different seed generally differs.
    assert m.fit(targets, None, spec, {})["examples"] == fs["examples"]
    other = LlmPrediction(max_train_examples=5, sample_seed=99).fit(targets, None, spec, {})
    assert other["examples"] != fs["examples"]


def test_description_modes(pb):
    spec, targets = pb
    cond = next(iter(targets.values()))["condition"]
    m = LlmPrediction()  # mode is now a per-call/hyperparameter arg, not constructor state
    label = m._describe(spec, cond, "label")
    gist = m._describe(spec, cond, "gist")
    full = m._describe(spec, cond, "full")
    assert gist.startswith(label + " |")          # gist = label + first-sentence gist
    assert "action:" in full and "context:" in full
    assert len(full) > len(gist) > len(label)


def test_predict_prompts_the_analyst_and_decorates_entries(pb, monkeypatch):
    spec, targets = pb
    m = LlmPrediction()
    fs = m.fit(targets, None, spec, {})
    conds = [t["condition"] for t in list(targets.values())[:3]]

    seen = {}

    def fake_elicit(model_name, prompts, **kw):
        seen["model"] = model_name
        seen["prompts"] = prompts
        return {k: {"predicted_rate": 0.42, "n": kw.get("runs"), "samples": [], "raw": [],
                    "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    out = m.predict(spec, None, conds, {}, RunConfig(), fs)

    assert seen["model"] == m.predictor_model              # analyst model, not the model under test
    assert len(seen["prompts"]) == len(conds)
    for p in seen["prompts"].values():
        assert "TRAINING CASES" in p and "NEW CASE TO PREDICT" in p
    # Calibration is always applied: the raw analyst rate is shifted by (anchor - train_mean).
    expected = max(0.0, min(1.0, 0.42 - fs["train_mean"] + m.CALIBRATION_ANCHOR))
    for c in conds:
        e = out[spec.condition_key(c)]
        assert e["raw_rate"] == 0.42 and e["train_mean"] == fs["train_mean"]
        assert e["predicted_rate"] == pytest.approx(expected)
        assert e["scenario"] == c["scenario"] and e["condition"] == c
        assert e["n_train"] == len(fs["examples"])


def test_predict_with_empty_fit_returns_none(pb):
    spec, targets = pb
    conds = [t["condition"] for t in list(targets.values())[:2]]
    out = LlmPrediction().predict(spec, None, conds, {}, RunConfig(), {"examples": []})
    assert all(e["predicted_rate"] is None for e in out.values())


# --- bias_contrast (DiscrimEval) grain: fit on contrast gaps, predict each contrast directly ----

_DE_TARGETS = "results/discrimeval/llama-3.3-70b/targets.json"


@pytest.fixture
def de():
    if not Path(_DE_TARGETS).exists():
        pytest.skip("committed discrimeval targets not present")
    spec = get_spec("discrimeval")
    targets = common.load_json(_DE_TARGETS)["targets"]
    return spec, targets


def test_parse_signed_prediction_tag():
    p = common.parse_signed_prediction_tag
    assert p("PREDICTION: -12") == pytest.approx(-0.12)
    assert p("thinking...\nPREDICTION: +30") == pytest.approx(0.30)
    assert p("PREDICTION: 0") == 0.0
    assert p("the gap is roughly -5") == pytest.approx(-0.05)   # fallback: last signed number
    assert p("PREDICTION: -250") == -1.0                        # clamped
    assert p("no numbers here") is None and p(None) is None


def test_bias_fit_builds_signed_contrast_examples(de):
    spec, targets = de
    m = LlmPrediction(max_train_examples=None)
    fs = m.fit(targets, None, spec, {})
    # One example per (category, axis, value) contrast — not per cell (no baseline entries).
    gaps = spec.score_contrasts(targets, {k: t.get("rate") for k, t in targets.items()})
    assert len(fs["examples"]) == len(gaps)
    for desc, gap in fs["examples"]:
        assert " vs " in desc and "(baseline)" in desc         # canonical contrast description
        assert -1.0 <= gap <= 1.0
    assert any(gap < 0 for _, gap in fs["examples"])           # signed labels, not rates
    assert fs["train_mean"] == pytest.approx(
        sum(g for _, g in fs["examples"]) / len(fs["examples"]))


def test_bias_predict_keys_by_contrast_and_calibrates_to_zero(de, monkeypatch):
    spec, targets = de
    m = LlmPrediction()
    fs = m.fit(targets, None, spec, {})
    # One category's cells: the baseline cell must drop out; each comparison cell -> its contrast.
    conds = [t["condition"] for t in targets.values()
             if t["scenario"] == "personal_lending"]
    n_contrasts = sum(1 for c in conds if c["axis"] != "baseline")

    seen = {}

    def fake_elicit(model_name, prompts, **kw):
        seen["prompts"], seen["parse_fn"] = prompts, kw.get("parse_fn")
        return {k: {"predicted_rate": -0.07, "n": 1, "samples": [], "raw": [],
                    "parse_failures": 0} for k in prompts}

    monkeypatch.setattr(common, "elicit_rates", fake_elicit)
    out = m.predict(spec, None, conds, {}, RunConfig(), fs)

    assert seen["parse_fn"] is common.parse_signed_prediction_tag
    assert len(out) == n_contrasts
    for key, e in out.items():
        assert "-vs-" in key                                   # contrast keys, not cell keys
        assert e["scenario"] == "personal_lending"
    for p in seen["prompts"].values():
        assert "TRAINING CONTRASTS" in p and "NEW CONTRAST TO PREDICT" in p
        assert "PREDICTION: <integer from -100 to 100>" in p
    # Calibration re-centres gaps on 0 (not CALIBRATION_ANCHOR) and allows negative predictions.
    expected = max(-1.0, min(1.0, -0.07 - fs["train_mean"]))
    for e in out.values():
        assert e["raw_rate"] == -0.07 and e["train_mean"] == fs["train_mean"]
        assert e["predicted_rate"] == pytest.approx(expected)
