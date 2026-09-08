"""Offline tests for the oracle_xmm ensembles (equal-weight + trained/learned). Component
prediction docs are stubbed by monkeypatching ``trained._load_components``."""
from __future__ import annotations

import pytest

from behavior_prediction.evals.base import EvalSpec
from behavior_prediction.evals import discrimeval as D
from behavior_prediction.methods import REGISTRY, DEFAULT_METHODS
from behavior_prediction.methods import trained
from behavior_prediction.methods.base import RunConfig
from behavior_prediction.methods.trained import OracleXmm, OracleXmmLearned


class _AbsSpec(EvalSpec):
    name = "abs_toy"
    scoring_semantics = "absolute_rate"

    def condition_key(self, cond):
        return cond["scenario"]

    def oracle_items(self, cond, k):     # so InformedOracle().applies_to -> True
        return {"protocol": "", "items": []}


class _FakeModel:
    name = "fake"
    full = "prov/fake"
    reasoning_cfg: dict = {}


def _abs_doc(d):
    return {k: {"predicted_rate": v, "scenario": k, "condition": {"scenario": k}}
            for k, v in d.items()}


def _stub(monkeypatch, xmm, oracle):
    monkeypatch.setattr(trained, "_load_components", lambda spec, model: (xmm, oracle))


def _conds(keys):
    return [{"scenario": k} for k in keys]


# --- registration / gating ----------------------------------------------------

def test_registered_but_not_default():
    """The oracle-xmm ensembles stay registered (runnable by name) but are excluded from
    DEFAULT_METHODS (policy, 2026-08-10): no default run generates them."""
    assert "oracle_xmm" in REGISTRY and "oracle_xmm_learned" in REGISTRY
    assert "oracle_xmm" not in DEFAULT_METHODS
    assert "oracle_xmm_learned" not in DEFAULT_METHODS
    assert REGISTRY["oracle_xmm_learned"].trained is True
    assert REGISTRY["oracle_xmm"].trained is False


def test_learned_applies_only_where_oracle_does():
    """The ensemble reads informed_oracle's prediction file, so it applies where the oracle
    does — EXCEPT on a ``test_only`` eval, where the learned (trained) variant has no dev pool
    to fit its fold weights on: agentic_misalignment grew oracle_items (2026-08-15), so the
    oracle applies there but the learned ensemble still skips."""
    from behavior_prediction.evals import get_spec
    m = REGISTRY["oracle_xmm_learned"]
    for ev in ("sycophancy_pushback", "capability_mmlu", "discrimeval", "propensitybench"):
        spec = get_spec(ev)
        assert m.applies_to(spec) == REGISTRY["informed_oracle"].applies_to(spec)
    assert m.applies_to(get_spec("propensitybench")) is True
    assert REGISTRY["informed_oracle"].applies_to(get_spec("agentic_misalignment")) is True
    assert m.applies_to(get_spec("agentic_misalignment")) is False   # trained: test_only gate


# --- equal-weight OracleXmm ---------------------------------------------------

def test_absolute_equal_weight_mean(monkeypatch):
    xmm = {"a": 0.1, "b": 0.2, "c": 0.3}
    oracle = {"a": 0.2, "b": 0.1, "c": 0.3}
    _stub(monkeypatch, _abs_doc(xmm), _abs_doc(oracle))
    out = OracleXmm().predict(_AbsSpec(), _FakeModel(), _conds("abc"), {}, RunConfig())
    z = OracleXmm._standardize
    zx, zo = z(xmm), z(oracle)
    for u in "abc":
        assert out[u]["predicted_rate"] == pytest.approx(0.5 * (zx[u] + zo[u]))
    assert out["c"]["predicted_rate"] == max(out[u]["predicted_rate"] for u in "abc")


def test_missing_component_returns_empty(monkeypatch):
    monkeypatch.setattr(trained, "_load_components", lambda spec, model: None)
    assert OracleXmm().predict(_AbsSpec(), _FakeModel(), _conds("a"), {}, RunConfig()) == {}


def test_bias_contrast_derives_prior_gap(monkeypatch):
    from behavior_prediction.evals import get_spec
    spec = get_spec("discrimeval")
    cat = "personal_lending"
    conds = [D.make_condition(cat, axis="baseline"),
             D.make_condition(cat, axis="race", value="Black"),
             D.make_condition(cat, axis="race", value="Asian")]
    xmm_doc = {c_key: {"predicted_rate": r, "scenario": cat, "condition": cond}
               for c_key, r, cond in [
                   (f"{cat}/baseline", 0.9, conds[0]),
                   (f"{cat}/race/Black", 0.8, conds[1]),
                   (f"{cat}/race/Asian", 0.85, conds[2])]}
    oracle_doc = {f"{cat}/race/Black-vs-white": {"predicted_rate": -0.2, "scenario": cat,
                                                 "condition": conds[1]},
                  f"{cat}/race/Asian-vs-white": {"predicted_rate": 0.1, "scenario": cat,
                                                 "condition": conds[2]}}
    _stub(monkeypatch, xmm_doc, oracle_doc)
    out = OracleXmm().predict(spec, _FakeModel(), conds, {}, RunConfig())
    assert set(out) == set(oracle_doc)                 # contrast-keyed, baseline dropped
    # xmm gaps: Black -0.1, Asian -0.05 -> z(-1,+1); oracle gaps z(-1,+1); mean -> -1, +1
    assert out[f"{cat}/race/Black-vs-white"]["predicted_rate"] == pytest.approx(-1.0)
    assert out[f"{cat}/race/Asian-vs-white"]["predicted_rate"] == pytest.approx(1.0)


# --- trained OracleXmmLearned -------------------------------------------------

def _targets(d):
    return {k: {"rate": v, "condition": {"scenario": k}} for k, v in d.items()}


def test_reliability_downweights_dead_component(monkeypatch):
    # xmm tracks the actual perfectly; oracle is anti-correlated -> its clipped reliability is 0,
    # so the learned weight favors xmm. shrink=1.0 gives the full downweight (w_oracle -> 0).
    actual = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
    xmm = dict(actual)                                  # r=+1
    oracle = {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}   # r=-1 -> clipped 0
    _stub(monkeypatch, _abs_doc(xmm), _abs_doc(oracle))
    m = OracleXmmLearned()
    fs_full = m.fit(_targets(actual), _FakeModel(), _AbsSpec(), {"shrink": 1.0})
    assert fs_full["w_xmm"] == pytest.approx(1.0) and fs_full["w_oracle"] == pytest.approx(0.0)
    # default shrink=0.5 -> halfway between equal (0.5) and the reliability share (1.0) = 0.75
    fs = m.fit(_targets(actual), _FakeModel(), _AbsSpec(), {})
    assert fs["w_xmm"] == pytest.approx(0.75) and fs["w_oracle"] == pytest.approx(0.25)
    out = m.predict(_AbsSpec(), _FakeModel(), _conds("abcd"), {}, RunConfig(), fs)
    assert sorted("abcd", key=lambda u: out[u]["predicted_rate"]) == ["a", "b", "c", "d"]
    assert all(v["w_xmm"] + v["w_oracle"] == pytest.approx(1.0) for v in out.values())  # convex


def test_learned_is_out_of_fold(monkeypatch):
    # Fit on a,b,c; predict held-out d,e. Weights come only from the fit fold.
    actual = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}
    xmm = dict(actual)
    oracle = {k: 0.5 for k in actual}                  # constant -> zero variance -> weight 0
    _stub(monkeypatch, _abs_doc(xmm), _abs_doc(oracle))
    m = OracleXmmLearned()
    fs = m.fit(_targets({k: actual[k] for k in "abc"}), _FakeModel(), _AbsSpec(), {"shrink": 1.0})
    assert fs["w_oracle"] == pytest.approx(0.0)         # dead (constant) component
    out = m.predict(_AbsSpec(), _FakeModel(), _conds("de"), {}, RunConfig(), fs)
    assert set(out) == {"d", "e"}
    assert out["e"]["predicted_rate"] > out["d"]["predicted_rate"]   # xmm order preserved OOF


def test_empty_fit_yields_none(monkeypatch):
    _stub(monkeypatch, _abs_doc({"a": 0.1, "b": 0.2}), _abs_doc({"a": 0.2, "b": 0.1}))
    m = OracleXmmLearned()
    fs = m.fit(_targets({"a": 0.1}), _FakeModel(), _AbsSpec(), {})   # <3 units -> {}
    assert fs == {}
    out = m.predict(_AbsSpec(), _FakeModel(), _conds("ab"), {}, RunConfig(), fs)
    assert all(v["predicted_rate"] is None for v in out.values())


def test_weights_are_convex_and_shrink_ordered(monkeypatch):
    # Both components positively but unequally correlated; check the shrink knob moves the weight
    # monotonically from equal (0.5) toward the reliability share, staying convex.
    actual = {"a": 0.1, "b": 0.25, "c": 0.2, "d": 0.4, "e": 0.35}
    xmm = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4, "e": 0.5}     # strong r
    oracle = {"a": 0.2, "b": 0.3, "c": 0.1, "d": 0.5, "e": 0.3}  # weaker r
    _stub(monkeypatch, _abs_doc(xmm), _abs_doc(oracle))
    m = OracleXmmLearned()
    w_half = m.fit(_targets(actual), _FakeModel(), _AbsSpec(), {"shrink": 0.5})["w_xmm"]
    w_full = m.fit(_targets(actual), _FakeModel(), _AbsSpec(), {"shrink": 1.0})["w_xmm"]
    assert 0.5 < w_half < w_full                         # xmm stronger -> weight above 0.5, grows with shrink
    out = m.predict(_AbsSpec(), _FakeModel(), _conds("abcde"), {}, RunConfig(),
                    m.fit(_targets(actual), _FakeModel(), _AbsSpec(), {}))
    assert len(out) == 5 and all(v["predicted_rate"] is not None for v in out.values())


def test_config_grid_ids():
    m = REGISTRY["oracle_xmm_learned"]
    ids = {m.method_id(hp) for hp in m.hyperparameter_grid()}
    assert ids == {"oracle_xmm_learned", "oracle_xmm_learned-full"}
