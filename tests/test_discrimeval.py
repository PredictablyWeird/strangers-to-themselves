"""DiscrimEval topic-category grain: split universe, folds, per-value dual-mode contrast scoring,
selection-CV.

All synthetic (no model calls) so it doesn't depend on committed targets."""

from __future__ import annotations

import re

from behavior_prediction import metrics, selection, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.evals.discrimeval import (
    AXES, CATEGORIES, COMPARISONS, TEST_CATEGORIES, cells_sweep, contrast_key, contrasts)
from behavior_prediction.methods import REGISTRY as METHODS

# Total contrasts per category = one per (axis, comparison value).
_PER_CAT = sum(len(v) for v in COMPARISONS.values())          # 4 + 2 + 4 = 10
_ALL_CONTRASTS = len(CATEGORIES) * _PER_CAT                    # 90


def _spec():
    return get_spec("discrimeval")


def _synthetic_targets(spec):
    """Cell targets for every (category, cell) — baseline + one cell per (axis, value)."""
    targets = {}
    for i, cond in enumerate(cells_sweep(CATEGORIES)):
        rate = 0.4 + (i % 7) * 0.05  # deterministic, varied, in [0,1)
        targets[spec.condition_key(cond)] = {
            "rate": rate, "count": int(rate * 20), "n": 20,
            "scenario": cond["scenario"], "condition": cond, "logs": [],
        }
    return targets


def _all_contrast_keys(scenarios=None):
    return {contrast_key(cat, axis, v)
            for cat in (scenarios or CATEGORIES) for axis in AXES for v in COMPARISONS[axis]}


def test_registered_runnable_and_universe():
    spec = _spec()
    assert spec.name == "discrimeval" and spec.config == "explicit"
    assert spec.broken is False
    assert spec.split_units() == CATEGORIES
    impl = get_spec("discrimeval_implicit")
    assert impl.name == "discrimeval_implicit" and impl.config == "implicit"
    assert impl.split_units() == CATEGORIES


def test_forced_test_and_three_folds():
    spec = _spec()
    manifest = splits.build_manifest(spec, CATEGORIES)
    assign = manifest["assignments"]
    assert {c for c in CATEGORIES if assign[c] == "test"} == TEST_CATEGORIES
    dev_units = sorted(c for c in CATEGORIES if assign[c] == "dev")
    assert len(dev_units) == 6
    folds = splits.cv_folds(spec, dev_units, seed=manifest["seed"])
    assert len(folds) == 3                          # 3 fold-groups (pairs) → 3 folds
    assert all(len(f) == 2 for f in folds)          # 4 train / 2 eval each
    assert sorted(u for f in folds for u in f) == dev_units


def test_contrasts_are_split_per_comparison_value():
    # race splits into one contrast per non-white race (not a single "race" number).
    assert {c["key"] for c in contrasts()} == _all_contrast_keys()
    race_keys = {c["key"] for c in contrasts(["immigration"]) if c["axis"] == "race"}
    assert race_keys == {f"immigration/race/{v}-vs-white" for v in COMPARISONS["race"]}


def test_score_contrasts_direct_and_derived():
    spec = _spec()
    targets = _synthetic_targets(spec)

    # Direct: predictions keyed by contrast key — used as-is.
    direct = {k: 0.1 for k in _all_contrast_keys()}
    sc_direct = spec.score_contrasts(targets, direct)
    assert len(sc_direct) == _ALL_CONTRASTS == 90
    for (actual, pred) in sc_direct.values():
        assert pred == 0.1

    # Derived: per-cell rate predictions — differenced (group − baseline).
    derived = {k: t["rate"] for k, t in targets.items()}   # oracle rates
    sc_derived = spec.score_contrasts(targets, derived)
    assert len(sc_derived) == 90
    for (actual, pred) in sc_derived.values():
        assert abs(pred - actual) < 1e-9                    # oracle reproduces the actual gap


def test_score_contrasts_respects_in_split_keys():
    spec = _spec()
    targets = _synthetic_targets(spec)
    one = CATEGORIES[0]
    keys = {k for k, t in targets.items() if t["scenario"] == one}
    sc = spec.score_contrasts(targets, {k: t["rate"] for k, t in targets.items()}, keys=keys)
    assert set(sc) == _all_contrast_keys([one])             # only the in-split category's contrasts


def test_selection_cv_runs_with_direct_and_derived_methods():
    spec = _spec()
    targets = _synthetic_targets(spec)
    manifest = splits.build_manifest(spec, CATEGORIES)
    oracle_rates = {k: t["rate"] for k, t in targets.items()}             # derived oracle
    oracle_direct = {ck: pair[0]                                          # actual gaps as direct preds
                     for ck, pair in spec.score_contrasts(targets, oracle_rates).items()}
    flat = {k: 0.5 for k in targets}
    methods = {"direct_oracle": oracle_direct, "derived_oracle": oracle_rates, "flat": flat}

    out = selection.selection_cv(spec, targets, methods, manifest)
    assert out is not None and out["n_folds"] >= 2
    assert out["pooled_r"] > 0.99                                         # an oracle is picked each fold


def test_report_scoring_matches_evaluation():
    """metrics.corr_spec equals selection._corr (the function the tuning/evaluation steps score
    with) on the same dev keys, for a varied (non-constant) predictor."""
    from behavior_prediction import selection
    spec = _spec()
    targets = _synthetic_targets(spec)
    manifest = splits.build_manifest(spec, CATEGORIES)
    dev = splits.split_keys(spec, targets, manifest, "dev")
    preds = {ck: 0.03 * (i % 5 - 2)                                       # varied direct gaps
             for i, ck in enumerate(sorted(_all_contrast_keys()))}
    r_report = metrics.corr_spec(spec, targets, preds, dev)
    r_eval = selection._corr(spec, targets, preds, dev)
    assert r_report is not None and abs(r_report - r_eval) < 1e-9


def test_direct_bias_method_merged_into_self_report():
    # The standalone direct_bias method is gone; self_report now predicts the gap directly.
    assert "direct_bias" not in METHODS
    assert "self_report" in METHODS


def test_introspective_methods_predict_bias_directly():
    spec = get_spec("discrimeval")
    # self_report emits a comparative prompt per contrast on every axis (both A/B orders).
    sr = spec.comparative_bias_prompts(["immigration"], base_method="self_report")
    assert set(sr) == _all_contrast_keys(["immigration"])
    assert all({"o0", "o1"} <= set(v) for v in sr.values())
    # the value method predicts the age axis only (under either aggregate setting).
    for params in ({}, {"aggregate": True}):
        prompts = spec.comparative_bias_prompts(["immigration"], base_method="value", params=params)
        assert set(prompts) == {k for k in _all_contrast_keys(["immigration"]) if "/age/" in k}
        assert all({"o0", "o1"} <= set(v) for v in prompts.values())
    # A non-bias eval doesn't expose the comparative-bias seam.
    assert get_spec("agentic_misalignment").comparative_bias_prompts(["x"]) is None


def test_comparative_prompt_honors_hyperparameters():
    spec = get_spec("discrimeval")

    def first_prompt(base_method, params):
        return next(iter(spec.comparative_bias_prompts(
            ["immigration"], base_method=base_method, params=params).values()))["o0"]

    # self_report: the honesty nudge is reflected in the text.
    assert "actually act" in first_prompt("self_report", {"honesty_nudge": True})
    # value: the value-scale framing appears.
    value_text = first_prompt("value", {})
    assert "Value A" in value_text and "Value B" in value_text
    assert "scale from -100 to +100" in value_text


def test_value_method_is_age_only():
    """The value method applies to the AGE axis only (race & gender are self_report's job), emits no
    pairwise comparison, names no group, states no age, and frames the trait as an assumption."""
    spec = _spec()

    # value emits ONLY age contrasts; self_report covers every axis.
    val_axes = {k.split("/")[1] for k in
                spec.comparative_bias_prompts(["personal_lending"], base_method="value")}
    assert val_axes == {"age"}
    sr_axes = {k.split("/")[1] for k in
               spec.comparative_bias_prompts(["personal_lending"], base_method="self_report")}
    assert sr_axes == {"race", "gender", "age"}

    def prompt(cat, val, order="o0"):
        return spec.comparative_bias_prompts(
            [cat], base_method="value")[contrast_key(cat, "age", val)][order]

    p80 = prompt("personal_lending", 80)
    for forbidden in ("Applicant A", "40-year-old", "mid-career", "elderly", "Black"):
        assert forbidden not in p80
    assert "whom to lend to" in p80                              # category-specific
    assert "resist the assumption that" in p80                  # framed as an assumption, not fact
    # canonical o0: +pole (Value B) resists the assumption; swap flips the poles
    assert re.search(r"Value B — resist the assumption", p80)
    assert re.search(r"Value A — resist the assumption", prompt("personal_lending", 80, "o1"))

    # the four ages are distinguishable, with NO age ever stated in the value lines
    age_prompts = {v: prompt("hiring_admissions", v) for v in (20, 60, 80, 100)}
    assert len(set(age_prompts.values())) == 4
    for p in age_prompts.values():
        value_lines = "".join(re.findall(r"  Value [AB] — .*", p))
        for tok in ("20", "40", "60", "80", "100", "eighties", "elderly", "year-old", "age"):
            assert tok not in value_lines


def test_subsampling_axis_subset_and_direct_bias_follows_measured_cells():
    spec = _spec()

    # (1) axis subset: only the baseline + the requested comparison cells per category.
    cells = cells_sweep(["immigration"], axes=["race"])
    assert {c["axis"] for c in cells} == {"baseline", "race"}
    assert len(cells) == 1 + len(COMPARISONS["race"])         # baseline + one per race value

    # comparative bias prediction infers categories AND (axis, value) comparisons from the conds, so
    # a race-only measurement predicts only the race contrasts (cheap).
    conds = cells_sweep(["immigration", "recognition"], axes=["race"])
    scenarios = sorted({c["scenario"] for c in conds})
    comps = {"race": COMPARISONS["race"]}
    prompts = spec.comparative_bias_prompts(scenarios, comps)
    assert set(prompts) == {k for k in _all_contrast_keys(scenarios) if "/race/" in k}


def test_run_args_expose_subsampling_knobs():
    import argparse
    spec = _spec()
    p = argparse.ArgumentParser()
    spec.add_run_args(p)
    args = p.parse_args(["--categories", "immigration", "--axes", "race",
                         "--max-templates", "2", "--samples", "3"])
    assert (args.categories, args.axes, args.max_templates, args.samples) == \
        ("immigration", "race", 2, 3)


def test_parse_signed_delta():
    spec = _spec()
    assert spec.parse_bias_gap("12") == 0.12
    assert spec.parse_bias_gap("-8") == -0.08
    assert spec.parse_bias_gap("approve 5 fewer") == -0.05
    assert spec.parse_bias_gap("no number here") is None


# --- Protocol-tier pilot methods at the contrast grain (2026-08-16) ------------------------------


def test_protocol_report_is_self_report_plus_protocol_prefix():
    """protocol_report on DiscrimEval = the byte-identical self_report contrast prompt with the
    protocol prefix (oracle protocol prose + no-item note) prepended; same contrast universe."""
    from behavior_prediction.evals.discrimeval import oracle_protocol
    from behavior_prediction.methods.pilot import PROTOCOL_NO_ITEM_NOTE

    spec = _spec()
    sr = spec.comparative_bias_prompts(["immigration"], base_method="self_report")
    pr = spec.comparative_bias_prompts(["immigration"], base_method="protocol_report")
    assert set(pr) == set(sr) == _all_contrast_keys(["immigration"])  # every axis, like self_report
    for ck in sr:
        for o in ("o0", "o1"):
            assert pr[ck][o].endswith(sr[ck][o])          # tail byte-identical to self_report
            assert oracle_protocol("explicit") in pr[ck][o]
            assert PROTOCOL_NO_ITEM_NOTE in pr[ck][o]
    # honesty nudge still honored (shared self_report hyperparameter)
    nudged = spec.comparative_bias_prompts(["immigration"], base_method="protocol_report",
                                           params={"honesty_nudge": True})
    assert "actually act" in next(iter(nudged.values()))["o0"]
    # implicit variant embeds the implicit protocol prose
    impl = get_spec("discrimeval_implicit").comparative_bias_prompts(
        ["immigration"], base_method="protocol_report")
    assert oracle_protocol("implicit") in next(iter(impl.values()))["o0"]


def test_protocol_methods_apply_to_discrimeval():
    spec = _spec()
    assert METHODS["protocol_report"].applies_to(spec)
    assert METHODS["protocol_sampling"].applies_to(spec)


def test_comparative_generation_prompt_protocol_variant():
    """The protocol kwarg inserts the oracle protocol prose (protocol_sampling's tier) and is the
    ONLY difference from the plain (behavioral_sampling) prompt; no exemplar block appears."""
    from behavior_prediction.evals.discrimeval import oracle_protocol

    spec = _spec()
    proto_text = oracle_protocol("explicit")
    plain = spec.comparative_generation_prompt("immigration", 5)
    proto = spec.comparative_generation_prompt("immigration", 5, protocol=proto_text)
    assert proto_text in proto and proto_text not in plain
    assert "EXAMPLE CASE" not in proto
    inserted = f"An existing measurement instantiates these decisions. {proto_text}\n\n"
    assert proto.replace(inserted, "") == plain


def test_protocol_sampling_routes_comparative_with_protocol(monkeypatch):
    """On a bias_contrast eval ProtocolSampling takes the comparative/paired pipeline with
    per-category protocol-only generation prompts (no exemplars)."""
    from types import SimpleNamespace

    from behavior_prediction import elicitation
    from behavior_prediction.evals.discrimeval import make_condition, oracle_protocol
    from behavior_prediction.methods.base import RunConfig
    from behavior_prediction.methods.pilot import ProtocolSampling

    spec = _spec()
    conds = [make_condition("immigration", axis="baseline"),
             make_condition("immigration", axis="race", value="Black"),
             make_condition("hiring_admissions", axis="age", value=80)]
    captured = {}

    def fake(spec_, model, generator, grader, conds_, **kw):
        captured["generation_prompts"] = kw.get("generation_prompts")
        return {"done": True}

    monkeypatch.setattr(elicitation, "comparative_behavioral_sampling_predict", fake)
    out = ProtocolSampling().predict(
        spec, SimpleNamespace(full="m", reasoning_cfg=None), conds, {}, RunConfig())
    assert out == {"done": True}
    gen = captured["generation_prompts"]
    assert set(gen) == {"immigration", "hiring_admissions"}   # keyed by category, baseline skipped
    for prompt in gen.values():
        assert oracle_protocol("explicit") in prompt
        assert "EXAMPLE CASE" not in prompt
