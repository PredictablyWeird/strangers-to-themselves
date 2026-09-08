"""Comparative (paired) behavioral_sampling on a bias_contrast eval (DiscrimEval): generate
demographic-neutral cases, sample the model's decision under each demographic on the SAME case,
grade, and difference per case. The generator / model-under-test / grader calls are faked so this
stays offline."""

from __future__ import annotations

import pytest

from behavior_prediction import common
from behavior_prediction.elicitation import (_SCENARIO_MARKER,
                                             comparative_behavioral_sampling_predict)
from behavior_prediction.evals import get_spec
from behavior_prediction.evals.discrimeval import contrast_key, make_condition


class _Output:
    def __init__(self, completion: str) -> None:
        self.completion = completion


def _fake_models(monkeypatch, *, approve_for):
    """Route generator -> k cases, model-under-test -> a reply whose text encodes whether it
    approved (so the grader, also faked, can read it back), grader -> yes/no.

    ``approve_for(descriptor) -> bool`` decides whether the model-under-test approves for a given
    applicant profile. The prompt for the model-under-test contains ``Applicant profile: <desc>.``;
    we parse that back out so behaviour depends on the demographic, identically across cases."""
    import inspect_ai.model as im
    import re

    k_marker = _SCENARIO_MARKER

    class _Model:
        def __init__(self, role):
            self.role = role

        async def generate(self, prompt, config=None):
            if self.role == "gen":
                # Produce 4 distinct neutral cases.
                return _Output("".join(f"{k_marker}\nCase {i}: the applicant requests X.\n"
                                       for i in range(4)))
            if self.role == "behave":
                m = re.search(r"Applicant profile: (.+?)\.", prompt)
                desc = m.group(1) if m else ""
                return _Output("APPROVED" if approve_for(desc) else "DENIED")
            # grader: read the behave reply embedded in the grade prompt
            return _Output("yes" if "APPROVED" in prompt else "no")

    def get_model(name):
        if name == "gen-model":
            return _Model("gen")
        if name == "grader-model":
            return _Model("grader")
        return _Model("behave")

    monkeypatch.setattr(im, "get_model", get_model)


def _dev_conds(n_cats=2):
    spec = get_spec("discrimeval")
    cats = ["academic_review", "hiring_admissions"][:n_cats]
    conds = []
    for cat in cats:
        conds.append(make_condition(cat, axis="baseline"))
        conds.append(make_condition(cat, axis="age", value=80))
        conds.append(make_condition(cat, axis="race", value="Black"))
    return spec, conds


def test_comparative_paired_gap_is_contrast_keyed(monkeypatch):
    spec, conds = _dev_conds()
    # Model approves the baseline (40yo white man) but denies the 80-year-old: a clean +bias for
    # the baseline / negative gap for the age contrast. Race held neutral (approves both).
    def approve_for(desc):
        if "80-year-old" in desc:
            return False
        return True
    _fake_models(monkeypatch, approve_for=approve_for)

    preds = comparative_behavioral_sampling_predict(
        spec, "behave-model", "gen-model", "grader-model", conds,
        k=4, r=1, retries=0, temperature=1.0, concurrency=4)

    # Keyed by CONTRAST (not cell), one entry per non-baseline cell.
    age_key = contrast_key("academic_review", "age", 80)
    race_key = contrast_key("academic_review", "race", "Black")
    assert set(preds) == {
        contrast_key(c["scenario"], c["axis"], c["value"])
        for c in conds if c["axis"] != "baseline"}
    # 80yo denied while baseline approved -> gap = approve(80) - approve(40) = 0 - 1 = -1 on each case.
    assert preds[age_key]["predicted_rate"] == -1.0
    assert preds[age_key]["n"] == 4            # paired over the 4 generated cases
    # Black approved like the baseline -> zero gap.
    assert preds[race_key]["predicted_rate"] == 0.0


def test_comparative_predictions_score_via_direct_contrast_path(monkeypatch):
    """The contrast-keyed gaps feed score_contrasts' direct path (not per-cell differencing)."""
    spec, conds = _dev_conds(n_cats=1)
    _fake_models(monkeypatch, approve_for=lambda desc: "80-year-old" not in desc)
    preds = comparative_behavioral_sampling_predict(
        spec, "behave-model", "gen-model", "grader-model", conds,
        k=2, r=1, retries=0, temperature=1.0, concurrency=2)
    # Build minimal targets (cell rates) so score_contrasts can run; predicted comes from our gaps.
    targets = {}
    for c in conds:
        from behavior_prediction.evals.discrimeval import condition_key
        targets[condition_key(c)] = {"condition": c, "rate": 0.5}
    flat = {k: v["predicted_rate"] for k, v in preds.items()}
    scored = spec.score_contrasts(targets, flat)
    age_key = contrast_key("academic_review", "age", 80)
    assert age_key in scored
    actual, predicted = scored[age_key]
    assert predicted == -1.0      # used our direct gap, not a differenced per-cell rate
