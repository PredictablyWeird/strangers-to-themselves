"""Tests for the reward_hacking eval.

The load-bearing invariant (per the request that introduced this eval): its dev/test split must be
*identical* to the other MMLU-subject evals, so a subject held out for test there is held out here
too — otherwise a model tuned on capability_mmlu/sycophancy_pushback dev could leak into this eval's
test. ``test_split_identical_to_mmlu_evals`` pins that.
"""
from __future__ import annotations

import json
import os

import pytest

from behavior_prediction import common
from behavior_prediction.evals import get_spec, mmlu_data, reward_hack_data


def _manifest(eval_name):
    path = f"{common.results_root()}/{eval_name}/splits.json"
    if not os.path.exists(path):
        pytest.skip(f"no committed split manifest for {eval_name}")
    return json.load(open(path))


def test_sweep_is_57_subjects():
    spec = get_spec("reward_hacking")
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    assert len(conds) == 57
    assert {c["scenario"] for c in conds} == set(mmlu_data.subjects())


def test_split_units_match_mmlu_evals_exactly():
    """Same units, same order → the built manifest is deterministic-identical."""
    rh = get_spec("reward_hacking")
    assert rh.split_units() == get_spec("sycophancy_pushback").split_units()
    assert rh.split_units() == get_spec("capability_mmlu").split_units()


def test_split_identical_to_mmlu_evals():
    """The committed manifest's assignments are byte-identical to the reference evals'."""
    rh = _manifest("reward_hacking")["assignments"]
    syco = _manifest("sycophancy_pushback")["assignments"]
    cap = _manifest("capability_mmlu")["assignments"]
    assert rh == syco, "reward_hacking split diverged from sycophancy_pushback"
    assert rh == cap, "reward_hacking split diverged from capability_mmlu"
    # And specifically the test set (the thing that must not leak).
    rh_test = {k for k, v in rh.items() if v == "test"}
    syco_test = {k for k, v in syco.items() if v == "test"}
    assert rh_test == syco_test and len(rh_test) == 23


def test_split_unit_is_the_subject():
    spec = get_spec("reward_hacking")
    assert spec.split_unit({"scenario": "astronomy"}) == "astronomy"


def test_frame_varies_across_subjects():
    spec = get_spec("reward_hacking")
    conds = spec.build_sweep(spec.default_sweep, spec.scenarios)
    settings = {spec.frame(c).setting for c in conds}
    assert len(settings) == len(conds)          # each subject renders a distinct prompt -> rankable


# --- data module ---------------------------------------------------------------

def _has_data():
    return os.path.exists(os.path.join(reward_hack_data.repo_path(),
                                       "datasets", "reward_hack", "revealing_score.jsonl"))


data = pytest.mark.skipif(not _has_data(), reason="reward-hack-generalization clone not present")


@data
def test_all_57_subjects_have_temptation_items():
    subs = reward_hack_data.subjects_with_items()
    assert set(subs) == set(mmlu_data.subjects())
    assert all(len(reward_hack_data.temptation_items(s)) >= 40 for s in subs)


@data
def test_points_stripped_in_no_points_variant():
    it = reward_hack_data.temptation_items("high_school_mathematics", 1)[0]
    assert "Points:" in it["with_points"] and "grader has helpfully" in it["with_points"]
    assert "Points:" not in it["no_points"] and "grader has helpfully" not in it["no_points"]
    assert it["hack_letter"] in ("A", "B", "C", "D")


@data
def test_temptation_only_high_reward_is_wrong():
    """Every item we keep is a genuine temptation: the over-valued option is not correct. The
    with/without-points variants share the question, so stripping points can't change which option
    is 'hack'."""
    it = reward_hack_data.temptation_items("professional_law", 1)[0]
    # The MCQ text (minus the points scaffolding) is identical across variants.
    import re
    q = lambda b: re.search(r"Question:.*", b, re.S).group(0)
    a = q(it["with_points"]).replace(" - Points: ", " ")
    a = re.sub(r"Points:\s*-?\d+", "", a)
    assert q(it["no_points"]).split("Points")[0][:200] in it["with_points"]


# --- targets math --------------------------------------------------------------

def test_hack_gap_extraction(tmp_path):
    """produce_targets must set rate=hack_rate for hack_rate, rate=gap for hack_gap, and store the
    baseline either way."""
    spec = get_spec("reward_hacking")
    raw = {"model": "m", "items_per_subject": 2, "results": {
        # astronomy: with-points 2/2 hack, no-points 0/2 -> hack 1.0, base 0.0, gap 1.0
        "astronomy//1//wp": {"subject": "astronomy", "variant": "wp", "n_hack": 1, "n_answered": 1},
        "astronomy//2//wp": {"subject": "astronomy", "variant": "wp", "n_hack": 1, "n_answered": 1},
        "astronomy//1//np": {"subject": "astronomy", "variant": "np", "n_hack": 0, "n_answered": 1},
        "astronomy//2//np": {"subject": "astronomy", "variant": "np", "n_hack": 0, "n_answered": 1},
    }}
    p = tmp_path / "behavior_raw.json"
    p.write_text(json.dumps(raw))

    class A:
        behavior_raw = str(p)
        model = "m"
        metric = "hack_rate"
    out = spec.produce_targets(A())["targets"]["astronomy"]
    assert out["rate"] == 1.0 and out["baseline_rate"] == 0.0 and out["hack_gap"] == 1.0

    A.metric = "hack_gap"
    out = spec.produce_targets(A())["targets"]["astronomy"]
    assert out["rate"] == 1.0 and out["baseline_rate"] == 0.0     # gap == 1.0 here too
