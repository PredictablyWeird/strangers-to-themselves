"""Self-vs-generic phrasing ablation: the prompt-level
guarantees the design pre-specifies.

* every ``*_3p`` Frame slot and every ``protocol_3p`` is free of second-person pronouns;
* arms A/B (and C/D) differ ONLY in the final-question substring — the description is
  byte-identical within a framing;
* arms A/C share the final question byte-for-byte; B/D differ only by the "not you" clause;
* the ``frame_3p`` / ``oracle_3p`` flags the adapters gate on match the actual slots;
* the ablation methods are registered but never in ``DEFAULT_METHODS``.

Offline: conditions come from the committed targets of one pool model (dev split only, so no
test-split text is ever rendered here); oracle items come from the evals whose exhibits need no
harness clone (the MMLU family + DiscrimEval, which needs the HF dataset cache, skipped if absent).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from behavior_prediction import common, elicitation, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.methods import DEFAULT_METHODS, REGISTRY
from behavior_prediction.tuning import ACTIVE_EVALS

MODEL = "llama-3.3-70b"
SECOND_PERSON = re.compile(r"\b(you|your|yours|yourself)\b", re.I)
ARMS = ["self_report_genmin", "self_report_3p", "self_report_3p_gen",
        "informed_oracle_genmin", "informed_oracle_3p", "informed_oracle_3p_gen"]


def _dev_conditions(ev: str) -> list[dict]:
    spec = get_spec(ev)
    p = Path(f"results/{ev}/{MODEL}/targets.json")
    if not p.exists():
        pytest.skip(f"no targets for {ev}/{MODEL}")
    targets = common.load_json(p)["targets"]
    dev = splits.split_keys(spec, targets, splits.load_manifest(ev), "dev")
    return [targets[k]["condition"] for k in sorted(dev)]


def _is_bias(spec) -> bool:
    return getattr(spec, "scoring_semantics", "") == "bias_contrast"


def _report_arms(spec, cond) -> dict[str, str] | None:
    if _is_bias(spec):
        from behavior_prediction.evals.discrimeval import bias_prompt
        if spec.condition_contrast(cond) is None:
            return None
        kw = dict(implicit=spec.config == "implicit")
        return {a: bias_prompt(cond["scenario"], cond["axis"], cond["value"], base_method=m, **kw)
                for a, m in [("A", "self_report"), ("B", "self_report_genmin"),
                             ("C", "self_report_3p"), ("D", "self_report_3p_gen")]}
    e = elicitation
    return {"A": e.self_report_prompt(spec, cond), "B": e.self_report_genmin_prompt(spec, cond),
            "C": e.self_report_3p_prompt(spec, cond), "D": e.self_report_3p_gen_prompt(spec, cond)}


def _split_question(prompt: str) -> tuple[str, str]:
    """(description, final question) — the question starts at the last 'Out of 100'."""
    i = prompt.rfind("Out of 100")
    assert i > 0, prompt
    return prompt[:i], prompt[i:]


def test_registered_but_not_default():
    for m in ARMS:
        assert m in REGISTRY
        assert m not in DEFAULT_METHODS


@pytest.mark.parametrize("ev", ACTIVE_EVALS)
def test_frame_3p_slots(ev):
    spec = get_spec(ev)
    conds = _dev_conditions(ev)
    assert spec.frame_3p, f"{ev}: frame_3p flag not set"
    for c in conds:
        f = spec.frame(c)
        assert f.has_3p(), (ev, spec.condition_key(c))
        for slot in ("setting_3p", "self_report_situation_3p", "target_action_3p"):
            text = getattr(f, slot)
            assert not SECOND_PERSON.search(text), (ev, slot, text)
            # Same information, same order: the rewrite stays within ±15% of the 2p length
            # (the design says ±10%; "the assistant"/"the agent" for "you" costs a few chars).
            ref = len(getattr(f, slot[:-3]))
            assert abs(len(text) - ref) <= max(12, 0.15 * ref), (ev, slot, len(text), ref)


@pytest.mark.parametrize("ev", ACTIVE_EVALS)
def test_report_arms_are_minimal_pairs(ev):
    spec = get_spec(ev)
    seen = 0
    for c in _dev_conditions(ev):
        arms = _report_arms(spec, c)
        if arms is None:
            continue
        seen += 1
        dA, qA = _split_question(arms["A"])
        dB, qB = _split_question(arms["B"])
        dC, qC = _split_question(arms["C"])
        dD, qD = _split_question(arms["D"])
        assert dA == dB, (ev, "A/B descriptions differ")          # W1: subject swap only
        assert dC == dD, (ev, "C/D descriptions differ")          # W2: the clean minimal pair
        assert qA == qC, (ev, "A/C final questions differ")       # the self question is fixed
        assert not SECOND_PERSON.search(dC), (ev, dC)             # reader absent before the Q
        assert "not you specifically" in qB and "not you specifically" not in qD
        assert qB.replace(" — not you specifically —", "") == qD, (ev, qB, qD)
        assert elicitation.HONESTY_NUDGE not in arms["A"]          # nudge off in every arm
        assert not SECOND_PERSON.search(qD), (ev, qD)             # D: no reader anywhere
    assert seen > 0


ORACLE_EVALS = ["capability_mmlu", "sycophancy_pushback", "reward_hacking", "discrimeval",
                "mask_subdomain_pressure"]


def _has_reward_hack_data() -> bool:
    from behavior_prediction.evals import reward_hack_data
    return Path(reward_hack_data.repo_path(), "datasets", "reward_hack",
                "revealing_score.jsonl").exists()


@pytest.mark.parametrize("ev", ORACLE_EVALS)
def test_oracle_3p_views(ev):
    # reward_hacking renders its exhibits from the reward-hack-generalization clone; without it
    # oracle_items exits rather than returning nothing, so skip the same way the tail does.
    if ev == "reward_hacking" and not _has_reward_hack_data():
        pytest.skip("reward-hack-generalization clone not present")
    spec = get_spec(ev)
    assert spec.oracle_3p
    n = 0
    for c in _dev_conditions(ev)[:6]:
        view = spec.oracle_items(c, 2) or {}
        if not view.get("items"):
            continue
        assert not SECOND_PERSON.search(view["protocol_3p"]), (ev, view["protocol_3p"])
        for it in view["items"]:
            n += 1
            assert "{subject}" in it["ask_gen"], (ev, it["ask_gen"])
            # After the subject slot only third-person pronouns (capability's predictor-directed
            # "You may work through ..." sentence is the one allowed exception).
            body = it["ask_gen"].split("{subject}", 1)[1].replace(
                "You may work through the question first if that helps your estimate.", "")
            assert not SECOND_PERSON.search(body), (ev, it["ask_gen"])
            # The four oracle prompts render and keep the exhibit byte-identical.
            e = elicitation
            ps = [e.informed_oracle_prompt(view["protocol"], it["exhibit"], it["ask"]),
                  e.informed_oracle_genmin_prompt(view["protocol"], it["exhibit"], it["ask_gen"]),
                  e.informed_oracle_3p_prompt(view["protocol_3p"], it["exhibit"], it["ask"]),
                  e.informed_oracle_3p_gen_prompt(view["protocol_3p"], it["exhibit"], it["ask_gen"])]
            blocks = {p.split("----- BEGIN MEASURED ITEM -----")[1]
                       .split("----- END MEASURED ITEM -----")[0] for p in ps}
            assert len(blocks) == 1
            # C' and D' share everything up to the ask.
            assert ps[2].split("----- END MEASURED ITEM -----")[0] == \
                ps[3].split("----- END MEASURED ITEM -----")[0]
    if n == 0:
        pytest.skip(f"{ev}: no oracle items renderable offline")


def test_adapters_apply_where_flagged():
    for ev in ACTIVE_EVALS:
        spec = get_spec(ev)
        for m in ARMS:
            applies = REGISTRY[m].applies_to(spec)
            if m.startswith("self_report"):
                assert applies == spec.frame_3p
            else:
                assert applies == (spec.oracle_3p and REGISTRY["informed_oracle"].applies_to(spec))
