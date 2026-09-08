"""MMLU prompt building + letter parsing + subject universe (pure, no network)."""
from __future__ import annotations

from behavior_prediction.evals import mmlu_data as M


def test_subjects_are_57_sorted_unique():
    subs = M.subjects()
    assert len(subs) == 57
    assert subs == sorted(subs)
    assert len(set(subs)) == 57
    assert "anatomy" in subs


def test_mcq_prompt_lists_all_choices_with_letters():
    q = M.Question(subject="anatomy", stem="What is the largest organ?",
                   choices=["Heart", "Skin", "Liver", "Brain"], answer_idx=1)
    p = M.mcq_prompt(q)
    assert "What is the largest organ?" in p
    assert "A) Heart" in p and "B) Skin" in p and "D) Brain" in p
    assert "single letter" in p.lower()


def test_parse_letter_variants():
    assert M.parse_letter("B") == 1
    assert M.parse_letter("The answer is C.") == 2
    assert M.parse_letter("Answer: a") == 0
    assert M.parse_letter("D)") == 3
    assert M.parse_letter("I am not sure") is None
    assert M.parse_letter(None) is None
    assert M.parse_letter("Both A and B") is None  # ambiguous → None (caller retries)


def test_parse_letter_reasoning_output():
    # Verbose / reasoning output mentions several option letters before the final answer.
    # The parser must take the FINAL declared answer, not the first letter it sees.
    assert M.parse_letter(
        "Let me think. Option A is wrong, C is wrong. Comparing B and D... the answer is B."
    ) == 1
    assert M.parse_letter("<think>weigh A vs B vs C vs D</think> Final answer: B") == 1
    assert M.parse_letter("Eliminating A and C, it's down to B. \\boxed{B}") == 1
    # changes its mind: last answer declaration wins
    assert M.parse_letter("The answer is A. Wait, no — the answer is D.") == 3
    # no answer declaration and multiple distinct letters → ambiguous → None (not a wrong guess)
    assert M.parse_letter("It could be A or C, hard to say.") is None
