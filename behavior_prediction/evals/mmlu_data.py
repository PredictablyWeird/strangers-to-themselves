"""Shared MMLU loader for the capability and sycophancy-pushback evals.

MMLU (``cais/mmlu`` on the Hub) is 57 multiple-choice subjects, 4 options each. We sample
deterministically — the first ``n`` questions per subject in dataset order — so every model
under test sees the same set and the measured accuracy is comparable. Pure helpers
(``mcq_prompt``, ``parse_letter``, ``subjects``) carry no network dependency and are unit-tested;
``load_questions`` hits the Hub (cached locally after first download).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LETTERS: tuple[str, ...] = ("A", "B", "C", "D")

# The 57 MMLU subjects (the dataset's config names). Hard-coded so the split universe is
# enumerable with no network call.
_SUBJECTS: tuple[str, ...] = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions",
)


@dataclass(frozen=True)
class Question:
    subject: str
    stem: str
    choices: list[str]
    answer_idx: int


def subjects() -> list[str]:
    """Canonical sorted list of the 57 MMLU subjects (model-independent split universe)."""
    return sorted(_SUBJECTS)


def subject_phrase(subject: str) -> str:
    """Human-readable subject name: underscores replaced with spaces."""
    return subject.replace("_", " ")


def load_questions(subject: str, n: int) -> list[Question]:
    """First ``n`` test questions for ``subject`` in dataset order (deterministic, no shuffle)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", subject, split="test")
    out: list[Question] = []
    for row in ds.select(range(min(n, len(ds)))):
        out.append(Question(subject=subject, stem=row["question"],
                            choices=list(row["choices"]), answer_idx=int(row["answer"])))
    return out


def mcq_prompt(q: Question) -> str:
    lines = [f"{LETTERS[i]}) {c}" for i, c in enumerate(q.choices)]
    return (
        f"{q.stem}\n\n" + "\n".join(lines) + "\n\n"
        "Answer with the single letter (A, B, C, or D) of the correct choice and nothing else."
    )


# Reasoning models wrap their chain-of-thought in <think>...</think>; drop it before parsing.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# An explicit final-answer declaration: "answer is B", "Answer: a", "answer = C", "\boxed{D}".
# The lazy [^A-D]{0,12} keeps the keyword and the letter close, so an incidental option letter
# elsewhere ("Option A is wrong") isn't mistaken for the answer.
_ANSWER_RE = re.compile(r"(?:answer|boxed)\b[^A-D\n]{0,12}?\(?\{?\s*([A-D])\b", re.IGNORECASE)
# A standalone A-D token (fallback when there's no explicit declaration).
_LETTER_TOKEN_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def parse_letter(text: str | None) -> int | None:
    """Index 0-3 of the model's *final* chosen letter, or ``None`` if absent/ambiguous.

    Robust to verbose / reasoning output: strips ``<think>`` blocks, then prefers the **last**
    explicit answer declaration (``answer is B`` / ``Answer: C`` / ``\\boxed{D}``) — last so a
    model that changes its mind is scored on its final answer. Failing that, returns the letter
    only when exactly one distinct A-D appears as a standalone token (so a bare ``"B"`` parses
    but ``"Both A and B"`` stays ambiguous → ``None``, which the caller retries rather than
    scoring a wrong guess from the first letter it happened to mention)."""
    if not text:
        return None
    s = _THINK_RE.sub(" ", text).strip()
    markers = _ANSWER_RE.findall(s)
    if markers:
        return LETTERS.index(markers[-1].upper())
    distinct = {m.upper() for m in _LETTER_TOKEN_RE.findall(s)}
    if len(distinct) == 1:
        return LETTERS.index(next(iter(distinct)))
    return None
