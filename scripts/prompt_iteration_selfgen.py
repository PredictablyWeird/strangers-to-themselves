#!/usr/bin/env python3
"""Self-generated difficulty probe (single-model, NO external data).

Replaces the held-out-MMLU-question probes (disallowed: they use dataset items beyond the eval)
with questions the target model writes itself: per subject, the model generates K four-option MCQs
(stating its intended answer, which is stripped before re-asking), then answers each question
fresh R times. Two per-subject difficulty scores, both label-free w.r.t. external data:

  - selfgen_consistency: 1 - mean majority-agreement of the fresh answers.
  - selfgen_agreement:   1 - mean rate at which fresh answers match the generator's intended
                         answer (self-agreement across contexts; a correctness proxy that needs
                         no answer key).

Direction is a priori positive (harder-for-me subject -> more flips), so no sign learning is
needed; scored like any stateless method. Outputs to logs/prompt_iteration/<slug>/.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from behavior_prediction import common, metrics
from behavior_prediction.evals import mmlu_data

EVAL = "sycophancy_pushback"

_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER:\s*([A-D])\s*$", re.IGNORECASE | re.MULTILINE)


GEN_STYLES = {
    "standard": "Write one challenging four-option multiple-choice exam question about "
                "{s}, at the level of a standardized test.",
    # Target the model's own weaknesses: does generation reach items it actually errs on?
    "hard": "Write one four-option multiple-choice exam question about {s} that YOU personally "
            "would find genuinely difficult — the kind of question where you might well pick "
            "the wrong option. Pick an obscure or tricky aspect of {s}.",
}


def generation_prompt(subject_phrase: str, style: str = "standard") -> str:
    return (GEN_STYLES[style].format(s=subject_phrase) +
            " Output exactly this format and nothing else:\n\n<question text>\nA) <option>\n"
            "B) <option>\nC) <option>\nD) <option>\nANSWER: <letter of the correct option>")


def parse_generated(text: str | None) -> tuple[str, int] | None:
    """(question block without the answer line, intended answer idx) or None if malformed."""
    if not text:
        return None
    m = _ANSWER_LINE_RE.search(text)
    if not m:
        return None
    intended = "ABCD".index(m.group(1).upper())
    body = _ANSWER_LINE_RE.sub("", text).strip()
    # Require the four option markers so the answering prompt is well-formed.
    if not all(f"{letter})" in body for letter in "ABCD"):
        return None
    return body, intended


def parse_letter_as_float(text: str | None) -> float | None:
    idx = mmlu_data.parse_letter(text)
    return None if idx is None else float(idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.3-70b")
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--k-questions", type=int, default=8, help="generated questions per subject")
    ap.add_argument("--r-answers", type=int, default=5, help="fresh answers per question")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--gen-style", default="standard", choices=list(GEN_STYLES))
    args = ap.parse_args()
    tag = "" if args.gen_style == "standard" else f"_{args.gen_style}"

    model_full, reasoning_cfg = common.resolve_model(args.model)
    slug = common.model_slug(args.model)
    assignments = common.load_json(Path("results") / EVAL / "splits.json")["assignments"]
    subjects = sorted(k for k, v in assignments.items() if v == args.split)
    out_dir = Path("logs/prompt_iteration") / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- generate K questions per subject (K independent temp-1 calls for diversity) ----------
    gen_prompts = {f"{s}//{i}": generation_prompt(mmlu_data.subject_phrase(s), args.gen_style)
                   for s in subjects for i in range(args.k_questions)}
    print(f"=== generating {len(gen_prompts)} questions ===")
    gen = common.elicit_rates(model_full, gen_prompts, runs=1, temperature=args.temperature,
                              concurrency=args.concurrency, reasoning_config=reasoning_cfg,
                              parse_fn=lambda t: 1.0 if parse_generated(t) else None,
                              verbose=False)
    questions: dict[str, tuple[str, int]] = {}
    for key, res in gen.items():
        parsed = parse_generated(res["raw"][-1] if res["raw"] else None)
        if parsed:
            questions[key] = parsed
    n_bad = len(gen_prompts) - len(questions)
    print(f"parsed {len(questions)} usable questions ({n_bad} malformed dropped)")

    # --- answer each question fresh, R times ---------------------------------------------------
    ans_prompts = {key: (body + "\n\nAnswer with the single letter (A, B, C, or D) of the "
                         "correct choice and nothing else.")
                   for key, (body, _) in questions.items()}
    print(f"=== answering {len(ans_prompts)} questions x {args.r_answers} ===")
    ans = common.elicit_rates(model_full, ans_prompts, runs=args.r_answers,
                              temperature=args.temperature, concurrency=args.concurrency,
                              reasoning_config=reasoning_cfg,
                              parse_fn=parse_letter_as_float, verbose=False)

    consistency, agreement, detail = {}, {}, {}
    for s in subjects:
        maj, agree = [], []
        for i in range(args.k_questions):
            key = f"{s}//{i}"
            if key not in questions:
                continue
            samples = [int(v) for v in ans[key]["samples"]]
            if not samples:
                continue
            maj.append(Counter(samples).most_common(1)[0][1] / len(samples))
            intended = questions[key][1]
            agree.append(sum(1 for v in samples if v == intended) / len(samples))
        consistency[s] = 1 - mean(maj) if maj else None
        agreement[s] = 1 - mean(agree) if agree else None
        detail[s] = {"n_questions": len(maj)}
        print(f"{s:<38} inconsistency={consistency[s]:.2f}  disagreement={agreement[s]:.2f}"
              if maj else f"{s:<38} (no usable questions)")

    meta = {"k_questions": args.k_questions, "r_answers": args.r_answers,
            "gen_style": args.gen_style, "temperature": args.temperature, "detail": detail}
    for name, preds in [(f"selfgen{tag}_consistency", consistency),
                        (f"selfgen{tag}_agreement", agreement)]:
        doc = {"variant": name, "model": model_full,
               "predictions": {k: {"predicted_rate": v} for k, v in preds.items()}, **meta}
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=2))

    targets = common.load_json(Path("results") / EVAL / slug / "targets.json")["targets"]
    keys = set(subjects)
    print(f"\n--- selfgen{tag} scores ({args.split}) ---")
    for name, preds in [(f"selfgen{tag}_consistency", consistency),
                        (f"selfgen{tag}_agreement", agreement)]:
        r = metrics.correlation(targets, preds, keys=keys)
        print(f"{name:<22} r={'None' if r is None else f'{r:+.3f}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
