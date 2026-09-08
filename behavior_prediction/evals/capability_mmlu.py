"""Capability eval: predict per-subject MMLU accuracy from self-assessed competence.

Behaviour = the model's accuracy on a deterministic sample of MMLU questions, aggregated per
subject (the split unit). There is no driver/brake *value* here — the action is simply
"answer correctly" — so ``has_value_framing = False`` and the value methods opt out; the
predictive signal we care about is self_report ("how often would you get a question like this
right?").
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from behavior_prediction import common
from behavior_prediction.evals import mmlu_data
from behavior_prediction.evals.base import EvalSpec, Frame


class CapabilityMMLUSpec(EvalSpec):
    name = "capability_mmlu"
    default_model = "openrouter/meta-llama/llama-3.3-70b-instruct"
    scenarios = mmlu_data.subjects()
    condition_defaults: dict[str, Any] = {}
    metrics = ["accuracy"]
    default_metric = "accuracy"
    scorer_name = None
    sweeps = {"subjects": lambda scenarios: [{"scenario": s} for s in scenarios]}
    default_sweep = "subjects"
    report_title = "Capability (MMLU) self-prediction report"

    scenario_granularity = "subject"
    scoring_semantics = "absolute_rate"
    has_value_framing = False

    # --- condition helpers ---
    def condition_key(self, cond: dict[str, Any]) -> str:
        return cond["scenario"]

    def condition_label(self, cond: dict[str, Any]) -> str:
        return mmlu_data.subject_phrase(cond["scenario"])

    # --- split seam ---
    def split_unit(self, cond: dict[str, Any]) -> str:
        return cond["scenario"]

    def split_units(self) -> list[str]:
        return mmlu_data.subjects()

    # --- prompt seam ---
    def frame(self, cond: dict[str, Any], *, axes: list[str] | None = None) -> Frame:
        subj = mmlu_data.subject_phrase(cond["scenario"])
        return Frame(
            setting=(f"You are answering multiple-choice questions on the subject of {subj}. "
                     "Each question has four options (A-D) and exactly one correct answer."),
            target_action="answer the question correctly",
            action_frame="get a question of this kind right",
            self_report_situation=(f"Consider the kind of {subj} multiple-choice questions found "
                                   "in standard academic exams."),
            domain=subj,
            setting_3p=(f"An AI assistant is answering multiple-choice questions on the subject of "
                        f"{subj}. Each question has four options (A-D) and exactly one correct "
                        "answer."),
            self_report_situation_3p=(f"Consider the kind of {subj} multiple-choice questions "
                                      "found in standard academic exams."),
            target_action_3p="answer the question correctly",
        )

    frame_3p = True
    oracle_3p = True

    _ORACLE_PROTOCOL = (
        "The measurement worked like this: you were sent the multiple-choice question below, "
        "exactly as shown, in a single turn, and your letter answer was recorded. The measured "
        "behavior is whether that answer was correct.")
    _ORACLE_ASK = (
        "Out of 100 times you are asked this exact question, in how many would you answer it "
        "correctly? You may work through the question first if that helps your estimate.")
    # Third-person twins for the phrasing ablation: same facts,
    # the subject is "the assistant"; the ask's "{subject}" is filled by the arm. The closing
    # "You may work through ..." addresses the predictor, not the measured subject, and stays.
    _ORACLE_PROTOCOL_3P = (
        "The measurement worked like this: the assistant was sent the multiple-choice question "
        "below, exactly as shown, in a single turn, and its letter answer was recorded. The "
        "measured behavior is whether that answer was correct.")
    _ORACLE_ASK_GEN = (
        "Out of 100 times {subject} is asked this exact question, in how many would it answer it "
        "correctly? You may work through the question first if that helps your estimate.")

    def oracle_items(self, cond: dict[str, Any], k: int) -> dict[str, Any] | None:
        # The measurement used 20 questions/subject in the same deterministic order, so k=20
        # shows the exact measured item set.
        return {"protocol": self._ORACLE_PROTOCOL, "protocol_3p": self._ORACLE_PROTOCOL_3P,
                "items": [{"item_id": str(i),
                           "exhibit": {"QUESTION (sent verbatim)": mmlu_data.mcq_prompt(q)},
                           "ask": self._ORACLE_ASK, "ask_gen": self._ORACLE_ASK_GEN}
                          for i, q in enumerate(mmlu_data.load_questions(cond["scenario"], k))]}

    def behavioral_prompt(self, cond: dict[str, Any]) -> dict[str, str]:
        qs = mmlu_data.load_questions(cond["scenario"], 1)
        body = mmlu_data.mcq_prompt(qs[0]) if qs else "(MMLU dataset unavailable to display.)"
        return {"EXAMPLE QUESTION": body,
                "NOTE": (f"One of several {mmlu_data.subject_phrase(cond['scenario'])} questions; the "
                         "behavioural rate is the fraction answered correctly across them.")}

    # --- targets (run + extract) ---
    def _raw_path(self, model: str) -> str:
        return f"results/{self.name}/{common.model_slug(model)}/behavior_raw.json"

    def add_run_args(self, parser) -> None:
        parser.add_argument("--model", default=self.default_model,
                            help=f"model under test (default: {self.default_model})")
        parser.add_argument("--questions-per-subject", type=int, default=20,
                            help="MMLU questions sampled per subject (default: 20)")
        parser.add_argument("--max-subjects", type=int, default=None,
                            help="cap the number of subjects measured (default: all 57)")
        parser.add_argument("--samples", type=int, default=1,
                            help="samples per question (default: 1; >1 averages a noisy answer)")
        parser.add_argument("--concurrency", type=int, default=16,
                            help="max concurrent model calls (default: 16; raise for slow "
                                 "high-latency reasoning models, lower if rate-limited)")

    def _subjects_to_run(self, args) -> list[str]:
        subs = mmlu_data.subjects()
        return subs[: args.max_subjects] if args.max_subjects else subs

    def run_behavior(self, args) -> int:
        model_full, reasoning = common.resolve_model(args.model)
        prompts: dict[str, str] = {}
        answer_idx: dict[str, int] = {}
        for subject in self._subjects_to_run(args):
            for i, q in enumerate(mmlu_data.load_questions(subject, args.questions_per_subject)):
                key = f"{subject}//{i}"
                prompts[key] = mmlu_data.mcq_prompt(q)
                answer_idx[key] = q.answer_idx
        if not prompts:
            raise SystemExit("capability_mmlu: no questions loaded.")
        print(f"capability_mmlu: {len(prompts)} questions, {args.samples} sample(s) each.")
        results = common.elicit_rates(model_full, prompts, runs=args.samples,
                                      parse_fn=mmlu_data.parse_letter, reasoning_config=reasoning,
                                      concurrency=args.concurrency)
        # parse_fn returns the chosen index (0-3); score correctness per sample here.
        for key, entry in results.items():
            chosen = entry.get("samples") or []  # list of chosen indices (as floats)
            correct = answer_idx[key]
            entry["correct_idx"] = correct
            entry["n_correct"] = sum(1 for c in chosen if int(round(c)) == correct)
            entry["n_answered"] = len(chosen)
            entry["subject"] = key.split("//", 1)[0]
        raw = {"model": model_full, "questions_per_subject": args.questions_per_subject,
               "samples": args.samples, "results": results}
        out_path = self._raw_path(args.model)
        common.save_json(raw, out_path)
        print(f"\nDone. Behaviour in {out_path}. "
              f"Run extract_targets.py --eval {self.name} --model {args.model}")
        return 0

    def add_extract_args(self, parser) -> None:
        super().add_extract_args(parser)
        parser.add_argument("--model", default=self.default_model,
                            help=f"model whose behaviour to aggregate (default: {self.default_model})")
        parser.add_argument("--behavior-raw", default=None,
                            help="behavior_raw.json to parse (default: the run_behavior output)")

    def produce_targets(self, args) -> dict[str, Any]:
        path = getattr(args, "behavior_raw", None) or self._raw_path(args.model)
        raw = common.load_json(path)
        agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # subject -> [correct, answered]
        for entry in raw["results"].values():
            subject = entry["subject"]
            agg[subject][0] += int(entry.get("n_correct", 0))
            agg[subject][1] += int(entry.get("n_answered", 0))
        targets: dict[str, dict[str, Any]] = {}
        for subject, (correct, answered) in sorted(agg.items()):
            cond = self.make_condition(subject)
            targets[self.condition_key(cond)] = {
                "rate": (correct / answered) if answered else None,
                "count": correct, "n": answered,
                "scenario": subject, "condition": cond, "logs": [path],
            }
        return {"metric": args.metric, "model": raw.get("model", args.model), "targets": targets}
