"""Review dump for the self-vs-generic phrasing ablation.

For every active eval, prints each distinct (2nd-person, 3rd-person) slot pair the dev
conditions render — setting, self_report_situation, target_action — side by side, then the
four report-arm prompts (A/B/C/D) for one condition and, where the eval has oracle items, the
four oracle-arm prompts (A'/B'/C'/D') for one item. No API calls.

Run:  .venv/bin/python scripts/selfgeneric_dump.py [--evals a,b] [> results/reports/selfgeneric_dump.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from behavior_prediction import common, elicitation, splits
from behavior_prediction.evals import get_spec
from behavior_prediction.tuning import ACTIVE_EVALS

REVIEW_MODEL = "llama-3.3-70b"     # any pool model: the conditions are the same for all


def dev_conditions(ev: str) -> list[dict]:
    spec = get_spec(ev)
    targets = common.load_json(Path(f"results/{ev}/{REVIEW_MODEL}/targets.json"))["targets"]
    man = splits.load_manifest(ev)
    dev = splits.split_keys(spec, targets, man, "dev")
    return [targets[k]["condition"] for k in sorted(dev)]


def report_prompts(spec, cond) -> dict[str, str]:
    b = elicitation
    if getattr(spec, "scoring_semantics", "") == "bias_contrast":
        from behavior_prediction.evals.discrimeval import bias_prompt
        contrast = spec.condition_contrast(cond)
        if contrast is None:
            return {}
        axis, value = cond["axis"], cond["value"]
        return {arm: bias_prompt(cond["scenario"], axis, value, base_method=m,
                                 implicit=spec.config == "implicit")
                for arm, m in [("A", "self_report"), ("B", "self_report_genmin"),
                               ("C", "self_report_3p"), ("D", "self_report_3p_gen")]}
    return {"A": b.self_report_prompt(spec, cond), "B": b.self_report_genmin_prompt(spec, cond),
            "C": b.self_report_3p_prompt(spec, cond), "D": b.self_report_3p_gen_prompt(spec, cond)}


def oracle_prompts(spec, cond) -> dict[str, str]:
    view = spec.oracle_items(cond, 1) or {}
    items = view.get("items") or []
    if not items or not view.get("protocol_3p"):
        return {}
    it = items[0]
    instr = (elicitation.ORACLE_SIGNED_PREDICTION_INSTRUCTION
             if getattr(spec, "scoring_semantics", "") == "bias_contrast"
             else elicitation.ORACLE_PREDICTION_INSTRUCTION)
    e = elicitation
    return {"A'": e.informed_oracle_prompt(view["protocol"], it["exhibit"], it["ask"], instr),
            "B'": e.informed_oracle_genmin_prompt(view["protocol"], it["exhibit"], it["ask_gen"], instr),
            "C'": e.informed_oracle_3p_prompt(view["protocol_3p"], it["exhibit"], it["ask"], instr),
            "D'": e.informed_oracle_3p_gen_prompt(view["protocol_3p"], it["exhibit"], it["ask_gen"], instr)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default=",".join(ACTIVE_EVALS))
    ap.add_argument("--full-oracle", action="store_true",
                    help="print the whole oracle prompts (default: elide the exhibit block)")
    a = ap.parse_args()
    for ev in a.evals.split(","):
        spec = get_spec(ev)
        conds = dev_conditions(ev)
        print(f"\n{'#' * 100}\n# {ev}: {len(conds)} dev conditions; frame_3p={spec.frame_3p} "
              f"oracle_3p={spec.oracle_3p}\n{'#' * 100}")
        seen = set()
        for c in conds:
            f = spec.frame(c)
            for slot in ("setting", "self_report_situation", "target_action"):
                pair = (getattr(f, slot), getattr(f, slot + "_3p"))
                if pair in seen:
                    continue
                seen.add(pair)
                print(f"\n--- {slot} [{spec.condition_key(c)}]\n2p: {pair[0]}\n3p: {pair[1]}")
        first = next((c for c in conds if report_prompts(spec, c)), None)
        if first is None:
            continue
        print(f"\n===== report arms for {spec.condition_key(first)}")
        for arm, text in report_prompts(spec, first).items():
            print(f"\n[{arm}]\n{text}")
        ops = oracle_prompts(spec, first)
        if ops:
            print(f"\n===== oracle arms for {spec.condition_key(first)}")
            for arm, text in ops.items():
                if not a.full_oracle:
                    head, _, rest = text.partition("----- BEGIN MEASURED ITEM -----")
                    tail = rest.rpartition("----- END MEASURED ITEM -----")[2]
                    text = head + "----- [MEASURED ITEM elided] -----" + tail
                print(f"\n[{arm}]\n{text}")


if __name__ == "__main__":
    main()
