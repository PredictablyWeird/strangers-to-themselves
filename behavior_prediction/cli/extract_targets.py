#!/usr/bin/env python3
"""Extract prediction targets (ground-truth behavior rates) from a finished behavioral run.

Delegates to the selected eval's ``produce_targets`` (``--eval``, default agentic_misalignment),
which reads the eval's behavioral artifacts (``.eval`` logs for agentic_misalignment; the
harness output dir for others) and aggregates the chosen metric per condition. Writes a target
file consumed by ``compare_predictions.py`` / ``evaluate.py`` and prints a summary table.

Examples
--------
    .venv/bin/python extract_targets.py
    .venv/bin/python extract_targets.py --metric classifier_verdict
    .venv/bin/python extract_targets.py --eval propensitybench --pb-output <dir>
"""

from __future__ import annotations

import argparse

from behavior_prediction import common
from behavior_prediction.evals import add_eval_arg, require_runnable, resolve_spec_from_argv


def main() -> int:
    spec = resolve_spec_from_argv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_eval_arg(p)
    spec.add_extract_args(p)
    args = p.parse_args()
    require_runnable([spec], args.force)

    doc = spec.produce_targets(args)  # raises SystemExit on no/ambiguous artifacts
    targets = doc["targets"]
    # Stamp the full model identity (incl. reasoning) and key the path by the --model the run used
    # (a shortcut), not the full string from the logs — so targets land in the same reasoning-aware
    # directory as the predictions and the two can be consistency-checked at score time.
    ident = common.model_identity(args.model)
    doc["model"] = ident["model"]
    doc["model_shortcut"] = ident["shortcut"]
    doc["reasoning"] = ident["reasoning"]
    out = args.out or common.default_targets_out(spec, args.model)
    common.save_json(doc, out)

    # --- summary table (sorted by scenario, then key) ---
    print(f"Model:  {doc['model']}")
    print(f"Metric: {doc['metric']}")
    print(f"{'condition_key':<44}{'count/n':>9}{'rate':>8}{'#logs':>7}")
    print("-" * 72)
    ordered = sorted(
        targets.items(),
        key=lambda kv: (spec.scenarios.index(kv[1]["scenario"])
                        if kv[1]["scenario"] in spec.scenarios else 99, kv[0]),
    )
    for key, t in ordered:
        rate = "n/a" if t["rate"] is None else f"{t['rate']:.3f}"
        count_n = f"{t['count']}/{t['n']}"
        print(f"{key:<44}{count_n:>9}{rate:>8}{len(t.get('logs', [])):>7}")
    print(f"\n{len(targets)} conditions. Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
