#!/usr/bin/env python3
"""Generate behavioral targets by running an eval's behavior measurement.

Delegates to the selected eval's ``run_behavior`` (``--eval``, default agentic_misalignment).
For agentic_misalignment this drives the inspect eval over a condition sweep, writing ``.eval``
logs into ``--log-dir`` for ``extract_targets.py`` to read; other evals run their own harness.
The eval-specific args (``--epochs``, ``--grader-model``, ...) are added by the eval.

Examples
--------
    .venv/bin/python run_sweep.py --sweep goal_value --dry-run
    .venv/bin/python run_sweep.py --sweep goal_value --skip-existing
"""

from __future__ import annotations

import argparse

from behavior_prediction.evals import add_eval_arg, require_runnable, resolve_spec_from_argv


def main() -> int:
    spec = resolve_spec_from_argv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_eval_arg(p)
    spec.add_run_args(p)
    args = p.parse_args()
    require_runnable([spec], args.force)
    return spec.run_behavior(args)


if __name__ == "__main__":
    raise SystemExit(main())
