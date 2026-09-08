#!/usr/bin/env python3
"""Single-method elicitation CLI (``bp-elicit --method <name>``).

Consolidates the former standalone runners (``self_report``, ``value``, ``value_aggregate``,
``behavioral_sampling``) behind one entry point. Each method dispatches to
the same single-sourced ``elicitation.run_*`` function the scripts used, so prompts, sampling and
the output schema are unchanged — the report scripts still pick the files up automatically. For
sweeping every method across models/evals with split-aware scoring, use ``bp-benchmark`` instead.

Examples
--------
    bp-elicit --method self_report --runs 10
    bp-elicit --method self_report --runs 10 --honesty-nudge
    bp-elicit --method value --eval propensitybench --runs 10 --context abstract
    bp-elicit --method value --eval agentic_misalignment --aggregate --value-detail
    bp-elicit --method behavioral_sampling --eval propensitybench --model deepseek-v4-flash
"""

from __future__ import annotations

import argparse

from behavior_prediction import elicitation
from behavior_prediction.evals import add_eval_arg, require_runnable, resolve_spec_from_argv

DEFAULT_GRADER = "openrouter/anthropic/claude-sonnet-4"

METHODS = ("self_report", "value", "behavioral_sampling")


def _add_method_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", required=True, choices=METHODS,
                        help="which elicitation method to run")


def _resolve_method_from_argv(argv: list[str] | None = None) -> str | None:
    """Pre-parse ``--method`` so the right per-method argparser can be built (mirrors
    ``resolve_spec_from_argv`` for ``--eval``). Not ``required`` here so ``--help`` still works;
    the real parser enforces it."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", choices=METHODS)
    known, _ = pre.parse_known_args(argv)
    return known.method


def _build_parser(method: str, spec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_method_arg(p)
    add_eval_arg(p)
    if method == "self_report":
        elicitation.add_common_elicit_args(
            p, spec, with_detail=False,
            runs_help="self_report queries per condition (default: 10)")
    elif method == "value":
        elicitation.add_common_elicit_args(
            p, spec, with_detail=False, default_runs=10,
            runs_help="queries per condition; must be even, split over both A/B orders (default: 10)")
        elicitation.add_value_args(p)
    elif method == "behavioral_sampling":
        elicitation.add_common_elicit_args(
            p, spec, with_detail=False,
            runs_help="(unused by behavioral_sampling; use --samples-per-scenario)")
        p.add_argument("--grader-model", default=DEFAULT_GRADER,
                       help=f"model that judges whether a response took the action "
                            f"(default: {DEFAULT_GRADER})")
        p.add_argument("--generator-model", default=None,
                       help="neutral model that writes the one-turn scenarios (default: --grader-model)")
        p.add_argument("--scenarios-per-condition", type=int, default=25,
                       help="how many one-turn scenarios to generate per condition (default: 10)")
        p.add_argument("--samples-per-scenario", type=int, default=2,
                       help="free-text responses sampled from the model per scenario (default: 1)")
        p.add_argument("--generation-retries", type=int, default=3,
                       help="re-request scenarios for conditions the generator returns short, "
                            "up to this many times (default: 3)")
    return p


def main() -> int:
    method = _resolve_method_from_argv()
    spec = resolve_spec_from_argv()
    args = _build_parser(method, spec).parse_args()
    require_runnable([spec], args.force)
    if method == "self_report":
        return elicitation.run_elicitation(
            spec, args, base_method="self_report",
            builder=elicitation.self_report_prompt, with_detail=False)
    if method == "value":
        return elicitation.run_value_elicitation(spec, args)
    if method == "behavioral_sampling":
        return elicitation.run_behavioral_sampling_elicitation(spec, args)
    raise SystemExit(f"unknown method {method!r}")


if __name__ == "__main__":
    raise SystemExit(main())
