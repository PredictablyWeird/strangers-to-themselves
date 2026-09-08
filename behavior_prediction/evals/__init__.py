"""Eval-adapter registry. ``--eval <name>`` selects one of these specs everywhere."""

from __future__ import annotations

import argparse

from behavior_prediction.evals.agentic_misalignment import AgenticMisalignmentSpec
from behavior_prediction.evals.base import EvalSpec, Frame
from behavior_prediction.evals.capability_mmlu import CapabilityMMLUSpec
from behavior_prediction.evals.discrimeval import DiscrimEvalSpec
from behavior_prediction.evals.mask import MaskSpec
from behavior_prediction.evals.propensitybench import PropensityBenchSpec
from behavior_prediction.evals.reward_hacking import RewardHackingSpec
from behavior_prediction.evals.sycophancy_pushback import SycophancyPushbackSpec
from behavior_prediction.evals.tau2_policy import Tau2PolicySpec

DEFAULT_EVAL = "agentic_misalignment"

REGISTRY: dict[str, EvalSpec] = {
    "agentic_misalignment": AgenticMisalignmentSpec(),
    "capability_mmlu": CapabilityMMLUSpec(),
    "propensitybench": PropensityBenchSpec(benign=False),
    "propensitybench_benign": PropensityBenchSpec(benign=True),
    "discrimeval": DiscrimEvalSpec("explicit"),
    "discrimeval_implicit": DiscrimEvalSpec("implicit"),
    "reward_hacking": RewardHackingSpec(),
    "sycophancy_pushback": SycophancyPushbackSpec(),
    "tau2_policy": Tau2PolicySpec(),
    "tau2_transfer": Tau2PolicySpec("transfer"),
    "mask": MaskSpec(),
    "mask_subdomain": MaskSpec(grain="subdomain"),
    "mask_subdomain_pressure": MaskSpec(grain="subdomain_pressure"),
}

# Eval-name dirs under results/ that must not be mistaken for a model slug by the report.
EVAL_NAMES: set[str] = set(REGISTRY)


def get_spec(name: str) -> EvalSpec:
    if name not in REGISTRY:
        raise SystemExit(f"unknown --eval {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name]


def require_runnable(specs, force: bool) -> None:
    """Guard the eval-running CLIs: refuse to generate/measure any ``broken`` eval unless ``force``.
    Lists the offending evals with their ``broken_reason``. No-op for non-broken evals or when
    forced. (``get_spec`` itself is unguarded, so library/test access is unaffected.)"""
    broken = [s for s in specs if getattr(s, "broken", False)]
    if broken and not force:
        lines = "\n".join(f"  - {s.name}: {s.broken_reason or 'marked broken'}" for s in broken)
        raise SystemExit(
            "refusing to run eval(s) marked broken (pending review / costly):\n"
            f"{lines}\npass --force to run anyway.")


def add_eval_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval", default=DEFAULT_EVAL, choices=sorted(REGISTRY),
                        help=f"which behavior-prediction eval to use (default: {DEFAULT_EVAL})")
    parser.add_argument("--force", action="store_true",
                        help="run even if the selected eval is marked broken")


def resolve_spec_from_argv(argv: list[str] | None = None) -> EvalSpec:
    """Two-phase parse: grab ``--eval`` first so the full parser's choices/defaults can be
    spec-driven. Returns the selected spec."""
    pre = argparse.ArgumentParser(add_help=False)
    add_eval_arg(pre)
    known, _ = pre.parse_known_args(argv)
    return get_spec(known.eval)


__all__ = ["EvalSpec", "Frame", "REGISTRY", "DEFAULT_EVAL", "EVAL_NAMES",
           "get_spec", "require_runnable", "add_eval_arg", "resolve_spec_from_argv"]
