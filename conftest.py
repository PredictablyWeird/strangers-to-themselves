"""Pytest configuration for offline tests.

``evals/agentic_misalignment`` imports ``inspect_evals`` at module level (for the
behavioral-run helpers), but the package is not installed in the test environment.
Stub it out in sys.modules before any test collection fires, so the offline tests
that only exercise Frame/EvalSpec logic can import the registry without error.
"""
from __future__ import annotations

import sys
import types


def _stub_inspect_evals() -> None:
    """Insert a minimal stub for inspect_evals.agentic_misalignment.prompt_generator."""
    if "inspect_evals" in sys.modules:
        return  # already present (e.g. installed in CI)

    # Build package hierarchy: inspect_evals -> agentic_misalignment -> prompt_generator
    root = types.ModuleType("inspect_evals")
    sub = types.ModuleType("inspect_evals.agentic_misalignment")
    pg = types.ModuleType("inspect_evals.agentic_misalignment.prompt_generator")

    # The three names imported by evals/agentic_misalignment.py.
    # get_variables_for_condition is called by frame() via goal_statement(); return an empty
    # dict so goal_statement returns "" and the driver-value falls back to the no-goal path.
    # generate_prompts feeds behavioral_prompt()/oracle_items(); return a fixed dummy prompt
    # object so those render offline (the real package produces the actual eval prompts).
    pg.ALIGNMENT_FAKING_WARNING = ""
    pg.generate_prompts = lambda *args, **kwargs: types.SimpleNamespace(
        system_prompt="STUB SYSTEM PROMPT", user_prompt="STUB USER PROMPT",
        email_content="STUB EMAIL CONTENT")
    pg.get_variables_for_condition = lambda *args, **kwargs: {}

    root.agentic_misalignment = sub  # type: ignore[attr-defined]
    sub.prompt_generator = pg  # type: ignore[attr-defined]

    sys.modules["inspect_evals"] = root
    sys.modules["inspect_evals.agentic_misalignment"] = sub
    sys.modules["inspect_evals.agentic_misalignment.prompt_generator"] = pg


_stub_inspect_evals()
